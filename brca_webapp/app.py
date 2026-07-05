import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "models" / "stacking_improved_results.pkl"
EXAMPLES_DIR = BASE_DIR / "data" / "examples"
DATASET_PATH = PROJECT_ROOT / "datasets" / "TCGA_BRCA_tpm.tsv"
CLINICAL_PATH = PROJECT_ROOT / "datasets" / "brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BASE_DIR))  # exposes brca_webapp/ local modules

# Import the wrapper class before unpickling the model.
from src.train_stacking_improved import _LEWrapper  # noqa: F401
from enhanced_agent import EnhancedClinicalAgent
from kegg_service import get_gene_pathways, get_pathway_image

app = Flask(__name__)

agent = EnhancedClinicalAgent()
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

with MODEL_PATH.open("rb") as f:
    results = pickle.load(f)
    model = results["model"]
    # Strip the "BRCA_" prefix so class labels match the JS SUBTYPE_META keys
    # (e.g. "BRCA_Basal" → "Basal", "BRCA_LumA" → "LumA").
    def _short(label: str) -> str:
        return label.replace("BRCA_", "") if label.startswith("BRCA_") else label
    classes = [_short(c) for c in results["classes"]]

# ---------------------------------------------------------------------------
# Static subtype knowledge base
# ---------------------------------------------------------------------------
SUBTYPE_INFO = {
    "LumA": {
        "full_name": "Luminal A",
        "receptor_status": "ER\u207a / PR\u207a / HER2\u207b",
        "prognosis": "Best \u2014 low proliferation, excellent long-term survival",
        "biomarkers": ["ESR1", "PGR", "GATA3", "FOXA1"],
        "therapy": [
            "Endocrine therapy (Tamoxifen or Aromatase Inhibitor)",
            "CDK4/6 inhibitors (Palbociclib) in advanced disease",
            "Chemotherapy generally avoided if genomic risk is low",
        ],
        "color": "#00b4d8",
        "description": (
            "The most common subtype (~40% of BRCA). Driven by oestrogen/progesterone "
            "signalling. Slow-growing; responds well to hormone therapy."
        ),
    },
    "LumB": {
        "full_name": "Luminal B",
        "receptor_status": "ER\u207a / PR\u207a or \u207b / HER2\u207a or \u207b",
        "prognosis": "Intermediate \u2014 higher proliferation than LumA",
        "biomarkers": ["ESR1", "PGR", "ERBB2", "MKI67"],
        "therapy": [
            "Endocrine therapy + Chemotherapy",
            "Anti-HER2 therapy (Trastuzumab) if HER2\u207a",
            "CDK4/6 inhibitors",
        ],
        "color": "#4361ee",
        "description": (
            "Similar to Luminal A but with higher Ki-67 proliferation. More heterogeneous; "
            "often requires combined endocrine + chemotherapy."
        ),
    },
    "Her2": {
        "full_name": "HER2-Enriched",
        "receptor_status": "ER\u207b / PR\u207b / HER2\u207a",
        "prognosis": "Intermediate-poor \u2014 aggressive but targetable",
        "biomarkers": ["ERBB2", "GRB7", "PGAP3", "STARD3"],
        "therapy": [
            "Targeted anti-HER2: Trastuzumab + Pertuzumab",
            "Antibody-drug conjugate: T-DM1, T-DXd",
            "Chemotherapy backbone (Taxane + Carboplatin)",
            "Neoadjuvant chemotherapy before surgery",
        ],
        "color": "#f72585",
        "description": (
            "Characterised by HER2 gene amplification (~15-20% of BRCA). Historically aggressive; "
            "dramatically improved outcomes with targeted therapy."
        ),
    },
    "Basal": {
        "full_name": "Basal-like (Triple-Negative)",
        "receptor_status": "ER\u207b / PR\u207b / HER2\u207b",
        "prognosis": "Poor \u2014 highest recurrence risk, especially within first 5 years",
        "biomarkers": ["TP53", "BRCA1", "KRT5", "KRT14", "EGFR"],
        "therapy": [
            "Chemotherapy: Anthracycline + Taxane backbone",
            "Immunotherapy: Pembrolizumab (PD-L1\u207a cases)",
            "PARP inhibitors (Olaparib/Talazoparib) if BRCA1/2 mutated",
            "Sacituzumab govitecan (ADC) in metastatic disease",
        ],
        "color": "#ff4800",
        "description": (
            "Most aggressive subtype (~15-20%). No hormone receptors \u2014 cannot use hormone therapy. "
            "Highly responsive to chemo but relapse risk is high."
        ),
    },
    "Normal": {
        "full_name": "Normal-like",
        "receptor_status": "Mixed / unclear",
        "prognosis": "Generally favourable \u2014 similar to Luminal A",
        "biomarkers": ["ADIPOQ", "DCN", "PDPN"],
        "therapy": [
            "Often treated similarly to Luminal A",
            "Endocrine therapy if ER\u207a",
            "Clinical trial participation recommended",
        ],
        "color": "#2dc653",
        "description": (
            "Rare subtype (~5%) that resembles normal breast tissue expression patterns. "
            "May reflect tumour purity or adipose contamination."
        ),
    },
}


# ---------------------------------------------------------------------------
# Gene name resolver (mygene, cached)
# ---------------------------------------------------------------------------
_gene_name_cache: dict[str, dict] = {}  # ensembl_id → {symbol, name}

def _resolve_gene_names(ensembl_ids: list[str]) -> dict[str, dict]:
    """Batch-resolve Ensembl IDs → {symbol, name} via mygene. Results are cached."""
    missing = [eid for eid in ensembl_ids if eid not in _gene_name_cache]
    if missing:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            results = mg.querymany(
                missing, scopes="ensembl.gene",
                fields="symbol,name", species="human", verbose=False,
            )
            for hit in results:
                eid = hit.get("query", "")
                _gene_name_cache[eid] = {
                    "symbol": hit.get("symbol", eid),
                    "name":   hit.get("name", ""),
                }
        except Exception:  # noqa: BLE001
            # Fallback: use raw Ensembl IDs if mygene unavailable
            for eid in missing:
                _gene_name_cache[eid] = {"symbol": eid, "name": ""}
    return {eid: _gene_name_cache.get(eid, {"symbol": eid, "name": ""}) for eid in ensembl_ids}


# ---------------------------------------------------------------------------
# Top gene driver computation
# ---------------------------------------------------------------------------

def compute_top_gene_drivers(
    stacking_model, prepared_df: pd.DataFrame, n_top: int = 10,
) -> list[dict]:
    """Extract per-sample top gene drivers using base-learner feature importances.

    Steps:
      1. Average feature_importances_ across RF, XGBoost, LightGBM.
      2. Multiply by each sample's normalised expression (z-score) to get
         per-sample contribution scores.
      3. Return top-N genes sorted by absolute contribution for the first sample.
    """
    feature_names = list(getattr(stacking_model, "feature_names_in_", []))
    if not feature_names:
        return []

    # ── Collect base-learner importances ───────────────────────────────────
    importances = []
    for est in stacking_model.estimators_:
        if hasattr(est, "feature_importances_"):
            importances.append(est.feature_importances_)
        elif hasattr(est, "estimator") and hasattr(est.estimator, "feature_importances_"):
            importances.append(est.estimator.feature_importances_)

    if not importances:
        return []

    # Average across base learners and normalise to [0, 1]
    global_imp = np.mean(importances, axis=0)
    imp_max = global_imp.max()
    if imp_max > 0:
        global_imp_norm = global_imp / imp_max
    else:
        return []

    # ── Per-sample weighted importance (first sample) ─────────────────────
    sample = prepared_df.iloc[0].values.astype(float)
    # Z-score normalise expression values
    mean_val = np.nanmean(sample)
    std_val  = np.nanstd(sample)
    if std_val > 0:
        z_scores = (sample - mean_val) / std_val
    else:
        z_scores = sample - mean_val

    # Contribution = global_importance × |z-score of expression|
    contributions = global_imp_norm * np.abs(z_scores)

    # Top N indices by contribution
    top_indices = np.argsort(contributions)[::-1][:n_top]

    # ── Resolve gene names ────────────────────────────────────────────────
    # Strip version suffixes (.2, .15) for mygene lookup
    top_ensembl_raw = [feature_names[i] for i in top_indices]
    top_ensembl_clean = [eid.split(".")[0] for eid in top_ensembl_raw]
    gene_info = _resolve_gene_names(top_ensembl_clean)

    # ── Build result list ─────────────────────────────────────────────────
    drivers = []
    max_contribution = contributions[top_indices[0]] if len(top_indices) > 0 else 1.0
    for rank, idx in enumerate(top_indices, 1):
        eid_raw   = feature_names[idx]
        eid_clean = eid_raw.split(".")[0]
        info      = gene_info.get(eid_clean, {})
        expr_val  = float(sample[idx])

        drivers.append({
            "rank":             rank,
            "ensembl_id":       eid_raw,
            "gene_symbol":      info.get("symbol", eid_clean),
            "gene_name":        info.get("name", ""),
            "importance_score":  float(contributions[idx]),
            "importance_pct":   float(contributions[idx] / max_contribution) if max_contribution > 0 else 0.0,
            "global_importance": float(global_imp_norm[idx]),
            "expression_value":  round(expr_val, 2),
            "expression_level":  (
                "High" if z_scores[idx] > 1.5 else
                "Low"  if z_scores[idx] < -1.5 else
                "Normal"
            ),
        })

    return drivers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_expected_columns():
    return list(getattr(model, "feature_names_in_", []))


def prepare_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = get_expected_columns()
    if not expected_columns:
        if df.empty:
            raise ValueError("The uploaded CSV is empty.")
        return df

    if df.empty:
        raise ValueError("The uploaded CSV is empty.")

    missing_columns = [c for c in expected_columns if c not in df.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:5])
        raise ValueError(f"Missing required gene columns. Example missing: {preview}")

    aligned = df.reindex(columns=expected_columns)
    if aligned.empty:
        raise ValueError("The uploaded CSV has no usable rows.")
    return aligned


def build_prediction_response(prepared_df: pd.DataFrame, original_df: pd.DataFrame) -> dict:
    """Run model inference and return a rich response dict."""
    probabilities = model.predict_proba(prepared_df)
    raw_predictions = model.predict(prepared_df)
    # Normalise predictions to short labels (strip BRCA_ prefix if present)
    predictions = [_short(str(p)) for p in raw_predictions]

    samples = []
    for i in range(len(predictions)):
        prob_dict = {cls: float(probabilities[i][j]) for j, cls in enumerate(classes)}
        samples.append(
            {
                "subtype": predictions[i],
                "confidence": float(probabilities[i].max()),
                "probabilities": prob_dict,
            }
        )

    # Preview: first 5 rows, first 6 columns
    preview_cols = list(original_df.columns[:6])
    preview_data = (
        original_df[preview_cols].head(5).fillna(0).round(4).to_dict(orient="records")
    )

    # ── Top gene drivers (per-sample feature importance) ──────────────────
    top_gene_drivers = compute_top_gene_drivers(model, prepared_df, n_top=10)

    return {
        "samples": samples,
        "subtype": predictions[0],
        "confidence": float(probabilities[0].max()),
        "probabilities": {cls: float(probabilities[0][j]) for j, cls in enumerate(classes)},
        "detected_biomarkers": SUBTYPE_INFO.get(predictions[0], {}).get("biomarkers", []),
        "classes": classes,
        "rows_processed": int(len(prepared_df)),
        "preview": preview_data,
        "preview_cols": preview_cols,
        "top_gene_drivers": top_gene_drivers,
        "evidence_warning": None,   # filled in by agent layer
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    example_files = sorted(p.stem for p in EXAMPLES_DIR.glob("*.csv"))
    return render_template("index.html", example_files=example_files, subtype_info=SUBTYPE_INFO)


@app.route("/api/subtypes")
def api_subtypes():
    return jsonify(SUBTYPE_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"error": "Please choose a CSV file before predicting."}), 400

    try:
        df = pd.read_csv(file)
        prepared_df = prepare_input_frame(df)
        response = build_prediction_response(prepared_df, df)
        
        # Run agentic layer and merge into response (preserves model output fields)
        agent_result = agent.run_full_agent(response)
        response["agent_analysis"]  = agent_result
        response["evidence_warning"] = agent_result.get("evidence_warning")
        return jsonify(response)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/demo/<name>")
def demo(name):
    demo_path = EXAMPLES_DIR / f"{name}.csv"
    if not demo_path.exists():
        return jsonify({"error": "Demo file not found. Try generating demo data first."}), 404

    try:
        df = pd.read_csv(demo_path)
        prepared_df = prepare_input_frame(df)
        response = build_prediction_response(prepared_df, df)
        # Run agentic layer for demo too
        agent_result = agent.run_full_agent(response)
        response["agent_analysis"]  = agent_result
        response["evidence_warning"] = agent_result.get("evidence_warning")
        return jsonify(response)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/api/generate-demo")
def generate_demo():
    """Sample one row per subtype from TCGA and write CSVs into examples/."""
    if not DATASET_PATH.exists() or not CLINICAL_PATH.exists():
        return jsonify(
            {
                "error": (
                    "TCGA dataset not found in datasets/. "
                    "Upload your own CSV to use the classifier."
                ),
                "available": [],
            }
        )

    try:
        # Read using absolute paths so this works regardless of cwd.
        from src.data_preprocessing import preprocess  # noqa: PLC0415

        data   = pd.read_csv(DATASET_PATH,   sep="\t")
        clinic = pd.read_csv(CLINICAL_PATH,  sep="\t")
        df     = preprocess(data, clinic)
        EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

        generated = []
        for subtype in df["Subtype"].unique():
            sample = (
                df[df["Subtype"] == subtype]
                .drop(columns=["Subtype"])
                .sample(1, random_state=42)
            )
            # Use the short subtype key for the filename (strip BRCA_ prefix)
            short = _short(str(subtype))
            fname = EXAMPLES_DIR / f"{short.lower()}_sample.csv"
            sample.to_csv(fname, index=False)
            generated.append(fname.stem)

        return jsonify({"generated": generated, "available": generated})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/api/demo-list")
def demo_list():
    files = sorted(p.stem for p in EXAMPLES_DIR.glob("*.csv"))
    return jsonify({"files": files})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True)
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message."}), 400
    try:
        reply = agent.chat(
            message=message,
            history=body.get("history", []),
            context=body.get("context"),
        )
        return jsonify({"reply": reply})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# KEGG Pathway API
# ---------------------------------------------------------------------------

@app.route("/api/kegg/pathways/<gene_symbol>")
def api_kegg_pathways(gene_symbol: str):
    """Return KEGG pathway list for a gene symbol."""
    try:
        pathways = get_gene_pathways(gene_symbol)
        if pathways:
            return jsonify({"gene": gene_symbol, "pathways": pathways})
        return jsonify({
            "gene": gene_symbol,
            "pathways": [],
            "message": f"No KEGG pathways found for {gene_symbol}",
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/api/kegg/pathway-image/<pathway_id>")
def api_kegg_pathway_image(pathway_id: str):
    """Proxy a KEGG pathway PNG image to avoid CORS issues."""
    try:
        img_bytes = get_pathway_image(pathway_id)
        if img_bytes:
            return Response(img_bytes, mimetype="image/png")
        return jsonify({"error": "Pathway image not found"}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    # The XGBoost/LightGBM stacking model is not thread-safe across Flask threads.
    # Set threaded=False to prevent C++ segmentation faults on sequential requests.
    app.run(debug=True, threaded=False)

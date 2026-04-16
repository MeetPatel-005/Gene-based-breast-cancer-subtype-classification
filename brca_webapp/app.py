import pickle
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = BASE_DIR / "models" / "stacking_improved_results.pkl"
EXAMPLES_DIR = BASE_DIR / "data" / "examples"
DATASET_PATH = PROJECT_ROOT / "datasets" / "TCGA_BRCA_tpm.tsv"
CLINICAL_PATH = PROJECT_ROOT / "datasets" / "brca_tcga_pan_can_atlas_2018_clinical_data_filtered.tsv"

sys.path.insert(0, str(PROJECT_ROOT))

# Import the wrapper class before unpickling the model.
from src.train_stacking_improved import _LEWrapper  # noqa: F401

app = Flask(__name__)
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

    return {
        "samples": samples,
        "subtype": predictions[0],
        "confidence": float(probabilities[0].max()),
        "probabilities": {cls: float(probabilities[0][j]) for j, cls in enumerate(classes)},
        "classes": classes,
        "rows_processed": int(len(prepared_df)),
        "preview": preview_data,
        "preview_cols": preview_cols,
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
        return jsonify(build_prediction_response(prepared_df, df))
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
        return jsonify(build_prediction_response(prepared_df, df))
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


if __name__ == "__main__":
    # The XGBoost/LightGBM stacking model is not thread-safe across Flask threads.
    # Set threaded=False to prevent C++ segmentation faults on sequential requests.
    app.run(debug=True, threaded=False)

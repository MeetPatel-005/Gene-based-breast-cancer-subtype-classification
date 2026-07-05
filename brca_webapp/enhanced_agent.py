"""
enhanced_agent.py
EnhancedClinicalAgent — subclasses the read-only ClinicalAgent (project root src/)
and adds:
  Step 2.5 : EvidenceRetriever  (PubMed + ClinicalTrials.gov)
  Step 3   : probabilistic therapy routing (llama3, configurable via OLLAMA_MODEL env var)

Graceful fallback: if internet fetch fails, returns static knowledge + evidence_warning.
"""
import json
import os
import sys
from pathlib import Path

# ── Ensure both brca_webapp/ and project root are importable ─────────────────
_BASE_DIR    = Path(__file__).resolve().parent          # brca_webapp/
_PROJECT_ROOT = _BASE_DIR.parent                         # project root

for _p in (str(_BASE_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Now we can import from project-root src/ AND from brca_webapp/
from src.clinic_agent import ClinicalAgent              # noqa: E402 (root src)
from evidence_retriever import EvidenceRetriever        # noqa: E402 (brca_webapp)
from enhanced_prompts import (                          # noqa: E402
    EVIDENCE_EXTRACT_PROMPT,
    PROBABILISTIC_ROUTE_PROMPT,
    GENE_DRIVER_ANALYSIS_PROMPT,
)


def _extract_json(text: str) -> dict | None:
    """Pull the first {...} block out of an LLM response string."""
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


class EnhancedClinicalAgent(ClinicalAgent):
    """
    4-step agentic loop:
      1. analyze()           — biomarker alignment check
      2. synthesize()        — risk level + prognosis
      2.5 retrieve_evidence()— PubMed + ClinicalTrials.gov
      3. route_probabilistic()— therapy plan with response probabilities
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        super().__init__(ollama_url)
        # Use llama3 by default (better JSON extraction); override via env var
        self.model = os.environ.get("OLLAMA_MODEL", "llama3")
        self.evidence_retriever = EvidenceRetriever()

    # ── Step 2.5 : evidence extraction ───────────────────────────────────────

    def retrieve_evidence(self, subtype: str) -> tuple[dict, str | None]:
        """
        Fetches raw evidence then uses LLM to extract structured statistics.
        Returns (structured_evidence_dict, warning_string_or_None).
        """
        raw = self.evidence_retriever.fetch(subtype)

        if raw.get("error"):
            warning = (
                f"Live evidence fetch failed — showing knowledge-based estimates. "
                f"({raw['error']})"
            )
            return {"therapies": [], "summary": "", "trials": []}, warning

        # LLM extraction of response rates from abstracts
        abstracts_text = "\n\n---\n\n".join(raw["abstracts"][:3])
        structured = {"therapies": [], "summary": "", "trials": raw.get("trials", [])}

        if abstracts_text.strip():
            prompt = EVIDENCE_EXTRACT_PROMPT.format(
                subtype=subtype,
                abstracts=abstracts_text[:3500],   # stay within token budget
            )
            try:
                llm_response = self._query_llm(prompt)
                parsed = _extract_json(llm_response)
                if parsed:
                    structured.update(parsed)
                    structured["trials"] = raw.get("trials", [])
            except Exception:  # noqa: BLE001
                pass  # fallback to empty structured evidence

        return structured, None

    # ── Step 2.7 : gene driver clinical annotation ───────────────────────────

    def analyze_gene_drivers(self, subtype: str, gene_drivers: list[dict]) -> dict:
        """Ask LLM to provide clinical annotations for the top gene drivers."""
        if not gene_drivers:
            return {"gene_annotations": {}, "summary": ""}

        # Build a compact summary for the LLM
        drivers_for_llm = [
            {"gene_symbol": g["gene_symbol"], "gene_name": g["gene_name"],
             "expression_level": g["expression_level"], "rank": g["rank"]}
            for g in gene_drivers
        ]
        import json as _json
        prompt = GENE_DRIVER_ANALYSIS_PROMPT.format(
            subtype=subtype,
            gene_drivers_json=_json.dumps(drivers_for_llm, indent=2)[:2000],
        )
        try:
            response = self._query_llm(prompt)
            parsed = _extract_json(response)
            if parsed:
                return parsed
        except Exception:  # noqa: BLE001
            pass
        return {"gene_annotations": {}, "summary": ""}

    # ── Step 3 : probabilistic routing ───────────────────────────────────────

    def route_probabilistic(
        self, subtype: str, risk_level: str, evidence: dict
    ) -> dict:
        """Generate therapy plan with response probabilities grounded in evidence."""
        evidence_json = json.dumps(evidence.get("therapies", []), indent=2) or "[]"
        trials_json   = json.dumps(evidence.get("trials",    []), indent=2) or "[]"

        prompt = PROBABILISTIC_ROUTE_PROMPT.format(
            subtype=subtype,
            risk_level=risk_level,
            evidence_json=evidence_json[:2000],
            trials_json=trials_json[:1000],
        )

        try:
            response = self._query_llm(prompt)
            parsed   = _extract_json(response)
            if parsed:
                return parsed
        except Exception:  # noqa: BLE001
            pass

        # Hard fallback — delegate to parent's static route()
        return self.route(subtype, risk_level)

    # ── Full 4-step loop ──────────────────────────────────────────────────────

    def run_full_agent(self, model_output: dict) -> dict:
        """Execute the enhanced 4-step agentic pipeline."""
        subtype    = model_output["subtype"]
        confidence = model_output["confidence"]
        probs      = model_output["probabilities"]
        biomarkers = model_output.get("detected_biomarkers", [])
        gene_drivers = model_output.get("top_gene_drivers", [])

        # Step 1 — biomarker alignment
        analysis = self.analyze(subtype, confidence, probs, biomarkers)

        # Step 2 — clinical synthesis
        summary    = self.synthesize(analysis, subtype)
        risk_level = summary.get("risk_level", "intermediate")

        # Step 2.5 — live evidence retrieval
        evidence, evidence_warning = self.retrieve_evidence(subtype)

        # Step 2.7 — gene driver clinical annotation
        gene_analysis = self.analyze_gene_drivers(subtype, gene_drivers)

        # Step 3 — probabilistic routing
        therapy = self.route_probabilistic(subtype, risk_level, evidence)

        return {
            "subtype":          subtype,
            "confidence":       confidence,
            "analysis":         analysis,
            "clinical_summary": summary,
            "therapy_routes":   therapy,
            "evidence_sources": therapy.get("evidence_sources", []),
            "evidence_warning": evidence_warning,
            "gene_analysis":    gene_analysis,
            "top_gene_drivers": gene_drivers,
        }

    # ── Chat (pass evidence + gene drivers into context) ──────────────────────

    def chat(self, message: str, history: list, context: dict = None) -> str:
        """Override parent chat to inject top gene driver context."""
        if context and context.get("top_gene_drivers"):
            drivers = context["top_gene_drivers"]
            gene_lines = []
            for g in drivers[:10]:
                gene_lines.append(
                    f"  #{g['rank']} {g['gene_symbol']} ({g['gene_name']}) — "
                    f"expression: {g['expression_level']} ({g['expression_value']} TPM), "
                    f"importance: {g['importance_pct']:.0%}"
                )
            gene_block = "\n".join(gene_lines)

            # Inject into context so parent's chat sees it
            if "agent_analysis" not in context:
                context["agent_analysis"] = {}
            aa = context["agent_analysis"]
            existing_explanation = aa.get("analysis", {}).get("explanation", "Not available")
            gene_annotation_summary = aa.get("gene_analysis", {}).get("summary", "")
            aa.setdefault("analysis", {})["explanation"] = (
                f"{existing_explanation}\n\n"
                f"TOP GENE DRIVERS for this patient (from model feature importance):\n"
                f"{gene_block}\n"
                f"{gene_annotation_summary}"
            )

        return super().chat(message, history, context)

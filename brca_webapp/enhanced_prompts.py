"""
enhanced_prompts.py
Two new prompts for the probabilistic evidence pipeline:
  - EVIDENCE_EXTRACT_PROMPT : LLM extracts response rates from PubMed abstracts
  - PROBABILISTIC_ROUTE_PROMPT : LLM builds a probability-scored therapy plan from evidence
"""

EVIDENCE_EXTRACT_PROMPT = """
You are a clinical data extraction specialist. Below are PubMed abstracts about
{subtype} breast cancer treatment. Your task is to extract structured statistics.

ABSTRACTS:
{abstracts}

Instructions:
- Read every abstract carefully.
- Extract each distinct therapy mentioned along with its response/survival statistics.
- If a metric is not explicitly stated, set it to null — do NOT guess.
- Output ONLY valid JSON. No prose, no code fences, no markdown.

Required format:
{{
  "therapies": [
    {{
      "name": "short therapy name (e.g. Trastuzumab + Pertuzumab)",
      "response_rate": 0.72,
      "pfs_months": 18.7,
      "5yr_survival": 0.81,
      "trial_name": "CLEOPATRA",
      "pubmed_id": "12345678"
    }}
  ],
  "summary": "One sentence summarising the overall evidence quality and key finding."
}}
"""

PROBABILISTIC_ROUTE_PROMPT = """
You are a clinical oncology AI assistant. Generate a probabilistic treatment plan.

Patient details:
- Breast cancer subtype : {subtype}
- Risk level            : {risk_level}

Evidence extracted from clinical literature:
{evidence_json}

Active / completed clinical trials:
{trials_json}

Instructions:
- Use the evidence above to assign realistic probability_of_response values.
- If no evidence was found for a therapy, set probability_of_response to null.
- List the single best therapy as primary_therapy, then up to 3 alternatives.
- Output ONLY valid JSON. No prose, no code fences.

Required format:
{{
  "primary_therapy": {{
    "name": "Endocrine therapy + CDK4/6 inhibitor (Palbociclib)",
    "probability_of_response": 0.74,
    "5yr_survival_rate": 0.82,
    "evidence_basis": "PALOMA-2 / MONARCH-3 trials"
  }},
  "alternatives": [
    {{
      "name": "Endocrine therapy alone",
      "probability_of_response": 0.61,
      "5yr_survival_rate": null,
      "evidence_basis": "Standard of care guideline"
    }}
  ],
  "evidence_sources": ["PALOMA-2 trial", "MONARCH-3 trial"],
  "rationale": "2–3 sentence clinical rationale citing the evidence above."
}}
"""

GENE_DRIVER_ANALYSIS_PROMPT = """
You are a molecular oncology specialist. Provide brief clinical annotations for genes
that are driving a {subtype} breast cancer classification in a specific patient.

Top gene drivers (ranked by contribution to the model's prediction):
{gene_drivers_json}

Instructions:
- For each gene, provide a 1-sentence clinical annotation explaining WHY this gene
  is relevant to {subtype} breast cancer (e.g. its pathway role, oncogene/tumour
  suppressor status, known mutations in this subtype, therapeutic relevance).
- If you do not recognise a gene symbol, say "Role in breast cancer unclear".
- Output ONLY valid JSON. No prose, no code fences.

Required format:
{{
  "gene_annotations": {{
    "ESR1": "Oestrogen receptor 1 — primary driver of Luminal subtypes; target of Tamoxifen/AI therapy.",
    "ERBB2": "HER2 oncogene; amplification defines HER2+ subtype; target of Trastuzumab."
  }},
  "summary": "One sentence summarising the overall gene expression pattern and what it implies for this patient."
}}
"""

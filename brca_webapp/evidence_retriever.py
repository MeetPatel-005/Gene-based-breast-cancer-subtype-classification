"""
evidence_retriever.py
Fetches real clinical trial data from PubMed and ClinicalTrials.gov for a given
breast cancer subtype, then returns a raw bundle for the agent to parse.
Results are cached in-memory for 24 hours per subtype.
"""
import requests
from datetime import datetime, timedelta


# ── Per-subtype PubMed search queries ────────────────────────────────────────
_PUBMED_QUERIES = {
    "LumA":   "luminal A breast cancer hormone therapy response rate randomized trial",
    "LumB":   "luminal B breast cancer CDK4/6 inhibitor response rate randomized trial",
    "Her2":   "HER2 positive breast cancer trastuzumab pertuzumab response rate randomized trial",
    "Basal":  "triple negative breast cancer chemotherapy immunotherapy pembrolizumab response rate randomized",
    "Normal": "normal-like breast cancer endocrine therapy treatment outcome",
}

_TRIAL_CONDITIONS = {
    "LumA":   "Luminal A breast cancer",
    "LumB":   "Luminal B breast cancer",
    "Her2":   "HER2 positive breast cancer",
    "Basal":  "Triple negative breast cancer",
    "Normal": "breast cancer",
}

_PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_TRIALS_SEARCH = "https://clinicaltrials.gov/api/v2/studies"


class EvidenceRetriever:
    """Fetches PubMed abstracts + ClinicalTrials.gov data for breast cancer subtypes."""

    def __init__(self, cache_ttl_hours: int = 24):
        self._cache: dict = {}
        self._ttl = timedelta(hours=cache_ttl_hours)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self, subtype: str) -> dict:
        """
        Returns an evidence bundle dict:
        {
            "abstracts": [str, ...],   # raw abstract texts from PubMed
            "trials":    [dict, ...],  # trial metadata from ClinicalTrials.gov
            "fetched_at": str | None,
            "error": str | None
        }
        Falls back gracefully on any network / parsing error.
        """
        if self._is_fresh(subtype):
            return self._cache[subtype][1]

        bundle = self._do_fetch(subtype)
        self._cache[subtype] = (datetime.now(), bundle)
        return bundle

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_fresh(self, subtype: str) -> bool:
        if subtype not in self._cache:
            return False
        ts, _ = self._cache[subtype]
        return (datetime.now() - ts) < self._ttl

    def _do_fetch(self, subtype: str) -> dict:
        try:
            abstracts = self._fetch_pubmed(subtype)
            trials    = self._fetch_trials(subtype)
            return {
                "abstracts":  abstracts,
                "trials":     trials,
                "fetched_at": datetime.now().isoformat(),
                "error":      None,
            }
        except requests.exceptions.ConnectionError:
            return self._error_bundle("No internet connection — could not reach PubMed/ClinicalTrials.gov.")
        except requests.exceptions.Timeout:
            return self._error_bundle("Evidence fetch timed out. Check internet connection.")
        except Exception as exc:  # noqa: BLE001
            return self._error_bundle(str(exc))

    @staticmethod
    def _error_bundle(msg: str) -> dict:
        return {"abstracts": [], "trials": [], "fetched_at": None, "error": msg}

    def _fetch_pubmed(self, subtype: str) -> list[str]:
        """Search PubMed and return up to 5 abstract texts."""
        query = _PUBMED_QUERIES.get(subtype, f"{subtype} breast cancer treatment response rate")

        # Step 1 — search for IDs
        search = requests.get(
            _PUBMED_SEARCH,
            params={"db": "pubmed", "term": query, "retmax": 5,
                    "retmode": "json", "sort": "relevance"},
            timeout=10,
        )
        search.raise_for_status()
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Step 2 — fetch abstract text
        fetch = requests.get(
            _PUBMED_FETCH,
            params={"db": "pubmed", "id": ",".join(ids),
                    "rettype": "abstract", "retmode": "text"},
            timeout=15,
        )
        fetch.raise_for_status()

        # Split multi-abstract response into individual chunks
        raw = fetch.text
        abstracts, current = [], []
        for line in raw.splitlines():
            stripped = line.strip()
            # New abstract starts with an integer index followed by a period
            if stripped and stripped[0].isdigit() and stripped.find(".") in range(1, 4):
                if current:
                    abstracts.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            abstracts.append("\n".join(current).strip())

        return [a for a in abstracts if len(a) > 80][:5]

    def _fetch_trials(self, subtype: str) -> list[dict]:
        """Query ClinicalTrials.gov v2 for completed trials matching the subtype."""
        condition = _TRIAL_CONDITIONS.get(subtype, "breast cancer")
        resp = requests.get(
            _TRIALS_SEARCH,
            params={
                "query.cond":          condition,
                "filter.overallStatus": "COMPLETED",
                "pageSize":            5,
                "format":              "json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        studies = resp.json().get("studies", [])

        results = []
        for s in studies:
            proto  = s.get("protocolSection", {})
            ident  = proto.get("identificationModule", {})
            design = proto.get("designModule", {})
            phases = design.get("phases", [])
            results.append({
                "nct_id": ident.get("nctId", ""),
                "title":  ident.get("briefTitle", "Untitled"),
                "phase":  phases[0] if phases else "N/A",
            })

        return results[:5]

"""KEGG REST API client for pathway lookup and image retrieval.

Provides functions to:
  1. Find the human KEGG gene ID (hsa:xxxx) for a gene symbol.
  2. List pathways associated with that gene.
  3. Proxy pathway PNG images back to the caller.

All results are cached in-memory so repeated queries are instant.
A simple throttle ensures we stay below KEGG's ~3 req/s limit.
"""

from __future__ import annotations

import io
import time
import threading
from dataclasses import dataclass, field

import requests

# ── Constants ────────────────────────────────────────────────────────────────
KEGG_BASE = "https://rest.kegg.jp"
_MIN_INTERVAL = 0.35  # seconds between requests (≈ 2.8 req/s, safely under 3)
_REQUEST_TIMEOUT = 15  # seconds


# ── Throttle ─────────────────────────────────────────────────────────────────
class _Throttle:
    """Simple thread-safe rate limiter."""

    def __init__(self, min_interval: float = _MIN_INTERVAL):
        self._min_interval = min_interval
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


_throttle = _Throttle()


# ── Cache ────────────────────────────────────────────────────────────────────
@dataclass
class _Cache:
    """In-memory cache for KEGG lookups."""
    gene_to_kegg_id: dict[str, str | None] = field(default_factory=dict)
    gene_to_pathways: dict[str, list[dict]] = field(default_factory=dict)
    pathway_images: dict[str, bytes | None] = field(default_factory=dict)


_cache = _Cache()


# ── Internal helpers ─────────────────────────────────────────────────────────

def _kegg_get(path: str) -> requests.Response | None:
    """Throttled GET against the KEGG REST API.  Returns None on failure."""
    _throttle.wait()
    try:
        resp = requests.get(f"{KEGG_BASE}{path}", timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp
    except requests.RequestException:
        pass
    return None


# ── Public API ───────────────────────────────────────────────────────────────

def find_kegg_gene_id(gene_symbol: str) -> str | None:
    """Map a human gene symbol (e.g. 'TP53') → KEGG gene ID ('hsa:7157').

    Returns the first human (hsa:) hit, or None if not found.
    """
    symbol_upper = gene_symbol.strip().upper()
    if symbol_upper in _cache.gene_to_kegg_id:
        return _cache.gene_to_kegg_id[symbol_upper]

    resp = _kegg_get(f"/find/genes/{symbol_upper}")
    if resp is None:
        _cache.gene_to_kegg_id[symbol_upper] = None
        return None

    # Parse tab-separated results; each line is "kegg_id\tdescription"
    # We want the first human entry whose description starts with our symbol.
    for line in resp.text.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        kegg_id = parts[0].strip()
        desc = parts[1].strip()
        # Only accept human (hsa:) entries where the symbol matches exactly
        if kegg_id.startswith("hsa:"):
            # Description format: "SYMBOL, ALT; full name [EC:...]"
            # or "SYMBOL; full name"
            desc_symbols = desc.split(";")[0]  # "TP53, LFS1"
            symbols_in_desc = [s.strip().upper() for s in desc_symbols.split(",")]
            if symbol_upper in symbols_in_desc:
                _cache.gene_to_kegg_id[symbol_upper] = kegg_id
                return kegg_id

    _cache.gene_to_kegg_id[symbol_upper] = None
    return None


def _ensure_pathway_names() -> dict[str, str]:
    """Fetch and cache all human pathway names in one call.

    Uses ``/list/pathway/hsa`` which returns every human pathway
    in a single response — far cheaper than individual ``/get`` calls.
    """
    if hasattr(_cache, "_pathway_names"):
        return _cache._pathway_names

    names: dict[str, str] = {}
    resp = _kegg_get("/list/pathway/hsa")
    if resp is not None:
        for line in resp.text.strip().splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2:
                pid = parts[0].strip().replace("path:", "")
                raw_name = parts[1].strip()
                # Remove species suffix " - Homo sapiens (human)"
                if " - " in raw_name:
                    raw_name = raw_name.rsplit(" - ", 1)[0].strip()
                names[pid] = raw_name

    _cache._pathway_names = names  # type: ignore[attr-defined]
    return names


def get_gene_pathways(gene_symbol: str) -> list[dict]:
    """Return a list of {'pathway_id': str, 'name': str} for a gene symbol.

    Returns an empty list if the gene is not in KEGG or has no pathways.
    """
    symbol_upper = gene_symbol.strip().upper()
    if symbol_upper in _cache.gene_to_pathways:
        return _cache.gene_to_pathways[symbol_upper]

    kegg_id = find_kegg_gene_id(symbol_upper)
    if kegg_id is None:
        _cache.gene_to_pathways[symbol_upper] = []
        return []

    # Step 1: get pathway IDs linked to this gene
    resp = _kegg_get(f"/link/pathway/{kegg_id}")
    if resp is None or not resp.text.strip():
        _cache.gene_to_pathways[symbol_upper] = []
        return []

    pathway_ids = []
    for line in resp.text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            pid = parts[1].strip()  # e.g. "path:hsa04110"
            pid = pid.replace("path:", "")
            pathway_ids.append(pid)

    if not pathway_ids:
        _cache.gene_to_pathways[symbol_upper] = []
        return []

    # Step 2: resolve pathway names from the cached global list (1 API call)
    pathway_names = _ensure_pathway_names()
    pathways = []
    for pid in pathway_ids:
        name = pathway_names.get(pid, pid)
        pathways.append({"pathway_id": pid, "name": name})

    _cache.gene_to_pathways[symbol_upper] = pathways
    return pathways


def get_pathway_image(pathway_id: str) -> bytes | None:
    """Fetch the PNG image bytes for a KEGG pathway map.

    Returns raw PNG bytes or None on failure.
    """
    pid = pathway_id.strip()
    if pid in _cache.pathway_images:
        return _cache.pathway_images[pid]

    _throttle.wait()
    try:
        resp = requests.get(
            f"{KEGG_BASE}/get/{pid}/image",
            timeout=_REQUEST_TIMEOUT + 10,  # images can be large
            stream=True,
        )
        if resp.status_code == 200:
            img_bytes = resp.content
            _cache.pathway_images[pid] = img_bytes
            return img_bytes
    except requests.RequestException:
        pass

    _cache.pathway_images[pid] = None
    return None

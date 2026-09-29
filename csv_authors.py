#!/usr/bin/env python3
"""
Phase B author data: read pre-computed author_affiliations from the Cached_Data
CSV (no PDF fetch / LLM needed) and expose a clean per-paper lookup for the
bulk payload builder.

The CSV's per-author institution strings vary in quality — some are clean
("Stanford University"), many are address blobs ("Department of Chemistry,
University of Southern California, Los Angeles, CA, USA"). To avoid polluting the
graph with thousands of one-off Institution nodes, `affiliated_with` edges are
restricted to institutions that appear in the paper's own clean
`affiliations_json`. Author names (authored_by) are kept regardless — they are
clean and are the primary value of Phase B.
"""

from __future__ import annotations

import csv
import io
import json
import lzma
from pathlib import Path

ROOT = Path(__file__).parent
CSV_DIR = ROOT / "Cached_Data"
PROFILE_ID = "memory-innovation"


def _base_id(arxiv_key: str) -> str:
    return arxiv_key.split("v")[0] if arxiv_key else ""


def filter_author_affiliations(raw_aa: dict, clean_affils: set[str]) -> dict[str, list[str]]:
    """Keep every (clean-named) author; restrict each author's institutions to
    those present in the paper's clean affiliation list.

    Authors with no clean-matching institution are kept with an empty list, so
    they still produce an Author node + authored_by edge, just no affiliated_with.
    """
    out: dict[str, list[str]] = {}
    for author, insts in (raw_aa or {}).items():
        name = (author or "").strip()
        if not name:
            continue
        kept = [i.strip() for i in (insts or []) if i.strip() in clean_affils]
        out[name] = kept
    return out


def _parse_json(val: str, default):
    try:
        return json.loads(val) if val else default
    except (json.JSONDecodeError, TypeError):
        return default


def load_author_lookup(csv_dir: Path = CSV_DIR, profile_id: str = PROFILE_ID) -> dict[str, dict]:
    """Return {base_arxiv_id: filtered_author_affiliations} for the profile.

    Reassembles the split XZ CSV, and for each row builds the cleaned per-paper
    author->institution map.
    """
    parts = sorted(csv_dir.glob("*.xz.part-*"))
    if not parts:
        raise FileNotFoundError(f"No .xz.part-* files in {csv_dir}")
    data = lzma.decompress(b"".join(p.read_bytes() for p in parts)).decode("utf-8")

    lookup: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(data)):
        if row.get("profile") != profile_id:
            continue
        base = _base_id(row.get("arxiv_key", ""))
        if not base:
            continue
        raw_aa = _parse_json(row.get("author_affiliations_json"), {})
        if not raw_aa:
            continue
        clean = set(_parse_json(row.get("affiliations_json"), [])) | set(
            _parse_json(row.get("normalized_affiliations_json"), [])
        )
        clean = {c.strip() for c in clean if c and c.strip()}
        lookup[base] = filter_author_affiliations(raw_aa, clean)
    return lookup

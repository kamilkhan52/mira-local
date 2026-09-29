#!/usr/bin/env python3
"""
Phase A backfill: run classification over the pre-2604 "gap" papers that the
weekly reports selected but were never classified (their topics are missing,
which is why Jan-Mar 2026 has zero Topic / researches edges in the graph).

It reuses the live pipeline's `classify_papers` (same prompt, model, cache
format) so the results are identical to what a normal run would have produced.
Classification only needs the title + arXiv abstract, so no PDF fetch is
required.

After classifying, it merges the results into the flat `cache/classifications.json`
that `backfill_graph.py` reads, keyed by base arXiv id. Re-run
`backfill_graph.py` afterwards to push the new Topic / researches edges into
LightRAG.

Usage:
    # Validate first: classify a small sample, print topic-consistency report,
    # write nothing.
    python3 backfill_classify.py --sample 10

    # Full run: classify every gap paper and update the flat cache.
    python3 backfill_classify.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

import backfill_graph as bg

ROOT = Path(__file__).parent
FLAT_CACHE = ROOT / "cache" / "classifications.json"
GRAPHML = ROOT / "lightrag" / "working_dir" / "graph_chunk_entity_relation.graphml"

_ARXIV_API = "https://export.arxiv.org/api/query"
_FETCH_BATCH = 80
_FETCH_DELAY = 3.0  # arXiv asks for >=3s between requests
# arXiv asks API clients to identify themselves; anonymous clients get throttled.
_UA = "mira-backfill/1.0 (memory-innovation research agent; mailto:oed8205@gmail.com)"


# ─── Pure helpers (unit-tested) ───────────────────────────────────────────────

def is_gap_id(base_id: str, end_prefix: str) -> bool:
    """True if the paper's arXiv YYMM prefix is strictly before end_prefix.

    e.g. is_gap_id("2603.12345", "2604") -> True   (March, in the gap)
         is_gap_id("2605.00001", "2604") -> False  (May, already covered)
    """
    m = re.match(r"(\d{4})\.\d{4,5}", base_id)
    return bool(m) and m.group(1) < end_prefix


def _norm_topic(s: str) -> str:
    """Normalize a topic string for drift detection (case/space/punct insensitive)."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def compare_topics(produced: set[str], existing: set[str]) -> dict:
    """Classify produced topic strings against the existing graph Topic nodes.

    Returns three buckets:
      - matched:     produced topics that already exist verbatim (will merge cleanly)
      - drift:       produced topics that match an existing one only after
                     normalization — i.e. a near-duplicate that would create a
                     SECOND node ("HBM" vs "hbm", "High-Bandwidth Memory")
      - brand_new:   produced topics with no existing counterpart at all
    """
    norm_existing = {_norm_topic(t): t for t in existing}
    matched, drift, brand_new = [], [], []
    for t in sorted(produced):
        if t in existing:
            matched.append(t)
        elif _norm_topic(t) in norm_existing:
            drift.append((t, norm_existing[_norm_topic(t)]))
        else:
            brand_new.append(t)
    return {"matched": matched, "drift": drift, "brand_new": brand_new}


def merge_flat_cache(existing: dict, new_results: dict) -> dict:
    """Merge new {base_id: result} entries into the flat classifications cache.

    Existing entries are never overwritten (idempotent re-runs); only missing
    keys are added.
    """
    merged = dict(existing)
    added = 0
    for base_id, result in new_results.items():
        if base_id not in merged:
            merged[base_id] = result
            added += 1
    merged["__added__"] = added  # caller pops this; avoids a second return value
    return merged


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_existing_topics(graphml: Path) -> set[str]:
    """Read existing Topic node names straight from the GraphML."""
    import xml.etree.ElementTree as ET

    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ET.parse(graphml).getroot()
    topics: set[str] = set()
    for n in root.findall(".//g:node", ns):
        etype = ""
        for d in n.findall("g:data", ns):
            if d.get("key") == "d1":
                etype = (d.text or "").strip().strip('"')
        if etype == "Topic":
            topics.add(n.get("id"))
    return topics


def collect_gap_ids(reports_dir: Path, end_prefix: str) -> list[str]:
    """Return unique base arXiv ids of selected papers in reports, pre-end_prefix."""
    paths = sorted(reports_dir.glob("*.json"))
    selected = bg.pick_best_per_date(paths)
    ids: list[str] = []
    seen: set[str] = set()
    for path in selected:
        try:
            _, body = bg.extract_body(path)
        except Exception:
            continue
        for m in re.finditer(r"arxiv\.org/abs/(\d+\.\d+)", body):
            base = m.group(1).split("v")[0]
            if base in seen or not is_gap_id(base, end_prefix):
                continue
            seen.add(base)
            ids.append(base)
    return ids


def _arxiv_get(chunk: list[str], retries: int = 5) -> requests.Response:
    """GET one id_list batch from arXiv, backing off on 429/5xx."""
    delay = _FETCH_DELAY
    last_err: Exception | None = None
    for attempt in range(retries):
        resp = requests.get(
            _ARXIV_API,
            params={"id_list": ",".join(chunk), "max_results": len(chunk)},
            headers={"User-Agent": _UA},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503):
            last_err = requests.HTTPError(f"{resp.status_code} from arXiv")
            print(f"  arXiv {resp.status_code}, backing off {delay:.0f}s (attempt {attempt + 1}/{retries})")
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
    raise RuntimeError(f"arXiv fetch failed after {retries} attempts: {last_err}")


def fetch_abstracts(base_ids: list[str]) -> dict[str, dict]:
    """Fetch title/abstract/authors for each base id from the arXiv API.

    Returns {base_id: paper_dict} where paper_dict matches mira.fetch's shape
    (id is versioned, plus raw_id/title/summary). Missing ids are simply absent.
    """
    from mira import fetch

    out: dict[str, dict] = {}
    for i in range(0, len(base_ids), _FETCH_BATCH):
        chunk = base_ids[i : i + _FETCH_BATCH]
        resp = _arxiv_get(chunk)
        for paper in fetch._parse_xml(resp.text):
            base = paper["id"].split("v")[0]
            out[base] = paper
        print(f"  fetched abstracts {min(i + _FETCH_BATCH, len(base_ids))}/{len(base_ids)}")
        if i + _FETCH_BATCH < len(base_ids):
            time.sleep(_FETCH_DELAY)
    return out


def _flat_entry(base_id: str, result: dict) -> dict:
    """Shape a classify result like the existing flat-cache entries."""
    entry = {"arxiv_id": base_id}
    entry.update(result)
    return entry


# ─── Orchestration ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A: classify gap papers")
    parser.add_argument("--reports-dir", default=str(bg.REPORTS_DIR))
    parser.add_argument("--end-prefix", default="2604",
                        help="Classify papers with arXiv YYMM strictly before this (default 2604)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Validation mode: classify only N papers, print a topic-consistency report, write nothing")
    args = parser.parse_args()

    # Importing config loads .env (OPENROUTER_API_KEY, RECIPIENT_EMAIL).
    from mira.config import load_config, make_llm_client
    from mira.graph_ingest import normalize_topic
    from mira.pipeline import classify_papers

    reports_dir = Path(args.reports_dir)
    print(f"Collecting gap papers (arXiv < {args.end_prefix}) from {reports_dir.name}...")
    gap_ids = collect_gap_ids(reports_dir, args.end_prefix)
    print(f"  {len(gap_ids)} unique gap papers selected across reports")

    if args.sample:
        gap_ids = gap_ids[: args.sample]
        print(f"  SAMPLE mode: classifying first {len(gap_ids)}")

    print("Fetching abstracts from arXiv...")
    abstracts = fetch_abstracts(gap_ids)
    missing = [i for i in gap_ids if i not in abstracts]
    if missing:
        print(f"  WARNING: {len(missing)} ids returned no arXiv entry (skipped): {missing[:5]}...")

    papers = [abstracts[i] for i in gap_ids if i in abstracts]
    if not papers:
        print("Nothing to classify.")
        return

    config = load_config("memory-innovation", "weekly")
    client = make_llm_client()

    print(f"Classifying {len(papers)} papers with {config['llm_models']['classification']}...")
    classify_papers(papers, config, client)

    # Map results back to base ids
    new_results: dict[str, dict] = {}
    produced_topics: set[str] = set()
    for p in papers:
        base = p["id"].split("v")[0]
        result = {
            "primary_topic": p.get("primary_topic", ""),
            "secondary_topics": p.get("secondary_topics", []),
            "relevance_score": p.get("relevance_score", 0),
            "key_findings": p.get("key_findings", ""),
            "actionable": p.get("actionable", "No"),
        }
        new_results[base] = _flat_entry(base, result)
        # The flat cache keeps the raw classifier output; normalization is applied
        # at ingest time. The report mirrors that so it shows the real merge outcome.
        if result["primary_topic"]:
            produced_topics.add(normalize_topic(result["primary_topic"]))
        produced_topics.update(normalize_topic(t) for t in result["secondary_topics"] if t)

    # Topic-consistency report (always shown), normalized on both sides to match ingest
    existing_topics = {
        normalize_topic(t) for t in (load_existing_topics(GRAPHML) if GRAPHML.exists() else set())
    }
    report = compare_topics(produced_topics, existing_topics)
    print("\n─── Topic consistency vs existing graph (post-normalization) ───")
    print(f"  produced {len(produced_topics)} distinct topics across the sample")
    print(f"  ✅ already in graph (clean merge): {len(report['matched'])}")
    for t in report["matched"]:
        print(f"       {t}")
    print(f"  ⚠️  DRIFT — near-duplicate of an existing node (would create a 2nd node): {len(report['drift'])}")
    for produced, existing in report["drift"]:
        print(f"       {produced!r}  ~  existing {existing!r}")
    print(f"  🆕 brand-new topics (no existing counterpart): {len(report['brand_new'])}")
    for t in report["brand_new"]:
        print(f"       {t}")

    if args.sample:
        print("\nSAMPLE mode — flat cache NOT written. Review the drift list above.")
        print("If drift is acceptable, re-run without --sample to classify all gap papers.")
        return

    # Merge into flat cache
    existing = json.loads(FLAT_CACHE.read_text()) if FLAT_CACHE.exists() else {}
    merged = merge_flat_cache(existing, new_results)
    added = merged.pop("__added__")
    FLAT_CACHE.write_text(json.dumps(merged, indent=2))
    print(f"\nWrote {FLAT_CACHE.relative_to(ROOT)}: +{added} new entries ({len(merged)} total)")
    print("Next: re-run `python3 backfill_graph.py` to push Topic / researches edges into LightRAG.")


if __name__ == "__main__":
    main()

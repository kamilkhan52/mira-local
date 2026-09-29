#!/usr/bin/env python3
"""
Build a single merged custom_kg payload for the gap weeks (pre-end-prefix),
to be bulk-loaded into LightRAG via the library (one batched embed + one save)
instead of thousands of slow per-entity HTTP creates.

Reuses the same parsing (backfill_graph) and the same entity/relationship
mapping (graph_ingest._build_payload, which now normalizes topics), so the
result is identical to what the HTTP path would have produced — just applied in
one shot.

Output: lightrag/working_dir/_bulk_payload.json  (inside the mounted volume so
the in-container bulk_load.py can read it).

Usage:
    python3 build_bulk_payload.py                 # all gap weeks (arXiv < 2604)
    python3 build_bulk_payload.py --limit-weeks 1 # just the first gap week (for a test run)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import backfill_graph as bg
import csv_authors
from mira.graph_ingest import _build_payload

ROOT = Path(__file__).parent
OUT = ROOT / "lightrag" / "working_dir" / "_bulk_payload.json"


def _is_gap_week(body: str, end_prefix: str) -> bool:
    """True if the report contains at least one paper from before end_prefix."""
    for m in re.finditer(r"arxiv\.org/abs/(\d{4})\.\d{4,5}", body):
        if m.group(1) < end_prefix:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged custom_kg for gap weeks")
    parser.add_argument("--reports-dir", default=str(bg.REPORTS_DIR))
    parser.add_argument("--end-prefix", default="2604")
    parser.add_argument("--limit-weeks", type=int, default=0,
                        help="Only process the first N gap weeks (for a small test run)")
    parser.add_argument("--with-authors", action="store_true",
                        help="Phase B: attach author_affiliations from the Cached_Data CSV "
                             "(adds Author nodes + authored_by/affiliated_with edges)")
    args = parser.parse_args()

    author_lookup = {}
    if args.with_authors:
        print("Loading author data from Cached_Data CSV...")
        author_lookup = csv_authors.load_author_lookup()
        print(f"  {len(author_lookup)} papers with author affiliations")

    classifications, affiliations = bg.load_caches()
    paths = sorted(Path(args.reports_dir).glob("*.json"))
    weeks = bg.pick_best_per_date(paths)

    entities: dict[str, dict] = {}
    seen_edges: set[tuple] = set()
    relationships: list[dict] = []
    chunks: list[dict] = []

    processed = 0
    for path in weeks:
        try:
            meta, body = bg.extract_body(path)
        except Exception:
            continue
        if isinstance(meta, list):
            meta = meta[0]
        if not _is_gap_week(body, args.end_prefix):
            continue

        papers = bg.parse_papers(body, classifications, affiliations)
        media = bg.parse_media(body)
        if not papers and not media:
            continue

        if author_lookup:
            for p in papers:
                aa = author_lookup.get(p.get("id", ""))
                if aa:
                    p["author_affiliations"] = aa

        run_date = meta.get("run_date", "unknown")
        config = {
            "profile_id": meta.get("profile_id", "memory-innovation"),
            "current_date": run_date,
            "topic": {"focus": meta.get("topic_focus", "memory technology")},
        }
        source_id = f"{config['profile_id']}-{run_date}-bulk"
        payload = _build_payload(papers, media, config, source_id)

        for e in payload["entities"]:
            entities.setdefault(e["entity_name"], e)  # first occurrence wins
        for r in payload["relationships"]:
            key = (r["src_id"], r["tgt_id"], r["keywords"])
            if key not in seen_edges:
                seen_edges.add(key)
                relationships.append(r)
        chunks.extend(payload["chunks"])

        processed += 1
        print(f"  {run_date}: papers={len(papers)} media={len(media)} "
              f"(cumulative {len(entities)} entities, {len(relationships)} relations)")
        if args.limit_weeks and processed >= args.limit_weeks:
            break

    custom_kg = {
        "entities": list(entities.values()),
        "relationships": relationships,
        "chunks": chunks,
    }
    OUT.write_text(json.dumps(custom_kg))
    print(f"\nWrote {OUT.relative_to(ROOT)}")
    print(f"  {len(custom_kg['entities'])} entities, "
          f"{len(custom_kg['relationships'])} relationships, "
          f"{len(custom_kg['chunks'])} chunks across {processed} week(s)")


if __name__ == "__main__":
    main()

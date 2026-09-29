#!/usr/bin/env python3
"""
Backfill the LightRAG knowledge graph from historical report files.

Usage:
    python3 scripts/backfill_graph.py [--dry-run] [--profile memory-innovation]

For each report in report-files/prod/<profile>/, extracts selected paper ArXiv IDs
from body_markdown, reconstructs paper metadata from per-paper pipeline caches,
and ingests each report into the running LightRAG server.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ── Cache index ──────────────────────────────────────────────────────────────

def _build_cache_index(cache_stage_dir: Path) -> dict[str, dict]:
    """Build arxiv_base_id → most recent result from per-paper cache files."""
    index: dict[str, dict] = {}
    if not cache_stage_dir.exists():
        return index
    for f in cache_stage_dir.iterdir():
        if not f.suffix == ".json":
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        result = data.get("result")
        if not result:
            continue
        # Filename: {versioned_id}__{fingerprint}.json  e.g. 2604.09041v1__a1d42127.json
        versioned_id = f.stem.split("__")[0]
        base_id = re.sub(r"v\d+$", "", versioned_id)
        created_at = data.get("created_at", "")
        # Keep most recent entry per base_id
        if base_id not in index or created_at > index[base_id].get("_created_at", ""):
            index[base_id] = {**result, "_created_at": created_at}
    return index


def build_indices(profile_id: str) -> tuple[dict, dict]:
    cache_root = ROOT / "report-files" / "cache" / profile_id
    cls_index = _build_cache_index(cache_root / "classification")
    aff_index = _build_cache_index(cache_root / "affiliation")
    return cls_index, aff_index


# ── Report parsing ────────────────────────────────────────────────────────────

def extract_arxiv_ids(body_markdown: str) -> list[str]:
    """Return deduplicated list of versioned arxiv IDs found in body_markdown."""
    ids = re.findall(r"arxiv\.org/abs/([\d.]+v?\d*)", body_markdown)
    return list(dict.fromkeys(ids))


def reconstruct_papers(
    arxiv_ids: list[str],
    cls_index: dict,
    aff_index: dict,
    default_topic: str = "Memory Technology",
    arxiv_meta: dict | None = None,
) -> list[dict]:
    """Rebuild paper dicts for ingestion from the per-paper caches.

    `arxiv_meta` (base id -> {title, authors, ...}, from
    scripts/backfill_arxiv_metadata.py) supplies what the caches cannot: neither
    cache stage stores a title, so without it every paper falls back to its bare
    versioned ID and the graph's Paper nodes end up named "2511.03432v1". It is
    also the only author source for graphs the Cached_Data CSV does not cover.
    """
    arxiv_meta = arxiv_meta or {}
    papers = []
    for versioned_id in arxiv_ids:
        base_id = re.sub(r"v\d+$", "", versioned_id)
        cls = cls_index.get(base_id, {})
        aff = aff_index.get(base_id, {})
        meta = arxiv_meta.get(base_id, {})

        paper = {
            "id": base_id,
            "raw_id": f"https://arxiv.org/abs/{versioned_id}",
            "title": meta.get("title") or cls.get("title") or aff.get("title") or versioned_id,
            "affiliations": aff.get("affiliations", []),
            "primary_topic": cls.get("primary_topic") or default_topic,
            "secondary_topics": cls.get("secondary_topics", []),
            "relevance_score": int(cls.get("relevance_score") or 0),
            "credibility_tier": int(aff.get("credibility_tier") or 0),
            "key_findings": cls.get("key_findings", ""),
            # short_summary not cached separately — use key_findings as fallback
            "short_summary": cls.get("key_findings", ""),
        }
        # arXiv gives author names but not per-author affiliations, so each maps
        # to an empty institution list: Author nodes and authored_by edges (what
        # the hypothesis stack reads) are created, affiliated_with edges are not.
        # Paper→Institution edges still come from the affiliation cache above.
        if meta.get("authors"):
            paper["author_affiliations"] = {a: [] for a in meta["authors"]}
        papers.append(paper)
    return papers


def load_report(report_file: Path) -> dict | None:
    """Load a report file (handles both dict and list-wrapped formats)."""
    try:
        raw = json.loads(report_file.read_text())
    except Exception as exc:
        print(f"  WARNING: could not parse {report_file.name}: {exc}")
        return None
    d = raw[0] if isinstance(raw, list) else raw
    if not isinstance(d, dict):
        return None
    return d


# ── Bulk payload accumulation ─────────────────────────────────────────────────
# The HTTP path posts entities/relations one-by-one (slow, and omits chunks).
# --bulk-out instead merges every report into one custom_kg payload for the
# in-container loader, which embeds in a single batched pass AND inserts chunks.

def _new_accumulator() -> dict:
    return {"entities": {}, "seen_edges": set(), "relationships": [], "chunks": []}


def _merge_payload(acc: dict, payload: dict) -> None:
    for e in payload["entities"]:
        acc["entities"].setdefault(e["entity_name"], e)  # first occurrence wins
    for r in payload["relationships"]:
        key = (r["src_id"], r["tgt_id"], r["keywords"])
        if key not in acc["seen_edges"]:
            acc["seen_edges"].add(key)
            acc["relationships"].append(r)
    acc["chunks"].extend(payload["chunks"])


def _finalize_accumulator(acc: dict) -> dict:
    return {
        "entities": list(acc["entities"].values()),
        "relationships": acc["relationships"],
        "chunks": acc["chunks"],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill LightRAG graph from historical reports")
    parser.add_argument("--profile", default="memory-innovation", help="Profile ID")
    parser.add_argument("--dry-run", action="store_true", help="Parse and show stats without ingesting")
    parser.add_argument("--from-date", default="", help="Only ingest reports on or after this date (YYYY-MM-DD)")
    parser.add_argument(
        "--base-url",
        default=None,
        help="LightRAG server base URL (default: $LIGHTRAG_BASE_URL or http://localhost:9621). "
        "Point at http://localhost:9622 to backfill the optical instance.",
    )
    parser.add_argument(
        "--arxiv-meta",
        default=None,
        metavar="PATH",
        help="arXiv metadata JSON from scripts/backfill_arxiv_metadata.py, supplying paper "
        "titles and authors the per-paper caches do not hold. Without it, Paper nodes are "
        "named by bare arXiv ID and no Author nodes are created. "
        "e.g. cache/arxiv_metadata_optical.json",
    )
    parser.add_argument(
        "--bulk-out",
        default=None,
        metavar="PATH",
        help="Instead of HTTP-posting, write one merged custom_kg payload (entities, "
        "relationships AND chunks) to PATH for the in-container bulk loader. Recommended "
        "for a full backfill — one batched embed pass, no per-entity timeouts. "
        "e.g. lightrag/working_dir_optical/_bulk_payload.json",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from mira.graph_ingest import ingest_report, _resolve_base_url, _build_payload

    base_url = _resolve_base_url(args.base_url)
    if args.bulk_out:
        print(f"Mode: bulk payload → {args.bulk_out}")
    else:
        print(f"Target LightRAG instance: {base_url}")

    report_dir = ROOT / "report-files" / "prod" / args.profile
    if not report_dir.exists():
        sys.exit(f"Report directory not found: {report_dir}")

    print(f"Building cache index for profile '{args.profile}'...")
    cls_index, aff_index = build_indices(args.profile)
    print(f"  Classification cache: {len(cls_index)} entries")
    print(f"  Affiliation cache:    {len(aff_index)} entries")

    arxiv_meta = {}
    if args.arxiv_meta:
        meta_path = Path(args.arxiv_meta)
        if not meta_path.exists():
            sys.exit(f"arXiv metadata not found: {meta_path}")
        arxiv_meta = json.loads(meta_path.read_text())
        n_authors = sum(len(m.get("authors") or []) for m in arxiv_meta.values())
        print(f"  arXiv metadata:       {len(arxiv_meta)} papers, {n_authors} author mentions")

    report_files = sorted(report_dir.glob("*.json"))
    print(f"\nFound {len(report_files)} report files in {report_dir}")

    processed = 0
    skipped = 0
    acc = _new_accumulator() if args.bulk_out else None

    for report_file in report_files:
        d = load_report(report_file)
        if d is None:
            skipped += 1
            continue

        run_date = d.get("run_date", "unknown")

        if args.from_date and run_date < args.from_date:
            skipped += 1
            continue

        body = d.get("body_markdown") or d.get("body") or ""
        arxiv_ids = extract_arxiv_ids(body)
        if not arxiv_ids:
            skipped += 1
            continue
        profile_id = d.get("profile_id", args.profile)
        # Cache-miss papers fall back to this report's own topic, never the
        # hardcoded memory default, so a non-memory graph stays domain-clean.
        default_topic = d.get("topic_name") or args.profile
        papers = reconstruct_papers(arxiv_ids, cls_index, aff_index,
                                    default_topic=default_topic, arxiv_meta=arxiv_meta)

        # Count how many have real metadata vs just the arxiv ID fallback
        with_topics = sum(1 for p in papers if p["primary_topic"] != default_topic)
        with_affiliations = sum(1 for p in papers if p["affiliations"])
        with_titles = sum(1 for p in papers if p["title"] != p["raw_id"].rsplit("/", 1)[-1])
        with_authors = sum(1 for p in papers if p.get("author_affiliations"))

        print(f"\n[{run_date}] {report_file.name}")
        print(f"  {len(papers)} papers | {with_topics} with topics | {with_affiliations} with affiliations "
              f"| {with_titles} with titles | {with_authors} with authors")

        if args.dry_run:
            print("  (dry-run — skipping ingest)")
            processed += 1
            continue

        # Build a minimal config dict for graph_ingest
        config = {
            "profile_id": profile_id,
            "current_date": run_date,
            "mode": d.get("period_label", "weekly"),
            "topic": {"focus": d.get("topic_focus", "memory technology")},
        }

        if acc is not None:
            source_id = f"{profile_id}-{run_date}-bulk"
            _merge_payload(acc, _build_payload(papers, [], config, source_id))
        else:
            ingest_report(papers, [], config, base_url=base_url)
        processed += 1

    if acc is not None:
        custom_kg = _finalize_accumulator(acc)
        out_path = Path(args.bulk_out)
        out_path.write_text(json.dumps(custom_kg))
        print(
            f"\nWrote {out_path} — {len(custom_kg['entities'])} entities, "
            f"{len(custom_kg['relationships'])} relationships, {len(custom_kg['chunks'])} chunks "
            f"across {processed} reports ({skipped} skipped)."
        )
        print("Load it with the in-container bulk loader (server stopped):")
        print("  docker compose -f lightrag/docker-compose.lightrag.yml stop lightrag-optical")
        print("  docker compose -f lightrag/docker-compose.lightrag.yml run --rm --no-deps \\")
        print("      --entrypoint python lightrag-optical /app/working_dir/bulk_load.py")
        print("  docker compose -f lightrag/docker-compose.lightrag.yml start lightrag-optical")
        return

    print(f"\nDone. {processed} reports ingested, {skipped} skipped (no arxiv links).")
    if not args.dry_run:
        print(f"Query the graph at {base_url}/webui/")


if __name__ == "__main__":
    main()

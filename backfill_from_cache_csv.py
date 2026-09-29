#!/usr/bin/env python3
"""
Backfill LightRAG from the Cached_Data CSV.

Reads affiliation_relevance_joined_dedup_with_authors_*.csv.xz.part-* files,
groups memory-innovation papers by ISO week, and ingests them into LightRAG.

Usage:
    python3 backfill_from_cache_csv.py [--csv-dir Cached_Data] [--dry-run]
                                       [--batch-size 50] [--relevance-min 5]
                                       [--checkpoint backfill_checkpoint.json]
                                       [--reset-checkpoint]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import lzma
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent

PROFILE_ID = "memory-innovation"
RELEVANCE_MIN = 5
REPORTS_DIR = ROOT / "report-files" / "prod" / "memory-innovation"

_ARXIV_VERSIONED_RE = re.compile(r"arxiv\.org/abs/([\d.]+v\d+)")
_TITLE_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://arxiv\.org/abs/[^\)]+)\)")


def load_checkpoint(path: Path) -> set[str]:
    """Return the set of already-completed week strings from the checkpoint file."""
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValueError):
        return set()


def save_checkpoint(path: Path, completed_weeks: set[str]) -> None:
    """Write the completed weeks set to the checkpoint file as a JSON list."""
    path.write_text(json.dumps(sorted(completed_weeks)))


def iso_week_monday(ts_str: str) -> str:
    """Return YYYY-MM-DD of the Monday of the ISO week for the given ISO timestamp."""
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def build_title_lookup(reports_dir: Path) -> dict[str, str]:
    """Scan report JSON files and return {versioned_arxiv_key: title}."""
    lookup: dict[str, str] = {}
    for path in reports_dir.glob("*.json"):
        try:
            with open(path) as f:
                d = json.load(f)
            if isinstance(d, list):
                d = d[0]
            body = d.get("body_markdown", d.get("body", ""))
            for m in _TITLE_LINK_RE.finditer(body):
                title = m.group(1).strip()
                url = m.group(2).strip()
                km = _ARXIV_VERSIONED_RE.search(url)
                if km and title:
                    lookup[km.group(1)] = title
        except Exception:
            continue
    return lookup


def csv_row_to_paper(row: dict, title_lookup: dict[str, str]) -> dict:
    """Map a CSV row to the paper dict shape expected by graph_ingest.ingest_report."""

    def _parse(val: str, default):
        try:
            return json.loads(val) if val else default
        except (json.JSONDecodeError, TypeError):
            return default

    def _safe_int(val: str, default: int = 0) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    arxiv_key = row["arxiv_key"]
    return {
        "title": title_lookup.get(arxiv_key) or f"arxiv:{arxiv_key}",
        "raw_id": row["arxiv_id"],
        "relevance_score": _safe_int(row.get("relevance_score", "")),
        "credibility_tier": _safe_int(row.get("credibility_tier", "")),
        "primary_topic": (row.get("primary_topic") or "").strip(),
        "secondary_topics": _parse(row.get("secondary_topics_json"), []),
        "affiliations": _parse(row.get("affiliations_json"), []),
        "authors": _parse(row.get("authors_json"), []),
        "author_affiliations": _parse(row.get("author_affiliations_json"), {}),
        "key_findings": row.get("key_findings", ""),
        "short_summary": row.get("potential_impact", ""),
    }


def load_csv(csv_dir: Path) -> list[dict]:
    """Reassemble split XZ parts, decompress, and return all rows."""
    parts = sorted(csv_dir.glob("*.xz.part-*"))
    if not parts:
        raise FileNotFoundError(f"No .xz.part-* files found in {csv_dir}")
    raw = b"".join(p.read_bytes() for p in parts)
    data = lzma.decompress(raw).decode("utf-8")
    return list(csv.DictReader(io.StringIO(data)))


def batch_papers(papers: list[dict], batch_size: int) -> list[list[dict]]:
    """Split papers into batches of at most batch_size."""
    return [papers[i:i + batch_size] for i in range(0, len(papers), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill LightRAG from Cached_Data CSV")
    parser.add_argument("--csv-dir", default="Cached_Data", help="Directory with .xz.part-* files")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without posting to LightRAG")
    parser.add_argument("--checkpoint", default="backfill_checkpoint.json", help="Path to checkpoint file (default: backfill_checkpoint.json)")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete checkpoint file before starting")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Papers per checkpoint batch (default: 20)")
    parser.add_argument("--batch-delay", type=float, default=30.0,
                        help="Seconds to wait between batches (default: 30)")
    parser.add_argument("--relevance-min", type=int, default=RELEVANCE_MIN,
                        help=f"Minimum relevance score (default: {RELEVANCE_MIN})")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if args.reset_checkpoint:
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    csv_dir = Path(args.csv_dir)

    print("Loading CSV (may take a moment)...")
    rows = load_csv(csv_dir)
    print(f"  Loaded {len(rows):,} rows")

    print(f"Building title lookup from {REPORTS_DIR}...")
    title_lookup = build_title_lookup(REPORTS_DIR)
    print(f"  Found {len(title_lookup):,} titles")

    papers_by_week: dict[str, list[dict]] = {}
    skipped = 0
    for row in rows:
        if row.get("profile") != PROFILE_ID:
            continue
        try:
            if int(row.get("relevance_score") or 0) < args.relevance_min:
                skipped += 1
                continue
        except (ValueError, TypeError):
            skipped += 1
            continue
        ts = row.get("classification_created_at", "")
        if not ts:
            skipped += 1
            continue
        try:
            week = iso_week_monday(ts)
        except (ValueError, TypeError):
            skipped += 1
            continue
        papers_by_week.setdefault(week, []).append(csv_row_to_paper(row, title_lookup))

    total = sum(len(v) for v in papers_by_week.values())
    print(f"Filtered to {total:,} papers across {len(papers_by_week)} weeks (skipped {skipped:,})")
    for week in sorted(papers_by_week):
        print(f"  {week}: {len(papers_by_week[week]):,} papers")

    completed = load_checkpoint(checkpoint_path)
    if completed:
        print(f"Resuming — {len(completed)} batches already completed.")

    if args.dry_run:
        print("Dry run — skipping LightRAG ingestion.")
        return

    from mira import graph_ingest

    config_base = {"profile_id": PROFILE_ID, "topic": {"focus": "memory technology"}}
    for week in sorted(papers_by_week):
        week_papers = papers_by_week[week]
        batches = batch_papers(week_papers, args.batch_size)
        n_batches = len(batches)
        for batch_idx, batch in enumerate(batches):
            key = f"{week}:{batch_idx}"
            if key in completed:
                print(f"  [{week} {batch_idx+1}/{n_batches}] already done, skipping")
                continue
            print(f"  [{week} {batch_idx+1}/{n_batches}] {len(batch)} papers...")
            config = {**config_base, "current_date": week}
            if graph_ingest.ingest_report(batch, [], config):
                completed.add(key)
                save_checkpoint(checkpoint_path, completed)
            else:
                print(f"  [{week} {batch_idx+1}/{n_batches}] FAILED — will retry on next run")
            if args.batch_delay > 0:
                time.sleep(args.batch_delay)

    print("Done.")


if __name__ == "__main__":
    main()

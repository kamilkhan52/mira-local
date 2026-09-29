#!/usr/bin/env python3
"""
Backfill LightRAG from existing report files.

Parses paper and media article data directly from saved report JSON files,
enriches paper metadata from the classification/affiliation caches, and
ingests into LightRAG using the custom_kg API.

For each run_date, only the report with the most papers is processed to avoid
double-counting duplicate runs.

Usage:
    python3 backfill_graph.py [--reports-dir report-files/prod/memory-innovation]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "report-files" / "prod" / "memory-innovation"


def load_caches() -> tuple[dict, dict]:
    cls_path = ROOT / "cache" / "classifications.json"
    aff_path = ROOT / "cache" / "affiliations.json"
    with open(cls_path) as f:
        classifications = json.load(f)
    with open(aff_path) as f:
        affiliations = json.load(f)
    return classifications, affiliations


def extract_body(path: Path) -> tuple[dict, str]:
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, list):
        d = d[0]
    raw = d.get("body_markdown", d.get("body", ""))
    m = re.search(r'"body":\s*"(.*)', raw, re.DOTALL)
    if m:
        body = m.group(1).encode().decode("unicode_escape", errors="replace")
    else:
        body = raw
    return d, body


def extract_arxiv_id(url: str) -> str:
    m = re.search(r"arxiv\.org/abs/(\d+\.\d+)", url)
    return m.group(1).split("v")[0] if m else ""


def parse_papers(body: str, classifications: dict, affiliations: dict) -> list[dict]:
    papers = []
    seen_urls: set[str] = set()
    pattern = re.compile(
        r"\[([^\]]+)\]\((https?://arxiv\.org/abs/[^\)]+)\)\s*[—–\-]+\s*([^\n]+)"
    )
    for m in pattern.finditer(body):
        title = m.group(1).strip()
        url = m.group(2).strip()
        if url in seen_urls:
            continue
        seen_urls.add(url)

        institutions_raw = m.group(3).strip()
        # "Also Worth Noting" entries append description after a colon — strip it
        institutions_raw = re.split(r":\s+[A-Z]", institutions_raw)[0]

        arxiv_id = extract_arxiv_id(url)

        # Summary: next non-empty paragraph after the paper line
        rest = body[m.end() : m.end() + 600]
        sm = re.search(r"\n\n([^\[#\n][^\n]{20,})", rest)
        summary = sm.group(1).strip() if sm else ""
        # Don't include "(Note: Full-text analysis unavailable)" as summary
        if summary.startswith("(Note:"):
            summary = ""

        cls = classifications.get(arxiv_id, {})
        aff = affiliations.get(arxiv_id, {})

        # Prefer cache affiliations (more accurate than report body)
        affils = aff.get("affiliations") or [
            a.strip() for a in re.split(r",|;", institutions_raw) if a.strip()
        ]

        papers.append(
            {
                "title": title,
                "raw_id": url,
                "id": arxiv_id,
                "affiliations": affils,
                "primary_topic": cls.get("primary_topic", ""),
                "secondary_topics": cls.get("secondary_topics") or [],
                "relevance_score": cls.get("relevance_score", ""),
                "credibility_tier": aff.get("credibility_tier", ""),
                "key_findings": cls.get("key_findings", ""),
                "short_summary": summary,
            }
        )
    return papers


def parse_media(body: str) -> list[dict]:
    articles = []
    idx = body.find("## Media Intelligence")
    if idx == -1:
        return articles

    section = body[idx:]
    # Trim at next top-level section
    cut = re.search(r"\n## ", section[3:])
    if cut:
        section = section[: cut.start() + 3]

    seen_urls: set[str] = set()

    # Format A: [Title](url) — Source, YYYY-MM-DD. Summary  (same line, dot or colon)
    pattern_a = re.compile(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)\s*[—–\-]+\s*([^,\n*]+),\s*([\d]{4}-[\d]{2}-[\d]{2})[.:]\s*([^\n]+)"
    )
    # Format B: [Title](url) — *Source, YYYY-MM-DD*\nSummary  (italics, summary next line)
    pattern_b = re.compile(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)\s*[—–\-]+\s*\*([^,\n]+),\s*([\d]{4}-[\d]{2}-[\d]{2})\*\s*\n([^\n\[#]{20,})"
    )
    # Format C: [Title](url) — Source, YYYY-MM-DD\nSummary  (no separator, summary next line)
    pattern_c = re.compile(
        r"\[([^\]]+)\]\((https?://[^\)]+)\)\s*[—–\-]+\s*([^,\n*]+),\s*([\d]{4}-[\d]{2}-[\d]{2})\s*\n([^\n\[#*]{20,})"
    )

    for pattern in (pattern_a, pattern_b, pattern_c):
        for m in pattern.finditer(section):
            url = m.group(2).strip()
            if url in seen_urls:
                continue
            seen_urls.add(url)
            articles.append(
                {
                    "title": m.group(1).strip(),
                    "url": url,
                    "source": m.group(3).strip(),
                    "date": m.group(4).strip(),
                    "summary": m.group(5).strip(),
                }
            )
    return articles


def pick_best_per_date(paths: list[Path]) -> list[Path]:
    """For each run_date, keep the file with the most arxiv paper links."""
    best: dict[str, tuple[int, Path]] = {}
    for path in paths:
        try:
            _, body = extract_body(path)
        except Exception:
            continue
        count = len(re.findall(r"arxiv\.org/abs/", body))
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, list):
            d = d[0]
        run_date = d.get("run_date") or path.stem
        if run_date not in best or count > best[run_date][0]:
            best[run_date] = (count, path)
    return sorted([v for _, v in best.values()], key=lambda p: p.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill LightRAG from report files")
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't ingest")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    if not reports_dir.exists():
        print(f"ERROR: {reports_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print("Loading caches...")
    try:
        classifications, affiliations = load_caches()
        print(f"  {len(classifications)} classified papers, {len(affiliations)} affiliation records")
    except FileNotFoundError as e:
        print(f"ERROR: cache file missing: {e}", file=sys.stderr)
        sys.exit(1)

    all_paths = sorted(reports_dir.glob("*.json"))
    selected = pick_best_per_date(all_paths)
    print(f"\nFound {len(all_paths)} report files → {len(selected)} unique weeks after dedup\n")

    if not args.dry_run:
        from mira.graph_ingest import ingest_report

    total_papers = total_articles = 0

    for i, path in enumerate(selected, 1):
        try:
            meta, body = extract_body(path)
        except Exception as e:
            print(f"[{i}/{len(selected)}] SKIP {path.name}: {e}")
            continue

        if isinstance(meta, list):
            meta = meta[0]

        run_date = meta.get("run_date", "unknown")
        papers = parse_papers(body, classifications, affiliations)
        articles = parse_media(body)

        print(f"[{i}/{len(selected)}] {run_date}  papers={len(papers)}  articles={len(articles)}  ({path.name[:50]})")

        if not papers and not articles:
            print("  (nothing to ingest, skipping)")
            continue

        total_papers += len(papers)
        total_articles += len(articles)

        if args.dry_run:
            continue

        config = {
            "profile_id": meta.get("profile_id", "memory-innovation"),
            "current_date": run_date,
            "topic": {
                "focus": meta.get("topic_focus", "memory technology"),
                "name": meta.get("topic_name", "Memory Technology Research"),
            },
        }

        ingest_report(papers, articles, config)

    print(f"\nDone. Processed {total_papers} papers and {total_articles} articles across {len(selected)} weeks.")
    if not args.dry_run:
        print("Open http://localhost:9621/webui/ to explore the graph.")


if __name__ == "__main__":
    main()

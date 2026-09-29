#!/usr/bin/env python3
"""Fetch titles and authors from the arXiv API for a graph's Paper nodes.

    python3 scripts/backfill_arxiv_metadata.py --graph optical

Why this exists: the optical graph's Paper nodes are keyed by arXiv ID rather
than title, and 421 of 591 carry no classification metadata. Neither cache stage
under report-files/cache/ stores a title at all, so titles cannot come from
there — `reconstruct_papers` falls back to the bare versioned ID for every paper.
arXiv has titles and authors for the same IDs, needs no key, and costs no LLM
calls.

Writes a metadata JSON keyed by base arXiv ID, consumed when rebuilding the
bulk payload. It does not touch the graph itself.

Why not the Cached_Data CSV (how the memory graph got its authors): measured
against the optical graph it covers 248 of 591 papers — it is a 2026-05-08
snapshot and most optical papers postdate it. It is also the source of the
corrupted names is_valid_author exists to filter.

Spec: docs/superpowers/specs/2026-07-16-optical-graphrag-parity-design.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mira.fetch import _HEADERS, _parse_xml  # noqa: E402
from mira.graph_target import DEFAULT_GRAPH, GRAPH_NAMES, resolve_target  # noqa: E402

# arXiv's API allows large id_list queries, but a modest batch keeps any single
# failure cheap to retry and stays well inside their rate guidance.
_BATCH = 50
_DELAY_SECONDS = 3.0
# Below this, something is wrong with the ID set rather than with a few papers;
# the caller should look before building a graph on it (spec §2).
COVERAGE_FLOOR = 0.95
_AUTHOR_PLACEHOLDERS = {
    "anonymous",
    "corresponding author",
    "et al",
    "unknown",
}


def is_valid_author(name: str) -> bool:
    """Reject obvious metadata artifacts while preserving plausible names.

    This is deliberately syntax-only. ``mira.author_validate.is_real_author``
    compares a graph node against a separate ground-truth author list; this
    backfill is creating that list from arXiv and therefore has no independent
    comparison set.
    """
    normalized = " ".join((name or "").split())
    return (
        bool(normalized)
        and normalized.casefold() not in _AUTHOR_PLACEHOLDERS
        and not any(character.isdigit() for character in normalized)
        and any(character.isalpha() for character in normalized)
    )


def base_id(arxiv_id: str) -> str:
    """Strip the version suffix: '2511.03432v1' -> '2511.03432'.

    Requests and responses are matched on this. arXiv resolves an unversioned
    ID to the *latest* version, so asking for '2604.18496' can come back as
    '2604.18496v2' — exact-string matching would score that a miss.
    """
    return re.sub(r"v\d+$", "", arxiv_id)


def paper_ids(graphml: Path) -> list[str]:
    """Paper node IDs from a target's graphml, in stable order."""
    import networkx as nx

    g = nx.read_graphml(graphml)
    return sorted(
        n for n, d in g.nodes(data=True) if d.get("entity_type") == "Paper"
    )


def report_paper_ids(profile: str) -> list[str]:
    """arXiv IDs from a profile's report files — every paper a rebuild will ingest.

    The graphml only knows papers already ingested, so it misses any report that
    landed after the last build. Reading the reports (the same source
    scripts/backfill_graph.py ingests from) covers those too, which matters when
    the metadata feeds a rebuild rather than a patch of the current graph.
    """
    from scripts.backfill_graph import extract_arxiv_ids, load_report

    report_dir = ROOT / "report-files" / "prod" / profile
    ids: dict[str, None] = {}
    for f in sorted(report_dir.glob("*.json")):
        d = load_report(f)
        if not d:
            continue
        body = d.get("body_markdown") or d.get("body") or ""
        for i in extract_arxiv_ids(body):
            ids.setdefault(i, None)
    return list(ids)


def _fetch_batch(ids: list[str]) -> list[dict]:
    resp = requests.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": ",".join(ids), "max_results": len(ids)},
        headers=_HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    return _parse_xml(resp.text)


def _batches(items: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def fetch_metadata(
    ids: list[str],
    fetch: Callable[[list[str]], list[dict]] = _fetch_batch,
    delay: float = _DELAY_SECONDS,
    log: Callable[[str], None] = print,
) -> dict[str, dict]:
    """Map base arXiv ID -> {title, authors} for every ID that resolves.

    `fetch` is injectable so tests never touch the network (the seam mirrors
    mira/hypothesis/novelty.py's). A batch that fails is reported and skipped
    rather than aborting the run — its IDs then surface as unresolved, which is
    the same signal as a withdrawn paper and is handled the same way.
    """
    out: dict[str, dict] = {}
    batches = list(_batches(ids, _BATCH))
    for i, batch in enumerate(batches, 1):
        try:
            papers = fetch(batch)
        except requests.RequestException as exc:
            log(f"  WARNING: batch {i}/{len(batches)} failed ({exc}) — IDs reported unresolved")
            continue
        for p in papers:
            authors = [a.strip() for a in p.get("authors") or [] if a and a.strip()]
            out[base_id(p["id"])] = {
                "arxiv_id": p["id"],
                # arXiv's own entry id ("http://arxiv.org/abs/2511.03432v1").
                # Keep it verbatim: the classification cache fingerprint hashes
                # this exact string, so a reconstructed URL (https://, or a
                # different version) silently misses the cache and re-bills the
                # LLM for a paper that was already classified.
                "raw_id": p.get("raw_id") or "",
                "title": (p.get("title") or "").strip(),
                # The abstract: the classification stage prompts on it
                # (mira/pipeline.py:_render_cls_prompts reads paper["summary"]).
                "summary": (p.get("summary") or "").strip(),
                "authors": [a for a in authors if is_valid_author(a)],
                "rejected_authors": [a for a in authors if not is_valid_author(a)],
            }
        log(f"  batch {i}/{len(batches)}: {len(papers)} entries ({len(out)}/{len(ids)} resolved)")
        if delay and i < len(batches):
            time.sleep(delay)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch arXiv titles + authors for a graph's papers.")
    ap.add_argument("--graph", choices=GRAPH_NAMES, default=DEFAULT_GRAPH)
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="Output JSON (default cache/arxiv_metadata_<graph>.json)")
    ap.add_argument("--from-reports", default=None, metavar="PROFILE",
                    help="Take paper IDs from this profile's report files instead of the "
                         "graphml, covering reports not yet ingested. Use when the metadata "
                         "feeds a rebuild. e.g. optical-interconnects")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only fetch the first N papers (smoke-testing)")
    args = ap.parse_args()

    target = resolve_target(args.graph)
    if args.from_reports:
        ids = report_paper_ids(args.from_reports)
        print(f"{len(ids)} papers across {args.from_reports} report files")
    else:
        if not target.graphml.exists():
            print(f"ERROR: no graphml at {target.graphml}", file=sys.stderr)
            return 1
        ids = paper_ids(target.graphml)
        print(f"{len(ids)} Paper nodes in the {args.graph} graph")
    if args.limit:
        ids = ids[: args.limit]

    meta = fetch_metadata(ids)

    requested = {base_id(i) for i in ids}
    unresolved = sorted(requested - set(meta))
    coverage = len(meta) / len(requested) if requested else 0.0
    titled = sum(1 for m in meta.values() if m["title"])
    with_authors = sum(1 for m in meta.values() if m["authors"])
    rejected = sum(len(m["rejected_authors"]) for m in meta.values())

    print(f"\nResolved {len(meta)}/{len(requested)} ({coverage:.1%})")
    print(f"  with title:   {titled}")
    print(f"  with authors: {with_authors}")
    print(f"  author names rejected by is_valid_author: {rejected}")
    if unresolved:
        print(f"\n{len(unresolved)} unresolved (withdrawn, or not arXiv items):")
        for u in unresolved:
            print(f"  {u}")

    out_path = Path(args.out) if args.out else ROOT / "cache" / f"arxiv_metadata_{args.graph}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"\nWrote {out_path}")

    # Report, don't swallow: a few unresolved IDs are expected and fine, but a
    # broad shortfall means the ID set is wrong and the graph should not be
    # rebuilt on top of it.
    if coverage < COVERAGE_FLOOR:
        print(f"\nERROR: coverage {coverage:.1%} is below the {COVERAGE_FLOOR:.0%} floor — "
              "stop and reassess rather than rebuilding on this.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

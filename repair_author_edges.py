#!/usr/bin/env python3
"""
Repair missing authored_by and affiliated_with edges in the LightRAG graph.

Finds Paper entities without authored_by edges, looks up author data from
the Cached_Data CSV and local affiliation cache, then posts missing Author
entities and relationships.

Usage:
    python3 repair_author_edges.py [--dry-run]
                                   [--checkpoint repair_checkpoint.json]
                                   [--reset-checkpoint]
                                   [--batch-delay 5]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import lzma
import re
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).parent

GRAPHML_PATH = ROOT / "lightrag" / "working_dir" / "graph_chunk_entity_relation.graphml"
REPORTS_DIR = ROOT / "report-files" / "prod" / "memory-innovation"
CACHE_PATH = ROOT / "cache" / "affiliations.json"
CSV_DIR = ROOT / "Cached_Data"

ENTITY_URL = "http://localhost:9621/graph/entity/create"
RELATION_URL = "http://localhost:9621/graph/relation/create"

_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
_TITLE_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://arxiv\.org/abs/[^\)]+)\)")
_ARXIV_VERSIONED_RE = re.compile(r"arxiv\.org/abs/([\d.]+v\d+)")


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def papers_missing_authored_by(graphml_path: Path) -> set[str]:
    """Return entity names of Paper nodes with no outgoing authored_by edge."""
    tree = ET.parse(graphml_path)
    root = tree.getroot()
    ns = _GRAPHML_NS

    # Build key-id → attr.name mapping
    key_map: dict[str, str] = {}
    for key_el in root.iter(f"{{{ns}}}key"):
        kid = key_el.attrib.get("id")
        attr_name = key_el.attrib.get("attr.name")
        if kid and attr_name:
            key_map[kid] = attr_name

    # Collect all Paper node ids
    paper_nodes: set[str] = set()
    for node in root.iter(f"{{{ns}}}node"):
        node_id = node.attrib.get("id", "")
        for data in node.iter(f"{{{ns}}}data"):
            if key_map.get(data.attrib.get("key", "")) == "entity_type":
                if (data.text or "").strip() == "Paper":
                    paper_nodes.add(node_id)

    # Collect sources of authored_by edges
    papers_with_edge: set[str] = set()
    for edge in root.iter(f"{{{ns}}}edge"):
        source = edge.attrib.get("source", "")
        for data in edge.iter(f"{{{ns}}}data"):
            if key_map.get(data.attrib.get("key", "")) == "keywords":
                if "authored_by" in (data.text or ""):
                    papers_with_edge.add(source)

    return paper_nodes - papers_with_edge


def title_to_arxiv_key(reports_dir: Path) -> dict[str, str]:
    """Scan *.json report files and return {title: versioned_arxiv_key}."""
    result: dict[str, str] = {}
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
                    result[title] = km.group(1)
        except Exception:
            continue
    return result


def csv_author_lookup(csv_dir: Path) -> dict[str, dict]:
    """Load split XZ CSV parts, filter to memory-innovation, return {arxiv_key: {author: [insts]}}."""
    parts = sorted(csv_dir.glob("*.xz.part-*"))
    if not parts:
        raise FileNotFoundError(f"No .xz.part-* files found in {csv_dir}")
    raw = b"".join(p.read_bytes() for p in parts)
    data = lzma.decompress(raw).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(data)))

    result: dict[str, dict] = {}
    for row in rows:
        if row.get("profile") != "memory-innovation":
            continue
        arxiv_key = row.get("arxiv_key", "").strip()
        if not arxiv_key:
            continue
        raw_val = row.get("author_affiliations_json", "").strip()
        if not raw_val:
            continue
        try:
            affiliations = json.loads(raw_val)
        except (json.JSONDecodeError, ValueError):
            continue
        if affiliations:
            result[arxiv_key] = affiliations
    return result


def cache_author_lookup(cache_path: Path) -> dict[str, dict]:
    """Load affiliations.json cache and return {base_arxiv_id: {author: [insts]}}."""
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}
    result: dict[str, dict] = {}
    for base_id, entry in raw.items():
        affiliations = entry.get("author_affiliations")
        if affiliations:
            result[base_id] = affiliations
    return result


def resolve_authors(
    entity_name: str,
    title_map: dict,
    csv_map: dict,
    cache_map: dict,
) -> dict:
    """Return {author: [institutions]} for a paper entity name, or {} if not found."""
    if entity_name.startswith("arxiv:"):
        key: str | None = entity_name[6:]
    else:
        key = title_map.get(entity_name)

    if key is None:
        return {}

    # Try CSV first with versioned key
    if key in csv_map:
        return csv_map[key]

    # Fallback: strip version suffix, try cache
    base_key = re.sub(r"v\d+$", "", key)
    if base_key in cache_map:
        return cache_map[base_key]

    return {}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> set[str]:
    """Return set of completed paper entity names from checkpoint file."""
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValueError):
        return set()


def save_checkpoint(path: Path, completed: set[str]) -> None:
    """Write completed set to checkpoint file as JSON list."""
    path.write_text(json.dumps(sorted(completed)))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_entity(session: requests.Session, entity_type: str, name: str) -> bool:
    """POST an entity; treat 400 as success (already exists)."""
    payload = {
        "entity_name": name,
        "entity_type": entity_type,
        "description": name,
        "source_id": "repair-author-edges",
    }
    resp = session.post(ENTITY_URL, json=payload, timeout=120)
    return resp.status_code in (200, 201, 400)


def _post_relation(
    session: requests.Session,
    source: str,
    target: str,
    keywords: str,
    description: str,
) -> bool:
    """POST a relation; treat 400 as success (already exists)."""
    payload = {
        "src_id": source,
        "tgt_id": target,
        "keywords": keywords,
        "description": description,
        "source_id": "repair-author-edges",
    }
    resp = session.post(RELATION_URL, json=payload, timeout=120)
    return resp.status_code in (200, 201, 400)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Repair missing authored_by edges in LightRAG")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without posting")
    parser.add_argument("--checkpoint", default="repair_checkpoint.json",
                        help="Checkpoint file path (default: repair_checkpoint.json)")
    parser.add_argument("--reset-checkpoint", action="store_true",
                        help="Delete checkpoint before starting")
    parser.add_argument("--batch-delay", type=int, default=5,
                        help="Seconds to wait every 10 papers (default: 5)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    # 1. Optionally reset checkpoint
    if args.reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    # 2. Load checkpoint
    completed = load_checkpoint(checkpoint_path)

    # 3. Find missing papers, subtract completed
    print(f"Parsing GraphML: {GRAPHML_PATH}")
    missing_all = papers_missing_authored_by(GRAPHML_PATH)
    missing = missing_all - completed
    print(f"  Papers missing authored_by: {len(missing_all):,}  (already repaired: {len(completed):,})")

    # 4. Build lookups
    print(f"Building title→arxiv_key map from {REPORTS_DIR} ...")
    title_map = title_to_arxiv_key(REPORTS_DIR)
    print(f"  {len(title_map):,} title mappings found")

    print(f"Loading affiliation cache from {CACHE_PATH} ...")
    cache_map = cache_author_lookup(CACHE_PATH)
    print(f"  {len(cache_map):,} entries in cache")

    print(f"Loading CSV from {CSV_DIR} (may take a moment) ...")
    try:
        csv_map = csv_author_lookup(CSV_DIR)
        print(f"  {len(csv_map):,} entries in CSV")
    except FileNotFoundError as exc:
        print(f"  WARNING: {exc} — CSV lookup disabled")
        csv_map = {}

    # 5. Resolve each missing paper
    to_repair: list[tuple[str, dict]] = []
    skipped = 0
    for name in sorted(missing):
        authors = resolve_authors(name, title_map, csv_map, cache_map)
        if not authors:
            skipped += 1
        else:
            to_repair.append((name, authors))

    # 6. Summary
    print(f"\nSummary: {len(to_repair):,} papers to repair, {skipped:,} skipped (no author data)")

    # 7. Dry run early exit
    if args.dry_run:
        print("Dry run — exiting without posting.")
        return

    # 8. Post entities and relations
    n_authors = 0
    n_authored_by = 0
    n_affiliated_with = 0

    with requests.Session() as session:
        for idx, (entity_name, author_affiliations) in enumerate(to_repair):
            try:
                for author, inst_list in author_affiliations.items():
                    # POST Author entity
                    _post_entity(session, "Author", author)
                    n_authors += 1

                    # POST authored_by edge
                    _post_relation(
                        session,
                        source=entity_name,
                        target=author,
                        keywords="authored_by author",
                        description="Paper authored by researcher",
                    )
                    n_authored_by += 1

                    # POST Institution entities + affiliated_with edges
                    for inst in inst_list:
                        _post_entity(session, "Institution", inst)
                        _post_relation(
                            session,
                            source=author,
                            target=inst,
                            keywords="affiliated_with institution",
                            description="Researcher affiliated with institution",
                        )
                        n_affiliated_with += 1

                completed.add(entity_name)
                save_checkpoint(checkpoint_path, completed)

            except Exception as exc:
                print(f"  FAILED {entity_name!r} — {exc} — will retry on next run")
                continue

            # Every 10 papers, sleep
            if (idx + 1) % 10 == 0:
                print(f"  [{idx + 1}/{len(to_repair)}] sleeping {args.batch_delay}s ...")
                time.sleep(args.batch_delay)

    print(
        f"\nDone. New Author entities: {n_authors:,} | "
        f"authored_by edges: {n_authored_by:,} | "
        f"affiliated_with edges: {n_affiliated_with:,}"
    )


if __name__ == "__main__":
    main()

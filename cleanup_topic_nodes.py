#!/usr/bin/env python3
"""
Merge fragmented Topic nodes into their canonical form.

Earlier ingestion (before topic normalization) created many near-duplicate Topic
nodes that differ only by a parenthetical qualifier or by case/punctuation —
e.g. ~68 'Emerging Memory (ReRAM)', 'Emerging Memory (SOT-MRAM)', … alongside a
bare 'Emerging Memory'; 'System Level Memory Innovation' vs 'System level memory
innovation'. These fragment Leiden clustering.

This script groups Topic nodes by their canonical key (parenthetical stripped via
graph_ingest.normalize_topic, then case/punct-folded) and merges each group's
variants into one target node using LightRAG's POST /graph/entities/merge, which
moves all relationships onto the target and deletes the sources.

Genuinely distinct labels are preserved: 'HBM3' / 'HBM3E' / 'HBM4' and 'Emerging
Memory Devices' / 'Emerging Memory Concepts' fold to different keys and are left
alone — only true variants of the same base label collapse.

Usage:
    python3 cleanup_topic_nodes.py             # dry-run: print the merge plan
    python3 cleanup_topic_nodes.py --execute   # perform the merges
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from mira.graph_ingest import normalize_topic

ROOT = Path(__file__).parent
WORKING_DIR = ROOT / "lightrag" / "working_dir"
GRAPHML = WORKING_DIR / "graph_chunk_entity_relation.graphml"
PLAN_OUT = WORKING_DIR / "_merge_plan.json"
_MERGE_URL = "http://localhost:9621/graph/entities/merge"
_MERGE_TIMEOUT = 300


def _canonical_key(name: str) -> str:
    """Group key: strip parenthetical qualifier, then fold case/punctuation."""
    return re.sub(r"[^a-z0-9]", "", normalize_topic(name).lower())


def _pick_target(members: list[str]) -> str:
    """Choose the node to keep: prefer paren-free, then shortest, then alphabetical."""
    return min(members, key=lambda n: (1 if "(" in n else 0, len(n), n))


def plan_merges(topic_names: list[str]) -> list[tuple[str, list[str]]]:
    """Return [(target, [sources...])] for every group with at least one duplicate."""
    groups: dict[str, list[str]] = {}
    for name in topic_names:
        groups.setdefault(_canonical_key(name), []).append(name)

    plan: list[tuple[str, list[str]]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        target = _pick_target(members)
        sources = sorted(m for m in members if m != target)
        plan.append((target, sources))
    # Largest collapses first — most impactful, easiest to eyeball
    plan.sort(key=lambda ts: len(ts[1]), reverse=True)
    return plan


def load_topic_nodes(graphml: Path) -> list[str]:
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ET.parse(graphml).getroot()
    topics: list[str] = []
    for n in root.findall(".//g:node", ns):
        etype = ""
        for d in n.findall("g:data", ns):
            if d.get("key") == "d1":
                etype = (d.text or "").strip().strip('"')
        if etype == "Topic":
            topics.append(n.get("id"))
    return topics


def _merge(session: requests.Session, target: str, sources: list[str]) -> bool:
    resp = session.post(
        _MERGE_URL,
        json={"entities_to_change": sources, "entity_to_change_into": target},
        timeout=_MERGE_TIMEOUT,
    )
    if resp.status_code == 200:
        return True
    print(f"    ! merge failed ({resp.status_code}): {resp.text[:160]}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge fragmented Topic nodes")
    parser.add_argument("--execute", action="store_true",
                        help="Perform the merges over HTTP (default is dry-run)")
    parser.add_argument("--dump-plan", action="store_true",
                        help="Write the merge plan to working_dir/_merge_plan.json for the "
                             "in-container bulk_merge.py (safe path, no server load)")
    parser.add_argument("--graphml", default=str(GRAPHML))
    args = parser.parse_args()

    graphml = Path(args.graphml)
    if not graphml.exists():
        print(f"ERROR: {graphml} not found", file=sys.stderr)
        sys.exit(1)

    topics = load_topic_nodes(graphml)
    plan = plan_merges(topics)
    total_removed = sum(len(s) for _, s in plan)
    print(f"{len(topics)} Topic nodes → {len(plan)} merge groups, "
          f"removing {total_removed} duplicate nodes "
          f"({len(topics) - total_removed} remain)\n")

    for target, sources in plan:
        print(f"  '{target}'  ← {len(sources)} variant(s)")
        for s in sources[:6]:
            print(f"        {s}")
        if len(sources) > 6:
            print(f"        … and {len(sources) - 6} more")

    if args.dump_plan:
        PLAN_OUT.write_text(json.dumps([{"target": t, "sources": s} for t, s in plan]))
        print(f"\nWrote merge plan → {PLAN_OUT.relative_to(ROOT)} ({len(plan)} groups)")
        print("Run it safely in-container with bulk_merge.py (see comments there).")
        return

    if not args.execute:
        print("\nDry-run. Re-run with --execute (HTTP) or --dump-plan (in-container).")
        return

    print("\nExecuting merges...")
    ok = 0
    with requests.Session() as session:
        for i, (target, sources) in enumerate(plan, 1):
            print(f"  [{i}/{len(plan)}] merging {len(sources)} → '{target}'")
            if _merge(session, target, sources):
                ok += 1
    print(f"\nDone. {ok}/{len(plan)} groups merged.")


if __name__ == "__main__":
    main()

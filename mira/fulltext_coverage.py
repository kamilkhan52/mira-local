"""Does a LightRAG working dir actually have full text for its Paper nodes?

A Paper node is *covered* when a full-text document exists for its arXiv id.
Full-text docs are the ones `bulk_load_fulltext.py` writes, identified by a
`fulltext-{arxiv_id}` source_id. Bulk metadata chunks use
`{profile}-{date}-bulk` and deliberately do not count: they hold the title,
abstract and topic tags only, so counting them would report a metadata-only
graph as fully populated.

Spec: docs/superpowers/specs/2026-07-23-storage-fulltext-backfill-design.md
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
_ABS_RE = re.compile(r"arxiv\.org/abs/([^\s/·,)\]]+)")
_FULLTEXT_PREFIX = "fulltext-"


def recover_arxiv_id(node_name: str, description: str) -> str:
    """Recover the versioned arXiv id from a Paper node.

    Stub nodes are named `arxiv:{id}`; titled nodes carry the abstract URL in
    their description. Returns "" if neither yields an id.
    """
    if node_name.startswith("arxiv:"):
        return node_name[len("arxiv:"):]
    m = _ABS_RE.search(description or "")
    return m.group(1) if m else ""


def paper_arxiv_ids(graphml: Path) -> dict[str, str]:
    """{node_name: arxiv_id} for every Paper node ("" when unrecoverable)."""
    root = ET.parse(graphml).getroot()
    keys = {k.get("id"): k.get("attr.name") for k in root.findall("g:key", _NS)}
    out: dict[str, str] = {}
    for node in root.findall(".//g:node", _NS):
        data = {
            keys.get(d.get("key")): (d.text or "")
            for d in node.findall("g:data", _NS)
        }
        if (data.get("entity_type") or "").strip().strip('"') != "Paper":
            continue
        name = node.get("id")
        out[name] = recover_arxiv_id(name, data.get("description", ""))
    return out


def fulltext_ids(text_chunks_path: Path) -> set[str]:
    """arXiv ids that have a full-text document in this working dir."""
    chunks = json.loads(Path(text_chunks_path).read_text())
    return {
        str(v["source_id"])[len(_FULLTEXT_PREFIX):]
        for v in chunks.values()
        if str(v.get("source_id", "")).startswith(_FULLTEXT_PREFIX)
    }


@dataclass(frozen=True)
class Coverage:
    total: int
    covered: int
    missing: list[str]
    no_id: list[str]

    @property
    def pct(self) -> float:
        return 100.0 * self.covered / self.total if self.total else 0.0


def coverage(working_dir: Path) -> Coverage:
    working_dir = Path(working_dir)
    papers = paper_arxiv_ids(working_dir / "graph_chunk_entity_relation.graphml")
    have = fulltext_ids(working_dir / "kv_store_text_chunks.json")
    missing, no_id, covered = [], [], 0
    for name, aid in papers.items():
        if not aid:
            no_id.append(name)
        elif aid in have:
            covered += 1
        else:
            missing.append(name)
    return Coverage(
        total=len(papers), covered=covered, missing=missing, no_id=no_id)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("working_dir", type=Path, nargs="+")
    ap.add_argument("--list-missing", action="store_true",
                    help="print the name of every uncovered Paper node")
    args = ap.parse_args()
    for wd in args.working_dir:
        c = coverage(wd)
        print(f"{wd.name:<28} {c.covered:>5}/{c.total:<5} "
              f"({c.pct:5.1f}%)  missing={len(c.missing)}  no_id={len(c.no_id)}")
        if args.list_missing:
            for name in c.missing:
                print(f"    {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

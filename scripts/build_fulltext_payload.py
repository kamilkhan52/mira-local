#!/usr/bin/env python3
"""Build a citation-preserving full-text payload for an isolated graph target.

The builder is deliberately host-side: it reads the target graph and writes
payload files, but never opens a LightRAG store or contacts a graph service.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mira.config import ROOT
from mira.fulltext_coverage import recover_arxiv_id
from mira.graph_target import GraphTarget, resolve_target


GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s/·]+)")
BODY_CHARS = 4_800
OVERLAP_CHARS = 400
COVERAGE_THRESHOLD_PCT = 95.0


@dataclass(frozen=True)
class CacheDirs:
    pdf_library: Path
    full_text: Path
    full_pdfs: Path


@dataclass(frozen=True)
class BuildPaths:
    target: GraphTarget
    graphml: Path
    output_dir: Path
    cache_dirs: CacheDirs


def storage_cache_dirs(temp_dir: Path | None = None, graph: str = "storage") -> CacheDirs:
    """Return cache paths that are isolated from every other graph target."""
    root = temp_dir or ROOT / "scripts" / "temp"
    return CacheDirs(
        pdf_library=root / f"{graph}_pdf_library",
        full_text=root / f"{graph}_full_text_cache",
        full_pdfs=root / f"{graph}_full_pdfs",
    )


def resolve_build_paths(graph: str) -> BuildPaths:
    """Resolve all graph-owned paths through the common GraphTarget seam."""
    target = resolve_target(graph)
    return BuildPaths(
        target=target,
        graphml=target.graphml,
        output_dir=target.working_dir,
        cache_dirs=storage_cache_dirs(graph=target.name),
    )


def normalize_arxiv_id(value: str) -> str:
    """Return a version-preserving arXiv id from an id, URL, or stub name."""
    value = (value or "").strip()
    if value.startswith("arxiv:"):
        value = value[len("arxiv:"):]
    match = ARXIV_ID_RE.search(value)
    if match:
        value = match.group(1)
    return value.removesuffix(".pdf").strip()


def read_paper_nodes(graphml: Path) -> list[dict[str, str]]:
    """Read Paper nodes while accepting GraphML's generated or named data keys."""
    root = ET.parse(graphml).getroot()
    key_names = {
        key.get("id", ""): key.get("attr.name", "")
        for key in root.findall("g:key", GRAPHML_NS)
    }
    papers: list[dict[str, str]] = []
    for node in root.findall(".//g:node", GRAPHML_NS):
        values: dict[str, str] = {}
        for data in node.findall("g:data", GRAPHML_NS):
            key = data.get("key", "")
            name = key_names.get(key, {"d1": "entity_type", "d2": "description"}.get(key, key))
            values[name] = data.text or ""
        if values.get("entity_type", "").strip().strip('"') != "Paper":
            continue
        paper_name = node.get("id", "")
        metadata = "\n".join(values.get(field, "") for field in ("description", "file_path"))
        papers.append({
            "name": paper_name,
            "arxiv_id": recover_arxiv_id(paper_name, metadata),
        })
    return papers


def _cache_file(arxiv_id: str, cache_dirs: CacheDirs) -> Path:
    return cache_dirs.full_text / f"{normalize_arxiv_id(arxiv_id).replace('/', '_')}.json"


def cached_full_text(arxiv_id: str, cache_dirs: CacheDirs) -> str:
    """Return a completed extraction from this target's text cache, if present."""
    try:
        cached = json.loads(_cache_file(arxiv_id, cache_dirs).read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return ""
    if not cached.get("success"):
        return ""
    return (cached.get("full_text") or "").strip()


def download_full_text(arxiv_id: str, cache_dirs: CacheDirs) -> str:
    """Use the existing downloader with temporary, target-local cache settings."""
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    downloader = importlib.import_module("download_full_arxiv_pdf")
    old_paths = {
        "CACHE_DIR": downloader.CACHE_DIR,
        "PDF_LIBRARY_DIR": downloader.PDF_LIBRARY_DIR,
        "FULL_TEXT_CACHE_DIR": downloader.FULL_TEXT_CACHE_DIR,
    }
    downloader.CACHE_DIR = str(cache_dirs.full_pdfs)
    downloader.PDF_LIBRARY_DIR = str(cache_dirs.pdf_library)
    downloader.FULL_TEXT_CACHE_DIR = str(cache_dirs.full_text)
    try:
        result = downloader.download_and_extract(arxiv_id, use_cache=True)
    finally:
        for name, value in old_paths.items():
            setattr(downloader, name, value)
    return (result.get("full_text") or "").strip() if result.get("success") else ""


def acquire_full_text(arxiv_id: str, cache_dirs: CacheDirs) -> str:
    """Prefer target-local extracted text before invoking the PDF downloader."""
    return cached_full_text(arxiv_id, cache_dirs) or download_full_text(arxiv_id, cache_dirs)


def fulltext_chunks(title: str, arxiv_id: str, full_text: str) -> list[dict[str, str | int]]:
    """Create deterministic, overlapping chunks whose text carries its citation."""
    body = (full_text or "").strip()
    if not body:
        return []
    url = f"https://arxiv.org/abs/{arxiv_id}"
    header = f"Paper: {title}\nArXiv: {url}\n\n"
    source_id = f"fulltext-{arxiv_id}"
    chunks: list[dict[str, str | int]] = []
    start = 0
    order = 0
    while start < len(body):
        section = body[start:start + BODY_CHARS]
        content = header + section
        chunks.append({
            "content": content,
            "source_id": source_id,
            "file_path": url,
            "chunk_order_index": order,
            # The loader derives this same id from content; keeping it here makes
            # payload determinism auditable without coupling the builder to LightRAG.
            "chunk_id": "chunk-" + hashlib.md5(content.encode()).hexdigest(),
        })
        if start + BODY_CHARS >= len(body):
            break
        start += BODY_CHARS - OVERLAP_CHARS
        order += 1
    return chunks


def coverage_report(
    *,
    total_papers: int,
    full_text: list[str],
    abstract_only: list[str],
    no_arxiv_id: list[str],
    chunk_count: int,
) -> dict[str, object]:
    recoverable = len(full_text) + len(abstract_only)
    coverage_pct = 100.0 if recoverable == 0 else len(full_text) / recoverable * 100
    return {
        "total_papers": total_papers,
        "recoverable_arxiv_ids": recoverable,
        "full_text": full_text,
        "abstract_only": abstract_only,
        "no_arxiv_id": no_arxiv_id,
        "chunk_count": chunk_count,
        "coverage_pct": coverage_pct,
        "meets_95_percent_threshold": coverage_pct >= COVERAGE_THRESHOLD_PCT,
    }


def build_payload(paths: BuildPaths, sample: int = 0) -> tuple[dict[str, object], dict[str, object]]:
    """Build a payload and coverage report without touching a running graph."""
    papers = read_paper_nodes(paths.graphml)
    if sample:
        papers = papers[:sample]
    full_text_ids: list[str] = []
    abstract_only: list[str] = []
    no_arxiv_id: list[str] = []
    chunks: list[dict[str, str | int]] = []
    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        if not arxiv_id:
            no_arxiv_id.append(paper["name"])
            continue
        text = acquire_full_text(arxiv_id, paths.cache_dirs)
        if not text:
            abstract_only.append(arxiv_id)
            continue
        full_text_ids.append(arxiv_id)
        chunks.extend(fulltext_chunks(paper["name"], arxiv_id, text))
    payload: dict[str, object] = {"chunks": chunks}
    coverage = coverage_report(
        total_papers=len(papers),
        full_text=full_text_ids,
        abstract_only=abstract_only,
        no_arxiv_id=no_arxiv_id,
        chunk_count=len(chunks),
    )
    return payload, coverage


def write_outputs(paths: BuildPaths, payload: dict[str, object], coverage: dict[str, object]) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "_fulltext_payload.json").write_text(json.dumps(payload, indent=2))
    (paths.output_dir / "_fulltext_coverage.json").write_text(json.dumps(coverage, indent=2))
    # The bulk loader consumes this artifact before importing chunks.  Full-text
    # payloads do not currently require title renames, but the empty plan keeps
    # the loader contract uniform and makes reruns deterministic.
    (paths.output_dir / "_rename_plan.json").write_text(json.dumps([], indent=2))


def build_for_graph(graph: str, sample: int = 0) -> tuple[dict[str, object], dict[str, object]]:
    paths = resolve_build_paths(graph)
    payload, coverage = build_payload(paths, sample=sample)
    if not sample:
        write_outputs(paths, payload, coverage)
    return payload, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a graph target's full-text payload")
    parser.add_argument("--graph", required=True, choices=("memory", "optical", "storage"))
    parser.add_argument("--sample", type=int, default=0, help="Process N papers without writing files")
    args = parser.parse_args()
    payload, coverage = build_for_graph(args.graph, sample=args.sample)
    print(
        f"Built {coverage['chunk_count']} chunks for {coverage['full_text']}; "
        f"coverage {coverage['coverage_pct']:.1f}%"
    )
    if args.sample:
        print("SAMPLE mode — no files written.")
    else:
        print(f"Wrote payload and coverage report to {resolve_build_paths(args.graph).output_dir}")


if __name__ == "__main__":
    main()

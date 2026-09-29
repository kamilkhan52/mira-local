import json
from pathlib import Path
import subprocess
import sys

import pytest

from mira.graph_target import GraphTarget
from scripts import build_fulltext_payload as builder


def _target(tmp_path: Path, name: str = "storage") -> GraphTarget:
    return GraphTarget(
        name=name,
        working_dir=tmp_path / "working_dir_storage",
        base_url="http://localhost:9624",
        ledger_path=tmp_path / "ledger.json",
        venue_db=tmp_path / "venue.sqlite",
        novelty_cache=tmp_path / "novelty.json",
    )


def _write_graphml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="kind" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="description" for="node" attr.name="description" attr.type="string"/>
  <graph id="G" edgedefault="undirected">
    <node id="Cached paper"><data key="kind">Paper</data><data key="description">https://arxiv.org/abs/2401.00001v1</data></node>
    <node id="Unavailable paper"><data key="kind">Paper</data><data key="description">https://arxiv.org/abs/2401.00002v2</data></node>
    <node id="No identifier"><data key="kind">Paper</data><data key="description">Publisher-only paper</data></node>
    <node id="A topic"><data key="kind">Topic</data></node>
  </graph>
</graphml>
"""
    )


def test_cli_runs_from_the_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/build_fulltext_payload.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--graph" in result.stdout


def test_storage_graph_resolves_target_paths_and_never_memory_paths(tmp_path, monkeypatch):
    target = _target(tmp_path)
    calls = []

    def resolve(name):
        calls.append(name)
        return target

    monkeypatch.setattr(builder, "resolve_target", resolve)

    resolved = builder.resolve_build_paths("storage")

    assert calls == ["storage"]
    assert resolved.target is target
    assert resolved.graphml == target.graphml
    assert resolved.output_dir == target.working_dir
    assert resolved.cache_dirs.full_text.name == "storage_full_text_cache"
    assert resolved.cache_dirs.full_text != builder.storage_cache_dirs(graph="memory").full_text
    assert "storage_" in resolved.cache_dirs.pdf_library.name
    assert "storage_" in resolved.cache_dirs.full_text.name


def test_cached_text_is_used_before_the_downloader(tmp_path, monkeypatch):
    cache_dirs = builder.storage_cache_dirs(tmp_path / "scripts" / "temp")
    cache_dirs.full_text.mkdir(parents=True)
    cache_path = cache_dirs.full_text / "2401.00001v1.json"
    cache_path.write_text(json.dumps({"success": True, "full_text": "cached body"}))

    def should_not_download(*_args, **_kwargs):
        raise AssertionError("cached full text should not invoke the downloader")

    monkeypatch.setattr(builder, "download_full_text", should_not_download)

    assert builder.acquire_full_text("2401.00001v1", cache_dirs) == "cached body"


def test_build_collects_failed_pdf_fallbacks_and_writes_storage_reports(tmp_path, monkeypatch):
    target = _target(tmp_path)
    _write_graphml(target.graphml)
    monkeypatch.setattr(builder, "resolve_target", lambda _name: target)

    def acquire(arxiv_id, _cache_dirs):
        return "Extracted body." if arxiv_id == "2401.00001v1" else ""

    monkeypatch.setattr(builder, "acquire_full_text", acquire)

    payload, coverage = builder.build_for_graph("storage")

    assert coverage["full_text"] == ["2401.00001v1"]
    assert coverage["abstract_only"] == ["2401.00002v2"]
    assert coverage["no_arxiv_id"] == ["No identifier"]
    assert coverage["chunk_count"] == len(payload["chunks"]) == 1
    chunk = payload["chunks"][0]
    assert chunk["content"].startswith(
        "Paper: Cached paper\nArXiv: https://arxiv.org/abs/2401.00001v1\n\n"
    )
    assert chunk["file_path"] == "https://arxiv.org/abs/2401.00001v1"
    assert chunk["source_id"] == "fulltext-2401.00001v1"
    assert (target.working_dir / "_fulltext_payload.json").exists()
    assert (target.working_dir / "_fulltext_coverage.json").exists()
    assert json.loads((target.working_dir / "_rename_plan.json").read_text()) == []


def test_fulltext_chunks_are_deterministic_and_cite_the_source_url():
    text = "x" * (builder.BODY_CHARS + 1)

    first = builder.fulltext_chunks("Deterministic paper", "2401.99999v3", text)
    second = builder.fulltext_chunks("Deterministic paper", "2401.99999v3", text)

    citation_url = "https://arxiv.org/abs/2401.99999v3"
    assert [chunk["chunk_id"] for chunk in first] == [chunk["chunk_id"] for chunk in second]
    assert [chunk["chunk_order_index"] for chunk in first] == [0, 1]
    assert [chunk["file_path"] for chunk in first] == [citation_url, citation_url]
    assert all(f"ArXiv: {citation_url}" in chunk["content"] for chunk in first)


def test_coverage_calculation_uses_only_recoverable_arxiv_ids():
    coverage = builder.coverage_report(
        total_papers=4,
        full_text=["one", "two"],
        abstract_only=["three"],
        no_arxiv_id=["four"],
        chunk_count=9,
    )

    assert coverage["recoverable_arxiv_ids"] == 3
    assert coverage["coverage_pct"] == pytest.approx(2 / 3 * 100)
    assert coverage["meets_95_percent_threshold"] is False
    assert coverage["chunk_count"] == 9

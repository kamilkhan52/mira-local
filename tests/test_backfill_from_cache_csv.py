import json
from pathlib import Path

import pytest

from backfill_from_cache_csv import (
    batch_papers,
    build_title_lookup,
    csv_row_to_paper,
    iso_week_monday,
    load_checkpoint,
    save_checkpoint,
)

_BASE_ROW = {
    "arxiv_key": "2604.05285v1",
    "arxiv_id": "http://arxiv.org/abs/2604.05285v1",
    "relevance_score": "8",
    "credibility_tier": "7",
    "primary_topic": "HBM Memory",
    "secondary_topics_json": '["DRAM Design", "3D Stacking"]',
    "key_findings": "Key finding here",
    "potential_impact": "High impact on industry",
    "affiliations_json": '["MIT"]',
    "authors_json": '["Alice Smith"]',
    "author_affiliations_json": '{"Alice Smith": ["MIT"]}',
}


def test_iso_week_monday_wednesday():
    assert iso_week_monday("2026-04-15T10:00:00Z") == "2026-04-13"


def test_iso_week_monday_already_monday():
    assert iso_week_monday("2026-04-13T00:00:00Z") == "2026-04-13"


def test_iso_week_monday_sunday():
    assert iso_week_monday("2026-04-19T23:59:59Z") == "2026-04-13"


def test_csv_row_to_paper_uses_title_lookup():
    paper = csv_row_to_paper(_BASE_ROW, {"2604.05285v1": "A Great Memory Paper"})
    assert paper["title"] == "A Great Memory Paper"


def test_csv_row_to_paper_fallback_title():
    paper = csv_row_to_paper({**_BASE_ROW, "arxiv_key": "2604.99999v1"}, {})
    assert paper["title"] == "arxiv:2604.99999v1"


def test_csv_row_to_paper_fields():
    paper = csv_row_to_paper(_BASE_ROW, {})
    assert paper["raw_id"] == "http://arxiv.org/abs/2604.05285v1"
    assert paper["relevance_score"] == 8
    assert paper["credibility_tier"] == 7
    assert paper["primary_topic"] == "HBM Memory"
    assert paper["secondary_topics"] == ["DRAM Design", "3D Stacking"]
    assert paper["affiliations"] == ["MIT"]
    assert paper["author_affiliations"] == {"Alice Smith": ["MIT"]}
    assert paper["key_findings"] == "Key finding here"
    assert paper["short_summary"] == "High impact on industry"


def test_csv_row_to_paper_bad_json_defaults():
    row = {**_BASE_ROW, "secondary_topics_json": "INVALID", "affiliations_json": ""}
    paper = csv_row_to_paper(row, {})
    assert paper["secondary_topics"] == []
    assert paper["affiliations"] == []


def test_build_title_lookup(tmp_path):
    data = [{"body_markdown": "[Test Paper](https://arxiv.org/abs/2604.05285v1) — MIT\n\nFindings."}]
    (tmp_path / "report.json").write_text(json.dumps(data))
    lookup = build_title_lookup(tmp_path)
    assert lookup.get("2604.05285v1") == "Test Paper"


def test_build_title_lookup_multiple_reports(tmp_path):
    r1 = [{"body_markdown": "[Paper One](https://arxiv.org/abs/2601.00001v1) — MIT"}]
    r2 = [{"body_markdown": "[Paper Two](https://arxiv.org/abs/2602.00002v2) — Stanford"}]
    (tmp_path / "r1.json").write_text(json.dumps(r1))
    (tmp_path / "r2.json").write_text(json.dumps(r2))
    lookup = build_title_lookup(tmp_path)
    assert lookup["2601.00001v1"] == "Paper One"
    assert lookup["2602.00002v2"] == "Paper Two"


def test_build_title_lookup_ignores_malformed_json(tmp_path):
    (tmp_path / "bad.json").write_text("not json at all")
    lookup = build_title_lookup(tmp_path)
    assert lookup == {}


def test_load_checkpoint_missing_file(tmp_path):
    assert load_checkpoint(tmp_path / "cp.json") == set()


def test_load_checkpoint_existing(tmp_path):
    p = tmp_path / "cp.json"
    p.write_text('["2026-04-06", "2026-04-13"]')
    assert load_checkpoint(p) == {"2026-04-06", "2026-04-13"}


def test_load_checkpoint_corrupt_file(tmp_path):
    p = tmp_path / "cp.json"
    p.write_text("not json")
    assert load_checkpoint(p) == set()


def test_save_checkpoint(tmp_path):
    p = tmp_path / "cp.json"
    save_checkpoint(p, {"2026-04-13", "2026-04-06"})
    assert json.loads(p.read_text()) == ["2026-04-06", "2026-04-13"]  # sorted


def test_save_checkpoint_overwrites(tmp_path):
    p = tmp_path / "cp.json"
    save_checkpoint(p, {"2026-04-06"})
    save_checkpoint(p, {"2026-04-06", "2026-04-13"})
    assert load_checkpoint(p) == {"2026-04-06", "2026-04-13"}


def test_batch_papers_splits_correctly():
    papers = [{"title": f"p{i}"} for i in range(10)]
    batches = batch_papers(papers, 3)
    assert len(batches) == 4
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_batch_papers_exact_multiple():
    papers = [{"title": f"p{i}"} for i in range(6)]
    assert [len(b) for b in batch_papers(papers, 3)] == [3, 3]


def test_batch_papers_smaller_than_batch():
    papers = [{"title": f"p{i}"} for i in range(2)]
    batches = batch_papers(papers, 50)
    assert len(batches) == 1
    assert len(batches[0]) == 2

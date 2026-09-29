import json
from pathlib import Path

import pytest

from mira import jev

ROOT = Path(__file__).resolve().parent.parent


def _profiles():
    d = json.loads((ROOT / "configs" / "memory-innovation-profile.json").read_text())
    return {p["profile_id"]: p for p in d["profiles"]}


@pytest.mark.parametrize("pid", ["memory-innovation", "cxl-research",
                                 "storage-innovation", "optical-interconnects"])
def test_profile_rubric_extracts_taxonomy_and_drops_output_rules(pid):
    r = jev.profile_rubric(_profiles()[pid])
    assert r["team"] and r["focus"]
    assert len(r["taxonomy"]) >= 5
    assert r["taxonomy"][-1].lower().startswith("not related") or any(
        t.lower().startswith("not related") for t in r["taxonomy"])
    joined = " ".join(r["guidance"] + r["taxonomy"])
    for leaked in ("JSON", "1-10", "relevance_score", "primary_topic", "specify"):
        assert leaked not in joined


def test_memory_rubric_keeps_not_related_rule():
    r = jev.profile_rubric(_profiles()["memory-innovation"])
    assert any("treat it as Not related to memory" in g for g in r["guidance"])


@pytest.mark.parametrize("level,expected", [(0, 2), (1, 4), (2, 6), (3, 8), (4, 10)])
def test_relevance_mapping_hits_band_midpoints(level, expected):
    assert jev.relevance_to_10(level) == expected


def test_credibility_mapping_is_monotonic_and_bounded():
    vals = [jev.credibility_to_10(x / 10) for x in range(0, 31)]
    assert vals == sorted(vals)
    assert vals[0] >= 1 and vals[-1] <= 10


def test_prescreen_splits_on_cutoff_and_fails_open(monkeypatch):
    levels = {"a": 0.01, "b": 0.9, "c": None}

    def fake_judge(p, rubric):
        if levels[p["id"]] is None:
            raise RuntimeError("service down")
        return {"relevance_level": levels[p["id"]], "relevance_confidence": 0.9,
                "primary_topic": "Not related to memory"}

    monkeypatch.setattr(jev, "judge_paper", fake_judge)
    prof = _profiles()["memory-innovation"]
    papers = [{"id": k, "title": k, "summary": k} for k in levels]
    kept, screened = jev.prescreen(papers, prof)
    assert [p["id"] for p in screened] == ["a"]
    assert [p["id"] for p in kept] == ["b", "c"]
    assert screened[0]["jev_prescreen"]["relevance_level"] == 0.01


def test_prescreen_is_noop_without_validated_cutoff(monkeypatch):
    monkeypatch.setattr(jev, "judge_paper", lambda *a: pytest.fail("should not call Jev"))
    prof = _profiles()["optical-interconnects"]
    papers = [{"id": "x", "title": "t", "summary": "s"}]
    assert jev.prescreen(papers, prof) == (papers, [])

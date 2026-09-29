import json

import pytest

from mira import jev, realtime


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(realtime, "STATE_DIR", tmp_path)
    return tmp_path


def _config():
    from pathlib import Path
    d = json.loads((Path(__file__).resolve().parent.parent / "configs" /
                    "memory-innovation-profile.json").read_text())
    p = next(x for x in d["profiles"] if x["profile_id"] == "memory-innovation")
    return {**p, "profile_id": "memory-innovation"}


def test_ledger_roundtrip(state_dir):
    led = realtime.Ledger("p")
    assert led.is_new("a")
    led.mark(["a", "b"])
    led.save()
    again = realtime.Ledger("p")
    assert not again.is_new("a") and not again.is_new("b") and again.is_new("c")


def test_load_settings_resolves_env_subscribers(monkeypatch):
    monkeypatch.setenv("RECIPIENT_EMAIL", "someone@example.com")
    s = realtime.load_settings()
    assert s["subscribers"]["memory-innovation"] == ["someone@example.com"]
    monkeypatch.delenv("RECIPIENT_EMAIL")
    assert realtime.load_settings()["subscribers"]["memory-innovation"] == []


def test_triage_papers_gates_ranks_and_skips_failed(monkeypatch):
    levels = {"low": 0.05, "mid": 0.5, "top": 2.5, "boom": None}

    def fake_judge(p, rubric):
        if levels[p["id"]] is None:
            raise RuntimeError("down")
        return {"relevance_level": levels[p["id"]], "relevance_score": 5,
                "primary_topic": "PIM", "relevance_confidence": 0.9}

    monkeypatch.setattr(jev, "judge_paper", fake_judge)
    monkeypatch.setattr(jev, "judge_credibility",
                        lambda focus, first_page_text: {"credibility_level": 2.0, "credibility_tier": 7})
    import mira.fetch
    monkeypatch.setattr(mira.fetch, "extract_first_pages",
                        lambda ps: [p.update(first_page_text="header") or p for p in ps])
    papers = [{"id": k, "title": k, "summary": k} for k in levels]
    passing, log = realtime.triage_papers(papers, _config())
    assert [p["id"] for p in passing] == ["top", "mid"]
    assert passing[0]["priority"] and not passing[1]["priority"]
    assert {r["id"] for r in log} == {"low", "mid", "top"}  # failed call is retried next run
    assert {r["id"] for r in log if r["alerted"]} == {"mid", "top"}


def test_build_digest_includes_abstract_summary_and_overflow():
    cfg = _config()
    paper = {"id": "2609.00001", "title": "T", "summary": "An abstract.", "published": "2026-09-27",
             "authors": ["A"], "jev": {"primary_topic": "PIM", "relevance_score": 9},
             "jev_cred": {"credibility_tier": 8}, "priority": True,
             "alert_summary": "Why it matters.", "summary_how": "llm"}
    news = {"title": "N", "url": "https://x", "source": "EE Times", "content": "Body",
            "alert_summary": "Key.", "summary_how": "key sentence"}
    subject, body = realtime.build_digest(cfg, [paper], [], [news], 10, 2, n_more=3)
    assert "1 paper" in subject and "1 news" in subject
    for s in ("https://arxiv.org/abs/2609.00001", "**Abstract:** An abstract.",
              "**Summary:** Why it matters.", "High priority", "EE Times",
              "Summary (key sentence)"):
        assert s in body

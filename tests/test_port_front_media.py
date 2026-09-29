# tests/test_port_front_media.py — "Media - *" nodes port (no crawling, no LLM)
from _paths import repoint
import json
from unittest.mock import MagicMock, patch

import pytest

import mira.media as media
from mira.media import (apply_selection, build_selection_prompt, build_summarize_prompt, filter_to_window,
                        map_to_pipeline, parse_summarized, prepare_media_output, run_media)
from test_port_front_n8n_js import requires_node, run_node

CONFIG = {
    "profile_id": "memory-innovation", "mode": "weekly",
    "topic": {"name": "Memory Technology Research", "focus": "memory technology"},
    "prompts": {}, "media": {"selection_guidance": "ignored, as in n8n"},
    "start_date_iso": "2026-09-21", "end_date_iso": "2026-09-28", "current_date": "2026-09-28",
    "period_label": "last 8 days", "period_range": "2026-09-21 to 2026-09-28",
    "llm_models": {"media_selection": "sel-model", "media_summary": "sum-model"},
}
RUN_MODE = {"currentDate": "2026-09-28", "periodLabel": "last 8 days",
            "periodRange": "2026-09-21 to 2026-09-28", "topicFocus": "memory technology"}
MEDIA_CFG = {"config": {"topic": CONFIG["topic"], "prompts": {}}, "topicFocus": "memory technology",
             "period_label": "last 8 days", "period_range": "2026-09-21 to 2026-09-28"}

RAW_EE = [
    {"url": "https://ee/1", "title": " HBM4 {{x}} ", "listAuthor": "Ann", "listDate": "09.25.2026",
     "content": "  Body one " + "y" * 7000},
    {"url": "https://ee/2", "listTitle": "List only", "listDate": "September 21, 2026", "content": "b2"},
    {"url": "https://ee/3", "title": "Old", "listDate": "09.17.2026", "content": "b3"},
    {"url": "https://ee/4", "title": "Undated", "listDate": "", "content": "b4"},
    {"url": "https://ee/5", "title": "ISO", "listDate": "2026-09-28T19:26:41.104Z", "content": "b5"},
    {"url": "https://ee/6", "title": "Future", "listDate": "2026-09-29", "content": "b6"},
]


@requires_node
def test_map_to_pipeline_matches_n8n(tmp_path):
    js = run_node(tmp_path, "Media - Map EE Times to Pipeline", [{"stdout": json.dumps(RAW_EE)}],
                  {"Set Run Mode": [RUN_MODE]})
    assert map_to_pipeline(RAW_EE, "eetimes", "2026-09-28") == js


def _mapped():
    return map_to_pipeline(RAW_EE, "eetimes", "2026-09-28") + map_to_pipeline(
        [{"url": "https://sa/1", "title": "SA", "listDate": "2026-09-26T13:36:32.170Z", "content": "c"}],
        "semianalysis", "2026-09-28")


@requires_node
def test_window_filter_matches_n8n(tmp_path):
    summaries = [{"type": "eetimes_summary", "date_from": "2026-09-21", "date_to": "2026-09-28"}]
    js = run_node(tmp_path, "Media - Split Summary and Articles", summaries + _mapped(),
                  {"Set Run Mode": [RUN_MODE]})[0]
    py = filter_to_window(_mapped(), CONFIG)
    assert py == js
    assert [a["title"] for a in py["articles"]] == ["HBM4 {{x}}", "List only", "ISO", "SA"]


@requires_node
def test_selection_prompt_matches_n8n(tmp_path):
    item = filter_to_window(_mapped() * 5, CONFIG)
    js = run_node(tmp_path, "Media - Build Selection Prompt", [item],
                  {"Set Run Mode": [RUN_MODE], "Media - Load Config": [MEDIA_CFG]})[0]
    user, system = build_selection_prompt(item, CONFIG)
    assert (user, system) == (js["selection_prompt"], js["selection_system"])
    assert "Select up to 5 articles from the following 20 articles" in user
    assert "{{x}}" not in user  # n8n's applyTemplate blanks braces even in article text


@requires_node
@pytest.mark.parametrize("agent_json,py_raw", [
    ({"output": {"selected_indices": [3, 0, 99, "1"]}}, {"selected_indices": [3, 0, 99, "1"]}),
    ({"output": {"selected_papers": [{"index": 2}, {"index": 0}]}}, {"selected_papers": [{"index": 2}, {"index": 0}]}),
    ({"output": {"nothing": True}}, {"nothing": True}),
])
def test_apply_selection_matches_n8n(tmp_path, agent_json, py_raw):
    item = filter_to_window(_mapped() * 2, CONFIG)
    js = run_node(tmp_path, "Media - Apply Selection", [agent_json], {"Media - Build Selection Prompt": [item]})[0]
    assert apply_selection(item, py_raw) == js
    assert apply_selection(item, json.dumps(py_raw)) == js  # raw model text parses the same


def test_apply_selection_regex_fallback_and_error():
    item = {"summary": {}, "articles": [{"i": i} for i in range(8)]}
    assert apply_selection(item, 'Sure! "selected_indices": [6, 7]')["articles"] == [{"i": 6}, {"i": 7}]
    assert apply_selection(item, {})["article_count"] == 5  # agent error -> first 5


@requires_node
@pytest.mark.parametrize("model_text", [
    '```json\n[{"title": "T1", "short_summary": "S1", "date": "Sep 25, 2026", "source": "eetimes"},'
    ' {"short_summary": null, "date": "not stated"}, "junk"]\n```',
    '{"articles": [{"title": "only", "date": "2026-09-22"}]}',
    'not json at all',
])
def test_summarize_prompt_and_parse_match_n8n(tmp_path, model_text):
    item = filter_to_window(_mapped(), CONFIG)
    js_prompt = run_node(tmp_path, "Media - Build Summarize Prompt", [item],
                         {"Set Run Mode": [RUN_MODE], "Media - Load Config": [MEDIA_CFG]})[0]
    assert build_summarize_prompt(item, CONFIG) == (js_prompt["summarize_prompt"], js_prompt["summarize_system"])
    js_parsed = run_node(tmp_path, "Media - Parse Summarized Articles", [{"output": model_text}],
                         {"Media - Build Summarize Prompt": [js_prompt]})[0]
    py_parsed = parse_summarized(item, model_text)
    assert py_parsed == js_parsed
    js_out = run_node(tmp_path, "Media - Prepare Media Output For Parent", [js_parsed])[0]
    assert prepare_media_output(py_parsed) == js_out


def test_prepare_media_output_empty():
    out = prepare_media_output({"summary": {"date_from": "a", "date_to": "b"}, "articles": []})
    assert out["media_period_range"] == "a to b" and out["media_article_count"] == 0
    assert out["media_markdown"] == "- No media articles available for this period."


def test_run_media_flow_selects_above_15_and_summarizes(monkeypatch):
    calls = []

    def fake_llm(client, model, system, user, **kw):
        calls.append(model)
        if model == "sel-model":
            return '{"selected_indices": [1, 2]}'
        return json.dumps([{"title": "A", "short_summary": "sa", "date": "2026-09-25", "source": "eetimes"},
                           {"title": "B", "short_summary": "sb", "date": "", "source": "semianalysis"}])
    monkeypatch.setattr("mira.config.llm_call", fake_llm)
    out = run_media(CONFIG, client=None, articles=_mapped() * 5)  # 20 in window > 15
    assert calls == ["sel-model", "sum-model"]
    assert out["media_article_count"] == 2 and out["media_period_range"] == "2026-09-21 to 2026-09-28"
    assert [a["url"] for a in out["media_articles"]] == ["https://ee/2", "https://ee/5"]
    assert set(out) == {"media_intelligence", "media_period_range", "media_article_count",
                        "media_articles", "media_markdown"}

    calls.clear()
    out = run_media(CONFIG, client=None, articles=_mapped())  # 4 in window: no selection call
    assert calls == ["sum-model"] and out["media_article_count"] == 4


def test_run_media_survives_llm_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("provider down")
    monkeypatch.setattr("mira.config.llm_call", boom)
    out = run_media(CONFIG, client=None, articles=_mapped() * 5)
    assert out["media_article_count"] == 5  # first five, excerpts as summaries
    assert out["media_articles"][0]["short_summary"].startswith("Body one")


def test_crawl_sources_uses_n8n_settings(tmp_path, monkeypatch):
    repoint(monkeypatch, "mira.media", tmp_path)
    runs = []

    def fake_run(cmd, env, cwd, capture_output, text, timeout):
        runs.append((cmd[-1], env["MAX_ARTICLES"], timeout, env["N8N_WEBHOOK_URL"], env.get("LIST_URL"),
                     env["OUTPUT_DIR"]))
        from pathlib import Path
        Path(env["OUTPUT_PATH"]).write_text(json.dumps([{"url": "u", "title": "t", "listDate": "2026-09-25"}]))
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/x")
    monkeypatch.delenv("MIRA_MEDIA_DIGITIMES", raising=False)
    with patch("mira.media.subprocess.run", side_effect=fake_run):
        arts = media.crawl_sources(CONFIG)
        assert sorted(r[0] for r in runs) == ["ee-times-crawler.ts", "semianalysis-crawler.ts",
                                              "trendforce-crawler.ts"]
        assert {r[1:5] for r in runs} == {("0", 600, "", None)}
        assert all(r[5].endswith("crawler-output") for r in runs)
        assert [a["source"] for a in arts] == ["eetimes", "semianalysis", "trendforce"]
        runs.clear()
        media.crawl_sources({**CONFIG, "media": {"digitimes": {"enabled": True}}})
        assert "digitimes-crawler.ts" in [r[0] for r in runs]
        runs.clear()
        monkeypatch.setenv("MIRA_MEDIA_DIGITIMES", "1")
        media.crawl_sources(CONFIG, include_digitimes=False)
        assert "digitimes-crawler.ts" not in [r[0] for r in runs]


def test_run_crawler_never_reuses_a_stale_file_and_keeps_partial_output(tmp_path, monkeypatch):
    repoint(monkeypatch, "mira.media", tmp_path)
    out_dir = tmp_path / "scripts" / "temp"
    out_dir.mkdir(parents=True)
    (out_dir / "eetimes-latest.json").write_text(json.dumps([{"title": "stale"}]))
    with patch("mira.media.subprocess.run", return_value=MagicMock(returncode=1, stderr="boom")):
        assert media._run_crawler("ee-times-crawler.ts", CONFIG, "eetimes") == []

    def partial(*a, **k):
        (out_dir / "eetimes-latest.json").write_text(json.dumps([{"title": "fresh"}]))
        return MagicMock(returncode=1, stderr="webhook failed")
    with patch("mira.media.subprocess.run", side_effect=partial):
        assert media._run_crawler("ee-times-crawler.ts", CONFIG, "eetimes") == [{"title": "fresh"}]


def test_fetch_media_keeps_list_contract(monkeypatch):
    monkeypatch.setattr(media, "run_media", lambda config, client, include_digitimes=None: {
        "media_articles": [{"title": "A", "short_summary": "s", "date": "", "url": "u", "source": "eetimes"}]})
    assert media.fetch_media(CONFIG, None) == [
        {"title": "A", "short_summary": "s", "date": "", "url": "u", "source": "eetimes", "summary": "s"}]

# tests/test_port_front_config.py — run configuration + LLM client port
from _paths import repoint
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from mira import config as cfgmod
from mira.config import (duration_label, js_number, load_config, llm_call, make_llm_client,
                         parse_bool, resolve_profile, title_case)
from test_port_front_n8n_js import requires_node, run_node

LIVE = json.loads((Path(__file__).parent.parent / "configs" /
                   "memory-innovation-profile.json").read_text())
NOW = datetime(2026, 9, 28, 14, 30)


@pytest.fixture
def live_configs(tmp_path, monkeypatch):
    """The real profile config, isolated from other files in configs/."""
    d = tmp_path / "configs"
    d.mkdir()
    (d / "memory-innovation-profile.json").write_text(json.dumps(LIVE))
    # list-rooted crawler snapshots live in configs/ in real checkouts
    (d / "eetimes-latest.json").write_text("[]")
    repoint(monkeypatch, "mira.config", tmp_path)
    monkeypatch.delenv("RECIPIENT_EMAIL", raising=False)
    monkeypatch.delenv("MIRA_LLM_CACHE_BYPASS", raising=False)
    return d


# --- routing --------------------------------------------------------------

@pytest.mark.parametrize("trigger,expected", [
    ("CXL Monthly Trigger", ("cxl-research", "monthly")),
    ("MCBU Weekly Trigger", ("mcbu-memory", "weekly")),
    ("AEBU Weekly Trigger", ("embedded-intelligence", "weekly")),
    ("Optical Weekly Trigger", ("optical-interconnects", "weekly")),
    ("Daily Trigger", ("memory-innovation", "daily")),
    ("Weekly Trigger", ("memory-innovation", "weekly")),
    ("Backfill Trigger", ("memory-innovation", "weekly")),
    ("", ("memory-innovation", "weekly")),
    (None, ("memory-innovation", "weekly")),
])
def test_resolve_profile_routes_like_load_topic_config(live_configs, trigger, expected):
    assert resolve_profile(trigger) == expected


def test_resolve_profile_refuses_unknown_trigger(live_configs):
    with pytest.raises(ValueError, match="No profile matched trigger"):
        resolve_profile("Quarterly Trigger")


def test_cxl_trigger_uses_profile_default_mode_when_name_has_none(live_configs):
    assert resolve_profile("CXL Trigger") == ("cxl-research", "monthly")


def test_load_config_by_trigger_name(live_configs):
    c = load_config(trigger_name="CXL Monthly Trigger", now=NOW)
    assert (c["profile_id"], c["mode"]) == ("cxl-research", "monthly")
    assert c["lookback_days"] == 30
    assert c["period_label"] == "last month" and c["period_title"] == "Last Month"
    assert c["start_date_iso"] == "2026-08-30" and c["end_date_iso"] == "2026-09-28"
    assert c["period_range"] == "2026-08-30 to 2026-09-28"
    assert c["digest_label"] == "Compute Express Link (CXL) Research Digest (Last Month)"
    assert c["email_cfg"] == {}  # cxl-research has no monthly email block
    assert c["profile_fallback"] is False


# --- date window / labels ---------------------------------------------------

@pytest.mark.parametrize("profile,mode,kw,start,end,label", [
    ("memory-innovation", "daily", {}, "2026-09-28", "2026-09-28", "last day"),
    ("memory-innovation", "weekly", {}, "2026-09-21", "2026-09-28", "last 8 days"),
    ("storage-innovation", "weekly", {}, "2026-09-22", "2026-09-28", "last week"),
    ("cxl-research", "monthly", {}, "2026-08-30", "2026-09-28", "last month"),
    ("cxl-research", "monthly", {"current_date": "2026-09-01"}, "2026-08-03", "2026-09-01", "last month"),
    ("memory-innovation", "weekly", {"current_date": "2026-08-31T09:00:00Z", "lookback_days": 14},
     "2026-08-18", "2026-08-31", "last 2 weeks"),
    ("memory-innovation", "weekly", {"lookback_days": "60"}, "2026-07-31", "2026-09-28", "last 2 months"),
    ("memory-innovation", "weekly", {"lookback_days": 0}, "2026-09-21", "2026-09-28", "last 8 days"),
    ("memory-innovation", "weekly", {"current_date": "not-a-date"}, "2026-09-21", "2026-09-28", "last 8 days"),
    ("memory-innovation", "daily", {"current_date": "2026-02-31"}, "2026-09-28", "2026-09-28", "last day"),
])
def test_date_window(live_configs, profile, mode, kw, start, end, label):
    c = load_config(profile, mode, now=NOW, **kw)
    assert (c["start_date_iso"], c["end_date_iso"], c["period_label"]) == (start, end, label)
    assert c["current_date"] == end
    assert c["start_date"] == start.replace("-", "") and c["end_date"] == end.replace("-", "")


def test_cli_window_still_supported(live_configs):
    c = load_config("memory-innovation", "weekly", start_date="2026-09-01", end_date="2026-09-14")
    assert (c["start_date"], c["end_date"], c["lookback_days"]) == ("20260901", "20260914", 14)
    assert c["period_label"] == "last 2 weeks"


def test_overrides_and_flags(live_configs, monkeypatch):
    c = load_config("memory-innovation", "daily", now=NOW)
    assert c["test_mode"] is False and c["llm_cache_bypass"] is False
    # n8n's Set Lookback Override hardcodes trend_enabled_override = true
    assert c["trend_enabled"] is True and c["trend_enabled_profile"] is True
    assert c["max_limit"] is None and c["recipient_email"] is None
    c = load_config("memory-innovation", "daily", now=NOW, test_mode="true", llm_cache_bypass="yes",
                    trend_enabled="false", max_limit="25", recipient_email="a@b.c")
    assert c["test_mode"] is True and c["llm_cache_bypass"] is True
    assert c["trend_enabled"] is False and c["max_limit"] == 25
    assert c["recipient_email"] == "a@b.c"
    monkeypatch.setenv("RECIPIENT_EMAIL", "env@b.c")
    monkeypatch.setenv("MIRA_LLM_CACHE_BYPASS", "1")
    c = load_config("memory-innovation", "daily", now=NOW)
    assert c["recipient_email"] == "env@b.c" and c["llm_cache_bypass"] is True
    with pytest.raises(ValueError, match="max_limit"):
        load_config("memory-innovation", "daily", now=NOW, max_limit="lots")


def test_parse_bool_matches_set_run_mode():
    for v in (True, "true", 1, "1", "yes", "on"):
        assert parse_bool(v) is True
    for v in (False, "false", 0, "TRUE", "no", 2, "Yes"):
        assert parse_bool(v, True) is False
    assert parse_bool(None, True) is True and parse_bool("", True) is True


def test_js_number_and_labels():
    assert [js_number(v) for v in (None, "", " 8 ", "7.5", True, "abc", [], ["3"], "0x10")] == \
        [0, 0, 8, 7.5, 1, None, 0, 3, 16]
    assert [duration_label(d) for d in (0, 1, 3, 7, 14, 21, 30, 45, 56, 60, 63, 90)] == [
        "recent period", "last day", "last 3 days", "last week", "last 2 weeks", "last 3 weeks",
        "last month", "last 45 days", "last 8 weeks", "last 2 months", "last 63 days", "last 3 months"]
    assert title_case("last 2 weeks") == "Last 2 Weeks"


@requires_node
@pytest.mark.parametrize("profile_id,trigger,override", [
    ("memory-innovation", "Daily Trigger", {}),
    ("memory-innovation", "Weekly Trigger", {"current_date_override": "2026-08-31"}),
    ("memory-innovation", "Backfill Trigger", {"lookback_days_override": 45, "test_mode_override": "on"}),
    ("cxl-research", "CXL Monthly Trigger", {"current_date_override": "2026-09-01T06:00"}),
    ("mcbu-memory", "MCBU Weekly Trigger", {"lookback_days_override": "21", "llm_cache_bypass": 1}),
    ("storage-innovation", "Storage Weekly Trigger", {"current_date_override": "garbage"}),
])
def test_run_mode_matches_n8n_set_run_mode(live_configs, tmp_path, profile_id, trigger, override):
    profile = next(p for p in LIVE["profiles"] if p["profile_id"] == profile_id)
    js = run_node(tmp_path, "Set Run Mode", now="2026-09-28", input_items=[{
        "config": {**profile, "llm_models": LIVE["llm_models"]}, "triggerNode": trigger,
        "trend_enabled_override": True, **override}])[0]
    py = load_config(
        trigger_name=trigger, now=NOW,
        current_date=override.get("current_date_override"),
        lookback_days=override.get("lookback_days_override"),
        test_mode=override.get("test_mode_override"),
        llm_cache_bypass=override.get("llm_cache_bypass"))
    assert py["profile_id"] == profile_id
    pairs = {
        "mode": "mode", "lookbackDays": "lookback_days", "periodLabel": "period_label",
        "periodTitle": "period_title", "periodRange": "period_range", "digestLabel": "digest_label",
        "selectionRangeLabel": "selection_range_label", "maxSelection": "max_selection",
        "reportSelectionRangeLabel": "report_selection_range_label",
        "reportMaxSelection": "report_max_selection", "currentDate": "current_date",
        "isTestMode": "test_mode", "llm_cache_bypass": "llm_cache_bypass",
        "trend_enabled": "trend_enabled", "profileSlug": "profile_slug", "topicName": "topic_name",
        "topicFocus": "topic_focus", "assistantSignature": "assistant_signature",
    }
    for js_key, py_key in pairs.items():
        expected = js[js_key]
        if js_key == "lookbackDays":
            expected = js_number(expected)
        assert py[py_key] == expected, js_key


# --- LLM client ---------------------------------------------------------------

def _spy_client(captured: dict):
    def create(**kw):
        captured.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_llm_call_caps_max_tokens(monkeypatch):
    monkeypatch.delenv("MIRA_LLM_MAX_TOKENS", raising=False)
    captured = {}
    llm_call(_spy_client(captured), "m", "s", "u")
    assert captured["max_tokens"] == 16000
    monkeypatch.setenv("MIRA_LLM_MAX_TOKENS", "4096")
    llm_call(_spy_client(captured), "m", "s", "u")
    assert captured["max_tokens"] == 4096
    llm_call(_spy_client(captured), "m", "s", "u", max_tokens=123)
    assert captured["max_tokens"] == 123
    monkeypatch.setenv("MIRA_LLM_MAX_TOKENS", "0")
    captured.clear()
    llm_call(_spy_client(captured), "m", "s", "u", reasoning_effort="high")
    assert "max_tokens" not in captured and captured["reasoning_effort"] == "high"


def test_make_llm_client_endpoint_and_key(monkeypatch):
    seen = {}
    monkeypatch.setattr(cfgmod, "OpenAI", lambda **kw: seen.update(kw) or "client")
    monkeypatch.delenv("MIRA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("MIRA_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    make_llm_client()
    assert seen == {"api_key": "or-key", "base_url": "https://openrouter.ai/api/v1"}
    monkeypatch.setenv("MIRA_LLM_BASE_URL", "http://llm.internal:8000/v1")
    monkeypatch.setenv("MIRA_LLM_API_KEY", "corp-key")
    make_llm_client()
    assert seen == {"api_key": "corp-key", "base_url": "http://llm.internal:8000/v1"}
    monkeypatch.delenv("MIRA_LLM_API_KEY")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError, match="MIRA_LLM_API_KEY"):
        make_llm_client()

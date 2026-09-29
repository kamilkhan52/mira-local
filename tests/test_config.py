# tests/test_config.py
from _paths import repoint
import json
import re

import pytest
from pathlib import Path
from mira.config import (OPTIONAL_STAGE_KEYS, load_config, apply_template,
                         parse_json_response, llm_call)
from json import JSONDecodeError

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"

ALL_STAGE_MODELS = {
    "affiliation": "model", "classification": "model", "selection": "model",
    "analysis": "model", "trend": "model", "report": "model",
    "media_selection": "model", "media_summary": "model",
}


EXPECTED_OPTICAL_THEMES = [
    "Scale-Up Architectures",
    "Memory Disaggregation & Pooling",
    "Modulator Technologies",
    "Light Sources & Lasers",
    "Advanced Packaging",
    "Optical Switching",
    "DSP & Retimers",
    "Photonic Integrated Circuits",
    "Optical Network Architecture",
]


def test_optical_themes_parity_across_config_copies():
    """Dual-copy invariant: the optical-interconnects `themes` list must be
    identical (same values, same order) in the live profile config and the
    standalone n8n copy."""
    live = json.loads((CONFIGS_DIR / "memory-innovation-profile.json").read_text())
    optical_profile = next(
        p for p in live["profiles"] if p["profile_id"] == "optical-interconnects"
    )
    standalone = json.loads((CONFIGS_DIR / "optical-workflow-prompts.json").read_text())

    assert optical_profile["themes"] == standalone["themes"]
    assert optical_profile["themes"] == EXPECTED_OPTICAL_THEMES


def test_load_config_resolves_weekly_dates(tmp_path, monkeypatch):
    # Create a minimal profile config in tmp_path
    config_data = {
        "llm_models": dict(ALL_STAGE_MODELS),
        "profiles": [{
            "profile_id": "test-profile",
            "topic": {"name": "Test", "focus": "testing", "assistant_signature": "TEST"},
            "arxiv": {"categories": ["cs.AR"], "max_results": 100},
            "thresholds": {"relevance_score_min": 5, "credibility_tier_min": 5},
            "prompts": {"affiliation": {"user": "", "system": ""},
                        "classification": {"user": "", "system": ""},
                        "selection": {"user": "", "system": ""},
                        "analysis": {"user": "", "system": ""},
                        "report": {"user": "", "system": ""}},
            "media": {},
            "email": {"weekly": {"header_label": "Test", "colors": {"primary": "#000", "accent": "#000"}},
                      "daily": {"header_label": "Test", "colors": {"primary": "#000", "accent": "#000"}}},
            "modes": {"weekly": {"lookback_days": 7, "selection_max": 10, "trend_enabled": False,
                                  "selection_range_label": "5-10"},
                      "daily": {"lookback_days": 1, "selection_max": 5, "trend_enabled": False,
                                "selection_range_label": "3-5"}},
        }]
    }
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "test-profile.json").write_text(json.dumps(config_data))

    repoint(monkeypatch, "mira.config", tmp_path)
    monkeypatch.setenv("RECIPIENT_EMAIL", "test@example.com")

    config = load_config("test-profile", "weekly")
    assert config["mode"] == "weekly"
    assert len(config["start_date"]) == 8
    assert len(config["end_date"]) == 8
    assert config["start_date"] < config["end_date"]
    assert config["mode_cfg"]["lookback_days"] == 7


def test_load_config_surfaces_themes(tmp_path, monkeypatch):
    config_data = {
        "llm_models": dict(ALL_STAGE_MODELS),
        "profiles": [{
            "profile_id": "themed-profile",
            "topic": {"name": "Test", "focus": "testing", "assistant_signature": "TEST"},
            "arxiv": {"categories": ["cs.AR"], "max_results": 100},
            "thresholds": {"relevance_score_min": 5, "credibility_tier_min": 5},
            "themes": ["Alpha", "Beta"],
            "prompts": {"affiliation": {"user": "", "system": ""},
                        "classification": {"user": "", "system": ""},
                        "selection": {"user": "", "system": ""},
                        "analysis": {"user": "", "system": ""},
                        "report": {"user": "", "system": ""}},
            "media": {},
            "email": {"weekly": {"header_label": "Test", "colors": {"primary": "#000", "accent": "#000"}},
                      "daily": {"header_label": "Test", "colors": {"primary": "#000", "accent": "#000"}}},
            "modes": {"weekly": {"lookback_days": 7, "selection_max": 10, "trend_enabled": False,
                                  "selection_range_label": "5-10"},
                      "daily": {"lookback_days": 1, "selection_max": 5, "trend_enabled": False,
                                "selection_range_label": "3-5"}},
        }]
    }
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "themed-profile.json").write_text(json.dumps(config_data))

    repoint(monkeypatch, "mira.config", tmp_path)
    monkeypatch.setenv("RECIPIENT_EMAIL", "test@example.com")

    config = load_config("themed-profile", "weekly")
    assert config["themes"] == ["Alpha", "Beta"]


def test_load_config_defaults_themes_to_empty_when_absent(tmp_path, monkeypatch):
    config_data = {
        "llm_models": dict(ALL_STAGE_MODELS),
        "profiles": [{
            "profile_id": "themeless-profile",
            "topic": {"name": "Test", "focus": "testing", "assistant_signature": "TEST"},
            "arxiv": {"categories": ["cs.AR"], "max_results": 100},
            "thresholds": {"relevance_score_min": 5, "credibility_tier_min": 5},
            "prompts": {"affiliation": {"user": "", "system": ""},
                        "classification": {"user": "", "system": ""},
                        "selection": {"user": "", "system": ""},
                        "analysis": {"user": "", "system": ""},
                        "report": {"user": "", "system": ""}},
            "media": {},
            "email": {"weekly": {"header_label": "Test", "colors": {"primary": "#000", "accent": "#000"}},
                      "daily": {"header_label": "Test", "colors": {"primary": "#000", "accent": "#000"}}},
            "modes": {"weekly": {"lookback_days": 7, "selection_max": 10, "trend_enabled": False,
                                  "selection_range_label": "5-10"},
                      "daily": {"lookback_days": 1, "selection_max": 5, "trend_enabled": False,
                                "selection_range_label": "3-5"}},
        }]
    }
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "themeless-profile.json").write_text(json.dumps(config_data))

    repoint(monkeypatch, "mira.config", tmp_path)
    monkeypatch.setenv("RECIPIENT_EMAIL", "test@example.com")

    config = load_config("themeless-profile", "weekly")
    assert config["themes"] == []


def test_load_config_unknown_profile_raises(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    repoint(monkeypatch, "mira.config", tmp_path)
    with pytest.raises(ValueError, match="not found"):
        load_config("nonexistent-profile", "weekly")


def test_apply_template_substitutes_known_vars():
    result = apply_template("Hello {{name}}, topic is {{topic}}", {"name": "Alice", "topic": "HBM"})
    assert result == "Hello Alice, topic is HBM"


def test_apply_template_leaves_unknown_vars_intact():
    result = apply_template("{{known}} and {{unknown}}", {"known": "X"})
    assert result == "X and {{unknown}}"


def test_parse_json_response_strips_markdown_fences():
    raw = "```json\n{\"key\": \"value\"}\n```"
    assert parse_json_response(raw) == {"key": "value"}


def test_parse_json_response_plain_json():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_raises_on_invalid():
    with pytest.raises(JSONDecodeError):
        parse_json_response("not json at all")


def test_llm_call_retries_on_exception(monkeypatch):
    from unittest.mock import MagicMock
    monkeypatch.setattr("mira.config.time.sleep", lambda _: None)  # skip real sleep
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        RuntimeError("transient"),
        MagicMock(choices=[MagicMock(message=MagicMock(content="success"))]),
    ]
    result = llm_call(client, "model", "sys", "user")
    assert result == "success"
    assert client.chat.completions.create.call_count == 2


def test_llm_call_raises_after_all_retries(monkeypatch):
    from unittest.mock import MagicMock
    monkeypatch.setattr("mira.config.time.sleep", lambda _: None)
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("always fails")
    with pytest.raises(RuntimeError, match="LLM call failed after 3 attempts"):
        llm_call(client, "model", "sys", "user", retries=3)


# --- per-stage model resolution -------------------------------------------

from mira.config import (
    STAGE_KEYS,
    model_for,
    record_parse_failure,
    parse_failure_report,
    reset_parse_failures,
    format_parse_failures,
)


def test_stage_keys_are_the_eight_canonical_stages():
    assert STAGE_KEYS == (
        "affiliation", "classification", "selection", "analysis",
        "trend", "report", "media_selection", "media_summary",
    )


def test_model_for_returns_configured_model():
    config = {"llm_models": {"analysis": "anthropic/claude-opus-4.8"}}
    assert model_for(config, "analysis") == "anthropic/claude-opus-4.8"


def test_model_for_raises_on_missing_stage():
    config = {"llm_models": {"affiliation": "google/gemini-3.5-flash-lite"}}
    with pytest.raises(KeyError) as exc:
        model_for(config, "report")
    assert "report" in str(exc.value)


def test_model_for_error_names_configured_stages():
    config = {"llm_models": {"affiliation": "m", "classification": "m"}}
    with pytest.raises(KeyError) as exc:
        model_for(config, "analysis")
    message = str(exc.value)
    assert "affiliation" in message and "classification" in message


def test_model_for_does_not_fall_back_to_another_stage():
    """Regression: the old code resolved a missing key to the first dict value."""
    config = {"llm_models": {"affiliation": "google/gemini-3.5-flash-lite"}}
    with pytest.raises(KeyError):
        model_for(config, "analysis")


def test_parse_failure_counter_records_per_stage():
    reset_parse_failures()
    record_parse_failure("classification")
    record_parse_failure("classification")
    record_parse_failure("affiliation")
    assert parse_failure_report() == {"classification": 2, "affiliation": 1}


def test_parse_failure_report_is_a_copy():
    reset_parse_failures()
    record_parse_failure("report")
    snapshot = parse_failure_report()
    record_parse_failure("report")
    assert snapshot == {"report": 1}


def test_reset_parse_failures_clears_all():
    record_parse_failure("trend")
    reset_parse_failures()
    assert parse_failure_report() == {}


def test_format_parse_failures_empty_is_blank():
    assert format_parse_failures({}) == ""


def test_format_parse_failures_lists_stages_sorted():
    out = format_parse_failures({"classification": 3, "affiliation": 1})
    assert "affiliation=1" in out
    assert "classification=3" in out
    assert out.index("affiliation") < out.index("classification")


def test_format_parse_failures_warns():
    assert "WARNING" in format_parse_failures({"classification": 12})


# --- live config stage coverage -------------------------------------------

def test_live_config_defines_every_stage_model():
    """Every pipeline stage is configured, plus only known optional keys.

    `llm_models` is also read by standalone tools (exhaustive research), so it
    may carry keys outside STAGE_KEYS -- but only ones the code recognises: a
    typo'd key would otherwise silently resolve to a hardcoded fallback."""
    raw = json.loads((CONFIGS_DIR / "memory-innovation-profile.json").read_text())
    assert set(STAGE_KEYS) <= set(raw["llm_models"])
    assert set(raw["llm_models"]) - set(STAGE_KEYS) <= set(OPTIONAL_STAGE_KEYS)
    assert raw["llm_models"]["exhaustive_map"] == "google/gemini-2.5-flash"
    assert raw["llm_models"]["exhaustive_synthesis"] == (
        "anthropic/claude-sonnet-5"
    )


def test_live_config_uses_dotted_anthropic_slugs():
    """OpenRouter's catalog uses dots (claude-opus-4.8), not hyphens."""
    raw = json.loads((CONFIGS_DIR / "memory-innovation-profile.json").read_text())
    for stage, model in raw["llm_models"].items():
        if model.startswith("anthropic/"):
            assert "-4-" not in model, f"{stage} uses hyphenated slug: {model}"


def test_load_config_rejects_incomplete_llm_models(tmp_path, monkeypatch):
    config_data = {
        "llm_models": {"affiliation": "m", "classification": "m"},
        "profiles": [{
            "profile_id": "test-profile",
            "topic": {"name": "T", "focus": "t", "assistant_signature": "T"},
            "arxiv": {"categories": ["cs.AR"], "max_results": 100},
            "thresholds": {"relevance_score_min": 5, "credibility_tier_min": 5},
            "prompts": {},
            "email": {"weekly": {}},
            "modes": {"weekly": {"lookback_days": 7}},
        }],
    }
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "p.json").write_text(json.dumps(config_data))
    repoint(monkeypatch, "mira.config", tmp_path)
    monkeypatch.setenv("RECIPIENT_EMAIL", "a@b.c")

    with pytest.raises(ValueError) as exc:
        load_config("test-profile", "weekly")
    message = str(exc.value)
    assert "analysis" in message and "report" in message


# --- trigger -> profile routing -------------------------------------------
#
# `Load Topic Config` in the n8n workflow picks a profile by substring-matching
# the trigger node's name against each profile's `trigger_match`, and used to
# fall back to `default_profile_id` whenever nothing matched. When the
# mcbu-memory and embedded-intelligence profiles went missing from the merged
# config, the AEBU and MCBU triggers silently resolved to memory-innovation and
# shipped digests labeled "Memory Technology Research". These tests pin the
# routing table so that regression is caught here instead of in an inbox.

EXPECTED_TRIGGER_ROUTING = {
    "MCBU Weekly Trigger": "mcbu-memory",
    "AEBU Weekly Trigger": "embedded-intelligence",
    "CXL Weekly Trigger": "cxl-research",
    "CXL Monthly Trigger": "cxl-research",
    "Storage Weekly Trigger": "storage-innovation",
    "Optical Weekly Trigger": "optical-interconnects",
}



def _live_config():
    return json.loads((CONFIGS_DIR / "memory-innovation-profile.json").read_text())


def _resolve_profile(config, trigger_name):
    """Mirror of the profile selection in the `Load Topic Config` node."""
    lowered = trigger_name.lower()
    for profile in config["profiles"]:
        if any(m.lower() in lowered for m in profile.get("trigger_match", [])):
            return profile["profile_id"]
    return None


@pytest.mark.parametrize("trigger,expected", sorted(EXPECTED_TRIGGER_ROUTING.items()))
def test_trigger_routes_to_its_own_profile(trigger, expected):
    resolved = _resolve_profile(_live_config(), trigger)
    assert resolved == expected, (
        f"{trigger!r} resolved to {resolved!r}; it would fall back to the default "
        f"profile and produce a report labeled for the wrong topic"
    )


def test_default_trigger_match_is_exact_not_substring():
    """The fallback allowlist is compared exactly. If it were a substring check,
    'MCBU Weekly Trigger' would be absorbed by the allow-listed 'Weekly Trigger'
    and the guard would wave the original bug straight through."""
    allowed = _live_config()["default_trigger_match"]
    for trigger in EXPECTED_TRIGGER_ROUTING:
        assert trigger not in allowed
    absorbed = [
        trigger
        for trigger in EXPECTED_TRIGGER_ROUTING
        if any(a.lower() in trigger.lower() for a in allowed)
    ]
    assert absorbed, (
        f"no trigger in {sorted(EXPECTED_TRIGGER_ROUTING)} contains an entry from "
        f"{allowed}, so this test no longer proves exact matching is required"
    )


def test_generic_triggers_are_allow_listed_for_the_default_profile():
    """Daily/Weekly/Backfill legitimately have no profile of their own, so they
    must stay on the allowlist or the guard will break the memory digest."""
    config = _live_config()
    for trigger in config["default_trigger_match"]:
        assert _resolve_profile(config, trigger) is None, (
            f"{trigger!r} now matches a profile; drop it from default_trigger_match"
        )
    assert config["default_profile_id"] == "memory-innovation"




def _workflow_trigger_names(workflow):
    """Nodes wired into `Capture Trigger Name` -- exactly the trigger names that
    reach the profile guard in `Load Topic Config`."""
    return sorted(
        source
        for source, conn in workflow["connections"].items()
        for outputs in conn.get("main", [])
        for target in outputs or []
        if target["node"] == "Capture Trigger Name"
    )


def _guarded_workflows():
    """Every tracked workflow that routes triggers through the profile guard.
    Globbed rather than named: more than one export of this workflow lives in
    `workflows/`, and the one n8n actually runs has changed before."""
    found = []
    for path in sorted(WORKFLOWS_DIR.glob("*.json")):
        workflow = json.loads(path.read_text())
        if any(n["name"] == "Capture Trigger Name" for n in workflow.get("nodes", [])):
            found.append(pytest.param(workflow, id=path.stem[:60]))
    return found


@pytest.mark.parametrize("workflow", _guarded_workflows())
def test_every_wired_trigger_resolves_without_hitting_the_guard(workflow):
    """The guard throws on an unmatched, un-allow-listed trigger, so a trigger
    wired into a workflow with no profile and no allowlist entry no longer sends
    the wrong digest -- it kills that run outright."""
    config = _live_config()
    allowed = {t.lower().strip() for t in config["default_trigger_match"]}
    triggers = _workflow_trigger_names(workflow)
    assert triggers, "no triggers feed 'Capture Trigger Name'; selector is stale"

    unresolved = [
        trigger
        for trigger in triggers
        if _resolve_profile(config, trigger) is None
        and trigger.lower().strip() not in allowed
    ]
    assert not unresolved, (
        f"{unresolved} match no profile and are not in default_trigger_match, so "
        f"`Load Topic Config` will throw on every one of those runs"
    )


def test_no_python_source_pins_a_hyphenated_anthropic_slug():
    """OpenRouter serves dots (claude-sonnet-4.6); a dash is a silent typo.

    Only the model *version* separator matters: `claude-sonnet-4-6` is wrong,
    while `claude-haiku-4-5` is the slug OpenRouter actually publishes, so the
    check targets the sonnet/opus families that use dotted versions."""
    root = Path(__file__).parent.parent
    offenders = []
    for source in sorted(root.rglob("*.py")):
        if "node_modules" in source.parts or ".git" in source.parts:
            continue
        for number, line in enumerate(
            source.read_text().splitlines(), 1
        ):
            if re.search(r"anthropic/claude-(sonnet|opus)-\d+-\d+", line):
                offenders.append(f"{source.relative_to(root)}:{number}")
    assert offenders == []

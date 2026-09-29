import pytest

import mira.discovery.feasibility as feas
from mira.discovery.feasibility import (
    TIER_MULTIPLIERS,
    build_feasibility_prompt,
    judge_feasibility,
    parse_tier,
)
from mira.hypothesis.gaps import GapCandidate


def _cand() -> GapCandidate:
    return GapCandidate("CXL", "PIM", ["HBM4"], ["Samsung"], ["Kim"], 1.0, 10.0)


def test_tier_multipliers_cover_all_tiers():
    assert set(TIER_MULTIPLIERS) == {"T1", "T2", "T3", "T4", "unrated"}
    assert TIER_MULTIPLIERS["T1"] > TIER_MULTIPLIERS["T4"]
    assert TIER_MULTIPLIERS["unrated"] == 1.0  # neutral (spec §3 Stage 4)


def test_parse_tier_happy_path():
    assert parse_tier("TIER: T2\nNeeds off-the-shelf CXL hardware.") == (
        "T2", "Needs off-the-shelf CXL hardware.")


def test_parse_tier_tolerates_case_and_whitespace():
    assert parse_tier("  tier: t4 \n Requires a 3nm fab. ") == ("T4", "Requires a 3nm fab.")


@pytest.mark.parametrize("garbage", ["no verdict here", "TIER: T9\nnope", ""])
def test_parse_tier_falls_back_to_unrated(garbage):
    tier, reason = parse_tier(garbage)
    assert tier == "unrated"
    assert reason == garbage.strip()


def test_prompt_contains_gap_facts():
    prompt = build_feasibility_prompt(_cand())
    assert "CXL" in prompt and "PIM" in prompt and "Samsung" in prompt


def test_judge_feasibility_parses_llm_reply(monkeypatch):
    monkeypatch.setattr(feas, "llm_call",
                        lambda *a, **k: "TIER: T1\nPure simulation study.")
    assert judge_feasibility(object(), _cand()) == ("T1", "Pure simulation study.")


def test_judge_feasibility_pins_temperature_zero(monkeypatch):
    captured = {}

    def spy(*a, **k):
        captured.update(k)
        return "TIER: T1\nSimulation."
    monkeypatch.setattr(feas, "llm_call", spy)
    judge_feasibility(object(), _cand())
    assert captured.get("temperature") == 0.0


def test_judge_feasibility_degrades_on_llm_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("LLM call failed after 3 retries")
    monkeypatch.setattr(feas, "llm_call", boom)
    tier, reason = judge_feasibility(object(), _cand())
    assert tier == "unrated"
    assert "judge unavailable" in reason

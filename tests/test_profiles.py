"""The profile abstraction: every profile/mode in the config is runnable, and
schedules / realtime settings only reference real profiles."""
import json
from pathlib import Path

import pytest

from mira.config import load_config, resolve_profile

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "configs" / "memory-innovation-profile.json").read_text())
PAIRS = [(p["profile_id"], m) for p in CONFIG["profiles"] for m in p["modes"]]


@pytest.mark.parametrize("profile,mode", PAIRS)
def test_every_profile_mode_loads(profile, mode):
    c = load_config(profile, mode, current_date="2026-09-26")
    assert c["profile_id"] == profile and c["mode"] == mode
    assert c["arxiv"]["categories"] and c["prompts"]["classification"]["user"]
    assert c["start_date_iso"] <= c["end_date_iso"] == "2026-09-26"


def test_every_profile_mode_has_a_schedule_entry():
    sched = json.loads((ROOT / "configs" / "schedules.json").read_text())
    scheduled = {(d["parameters"]["profile"], d["parameters"]["mode"])
                 for d in sched["deployments"] if d["flow"] == "digest"}
    assert scheduled == set(PAIRS)


def test_realtime_config_lists_exactly_the_profiles():
    rt = json.loads((ROOT / "configs" / "realtime.json").read_text())
    assert set(rt["subscribers"]) == {p["profile_id"] for p in CONFIG["profiles"]}


@pytest.mark.parametrize("trigger,expected", [
    ("CXL Monthly Trigger", ("cxl-research", "monthly")),
    ("Memory Daily Trigger", ("memory-innovation", "daily")),
    ("Storage Weekly Trigger", ("storage-innovation", "weekly")),
    ("Optical Weekly Trigger", ("optical-interconnects", "weekly")),
    ("MCBU Weekly Trigger", ("mcbu-memory", "weekly")),
    ("AEBU Weekly Trigger", ("embedded-intelligence", "weekly")),
])
def test_n8n_trigger_names_route_to_profiles(trigger, expected):
    assert resolve_profile(trigger) == expected

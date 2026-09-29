import json
from datetime import date

import pytest

from mira.discovery.ledger import SEEN_DEMOTION, Ledger, pair_key, parse_pair
from mira.hypothesis.gaps import GapCandidate


def _cand(a: str, c: str) -> GapCandidate:
    return GapCandidate(a, c, [], [], [], 0.0, 1.0)


def test_pair_key_is_order_insensitive():
    assert pair_key("PIM", "CXL") == pair_key("CXL", "PIM") == "CXL || PIM"


def test_parse_pair_single_bar():
    assert parse_pair("CXL|PIM") == ("CXL", "PIM")
    assert parse_pair(" CXL | PIM ") == ("CXL", "PIM")


def test_parse_pair_accepts_displayed_ledger_key_format():
    # Ledger keys print as "A || B"; pasting one back must round-trip.
    assert parse_pair("CXL || PIM") == ("CXL", "PIM")
    assert parse_pair(pair_key("PIM", "CXL")) == ("CXL", "PIM")


def test_parse_pair_without_separator_is_none():
    assert parse_pair("no separator here") is None
    assert parse_pair("") is None


def test_load_missing_file_gives_empty_ledger(tmp_path):
    led = Ledger.load(tmp_path / "ledger.json")
    assert led.entries == {} and led.warning is None


def test_upsert_new_then_repeat(tmp_path):
    led = Ledger.load(tmp_path / "ledger.json")
    led.upsert("CXL", "PIM", 0.8, date(2026, 7, 13))
    e = led.entry("CXL", "PIM")
    assert e["status"] == "new" and e["times_recommended"] == 1
    assert e["first_recommended"] == e["last_recommended"] == "2026-07-13"
    led.upsert("PIM", "CXL", 0.9, date(2026, 7, 20))  # reversed order: same entry
    e = led.entry("CXL", "PIM")
    assert e["times_recommended"] == 2
    assert e["best_score"] == 0.9
    assert e["first_recommended"] == "2026-07-13" and e["last_recommended"] == "2026-07-20"


def test_save_and_reload_roundtrip(tmp_path):
    path = tmp_path / "ledger.json"
    led = Ledger.load(path)
    led.upsert("CXL", "PIM", 0.8, date(2026, 7, 13))
    led.save()
    assert Ledger.load(path).entry("CXL", "PIM")["best_score"] == 0.8


def test_save_is_atomic_no_tmp_file_left_behind(tmp_path):
    path = tmp_path / "ledger.json"
    led = Ledger.load(path)
    led.upsert("CXL", "PIM", 0.8, date(2026, 7, 13))
    led.save()
    assert path.exists()
    assert json.loads(path.read_text())["CXL || PIM"]["best_score"] == 0.8
    assert not path.with_suffix(".tmp").exists()


def test_multiplier_demotes_seen_new_pairs_only(tmp_path):
    led = Ledger.load(tmp_path / "ledger.json")
    led.upsert("CXL", "PIM", 0.8, date(2026, 7, 13))
    assert led.multiplier("CXL", "PIM") == SEEN_DEMOTION
    assert led.multiplier("CXL", "DDR6") == 1.0  # unseen
    led.set_status(pair_key("CXL", "PIM"), "pursuing")
    assert led.multiplier("CXL", "PIM") == 1.0   # non-'new' statuses handled by partition


def test_set_status_validates(tmp_path):
    led = Ledger.load(tmp_path / "ledger.json")
    led.upsert("CXL", "PIM", 0.8, date(2026, 7, 13))
    assert led.set_status(pair_key("CXL", "PIM"), "rejected") is True
    assert led.entry("CXL", "PIM")["status"] == "rejected"
    assert led.set_status("nope || nada", "rejected") is False
    with pytest.raises(ValueError):
        led.set_status(pair_key("CXL", "PIM"), "bogus")


def test_partition_by_status(tmp_path):
    led = Ledger.load(tmp_path / "ledger.json")
    for pair, status in [(("A", "B"), "rejected"), (("C", "D"), "pursuing"),
                         (("E", "F"), "new")]:
        led.upsert(*pair, 0.5, date(2026, 7, 1))
        led.set_status(pair_key(*pair), status)
    cands = [_cand("A", "B"), _cand("C", "D"), _cand("E", "F"), _cand("G", "H")]
    eligible, pursuing, suppressed = led.partition(cands)
    assert [(c.topic_a, c.topic_c) for c in eligible] == [("E", "F"), ("G", "H")]
    assert [(c.topic_a, c.topic_c) for c in pursuing] == [("C", "D")]
    assert suppressed == [("A || B", "rejected")]


def test_corrupt_ledger_backed_up_not_overwritten(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text("{not json")
    led = Ledger.load(path)
    assert led.entries == {}
    assert "corrupt" in led.warning
    backups = list(tmp_path.glob("ledger.bak-*"))
    assert len(backups) == 1 and backups[0].read_text() == "{not json"


def test_summary_counts_statuses(tmp_path):
    led = Ledger.load(tmp_path / "ledger.json")
    led.upsert("A", "B", 0.5, date(2026, 7, 1))
    led.upsert("C", "D", 0.5, date(2026, 7, 1))
    led.set_status(pair_key("C", "D"), "pursuing")
    assert "2 tracked" in led.summary()
    assert "1 new" in led.summary() and "1 pursuing" in led.summary()


def test_malformed_entry_treated_as_corrupt(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"CXL || PIM": {"best_score": 0.8}}))  # missing keys
    led = Ledger.load(path)
    assert led.entries == {}
    assert "corrupt" in led.warning
    assert len(list(tmp_path.glob("ledger.bak-*"))) == 1


def test_invalid_status_value_treated_as_corrupt(tmp_path):
    path = tmp_path / "ledger.json"
    good = {"first_recommended": "2026-07-01", "last_recommended": "2026-07-01",
            "best_score": 0.5, "times_recommended": 1, "status": "bogus"}
    path.write_text(json.dumps({"A || B": good}))
    led = Ledger.load(path)
    assert led.entries == {} and "corrupt" in led.warning

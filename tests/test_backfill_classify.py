import backfill_classify as bc


# ─── is_gap_id ────────────────────────────────────────────────────────────────

def test_is_gap_id_before_prefix():
    assert bc.is_gap_id("2603.12345", "2604") is True
    assert bc.is_gap_id("2512.00001", "2604") is True
    assert bc.is_gap_id("2511.99999", "2604") is True


def test_is_gap_id_at_or_after_prefix():
    assert bc.is_gap_id("2604.00001", "2604") is False
    assert bc.is_gap_id("2605.12345", "2604") is False


def test_is_gap_id_rejects_garbage():
    assert bc.is_gap_id("not-an-id", "2604") is False
    assert bc.is_gap_id("", "2604") is False


# ─── compare_topics ───────────────────────────────────────────────────────────

def test_compare_topics_exact_match():
    out = bc.compare_topics({"HBM", "CXL"}, {"HBM", "CXL", "DRAM"})
    assert out["matched"] == ["CXL", "HBM"]
    assert out["drift"] == []
    assert out["brand_new"] == []


def test_compare_topics_detects_drift():
    # produced differs only by case / spacing / punctuation from an existing node
    out = bc.compare_topics({"High-Bandwidth Memory", "hbm"}, {"HBM", "Highbandwidth memory"})
    drift_sources = {d[0] for d in out["drift"]}
    assert "hbm" in drift_sources
    assert "High-Bandwidth Memory" in drift_sources
    assert out["matched"] == []
    assert out["brand_new"] == []


def test_compare_topics_brand_new():
    out = bc.compare_topics({"Quantum Memory"}, {"HBM", "CXL"})
    assert out["brand_new"] == ["Quantum Memory"]
    assert out["matched"] == []
    assert out["drift"] == []


def test_compare_topics_mixed():
    produced = {"HBM", "hbm ", "Photonic Memory"}
    existing = {"HBM", "DRAM"}
    out = bc.compare_topics(produced, existing)
    assert out["matched"] == ["HBM"]
    assert [d[0] for d in out["drift"]] == ["hbm "]
    assert out["brand_new"] == ["Photonic Memory"]


# ─── merge_flat_cache ─────────────────────────────────────────────────────────

def test_merge_flat_cache_adds_missing():
    existing = {"2605.0001": {"arxiv_id": "2605.0001", "primary_topic": "HBM"}}
    new = {"2603.0002": {"arxiv_id": "2603.0002", "primary_topic": "CXL"}}
    merged = bc.merge_flat_cache(existing, new)
    assert merged.pop("__added__") == 1
    assert set(merged) == {"2605.0001", "2603.0002"}


def test_merge_flat_cache_never_overwrites():
    existing = {"2603.0002": {"arxiv_id": "2603.0002", "primary_topic": "ORIGINAL"}}
    new = {"2603.0002": {"arxiv_id": "2603.0002", "primary_topic": "OVERWRITE"}}
    merged = bc.merge_flat_cache(existing, new)
    assert merged.pop("__added__") == 0
    assert merged["2603.0002"]["primary_topic"] == "ORIGINAL"


def test_merge_flat_cache_does_not_mutate_input():
    existing = {"a": {"x": 1}}
    new = {"b": {"y": 2}}
    bc.merge_flat_cache(existing, new)
    assert "b" not in existing
    assert "__added__" not in existing

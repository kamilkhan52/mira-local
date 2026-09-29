from datetime import date

import requests

from mira.hypothesis.gaps import GapCandidate
from mira.hypothesis.novelty import NoveltyChecker, classify, unverified_note


def _cand(topic_a, topic_c, novelty_hits):
    return GapCandidate(
        topic_a=topic_a, topic_c=topic_c, shared_topics=[],
        shared_institutions=[], shared_authors=[], velocity=0.0,
        structural_score=0.0, novelty_hits=novelty_hits)


def test_classify_thresholds():
    assert classify(None) == "unverified"
    assert classify(0) == "open gap"
    assert classify(2) == "open gap"
    assert classify(3) == "sparsely explored"
    assert classify(19) == "sparsely explored"
    assert classify(20) == "dropped"


def test_check_counts_recent_unique_titles(tmp_path):
    calls = []
    def fake_fetch(query):
        calls.append(query)
        return [
            {"title": "Recent paper", "year": 2025},
            {"title": "Old paper", "year": 2019},      # outside 3-year window
            {"title": "Recent paper", "year": 2025},   # duplicate title
        ]
    checker = NoveltyChecker(tmp_path / "cache.json", fetch=fake_fetch,
                             today=date(2026, 7, 9))
    hits, titles = checker.check("CXL", "PIM")
    assert hits == 1
    assert titles == ["Recent paper"]
    # quoted variant + relaxed variant
    assert calls == ['"CXL" "PIM"', "CXL PIM"]


def test_check_caches_to_disk(tmp_path):
    calls = []
    def fake_fetch(query):
        calls.append(query)
        return [{"title": "T", "year": 2026}]
    path = tmp_path / "cache.json"
    NoveltyChecker(path, fetch=fake_fetch, today=date(2026, 7, 9)).check("A", "B")
    n_calls = len(calls)
    # Fresh checker, same cache file: no new fetches.
    hits, _ = NoveltyChecker(path, fetch=fake_fetch, today=date(2026, 7, 9)).check("A", "B")
    assert hits == 1
    assert len(calls) == n_calls


def test_api_failure_returns_unverified_and_is_not_cached(tmp_path):
    def broken_fetch(query):
        raise requests.ConnectionError("down")
    path = tmp_path / "cache.json"
    checker = NoveltyChecker(path, fetch=broken_fetch, today=date(2026, 7, 9))
    assert checker.check("A", "B") == (None, [])
    assert "A || B" not in checker.cache


def test_unverified_note_none_when_all_pairs_verified():
    cands = [_cand("CXL", "PIM", 0), _cand("HBM", "TSV", 5)]
    assert unverified_note(cands) is None


def test_unverified_note_names_the_failed_pair():
    cands = [_cand("CXL", "PIM", 3), _cand("Optics", "MRM", None)]
    note = unverified_note(cands)
    assert note is not None
    assert "1 pair" in note                      # singular
    assert "Optics × MRM" in note                # only the failed pair named
    assert "CXL × PIM" not in note               # verified pair not named


def test_unverified_note_pluralizes_multiple_failures():
    cands = [_cand("A", "B", None), _cand("C", "D", None)]
    note = unverified_note(cands)
    assert "2 pairs" in note
    assert "A × B" in note and "C × D" in note


def test_cutoff_year_boundary_is_inclusive(tmp_path):
    def fake_fetch(query):
        return [
            {"title": "Exactly at cutoff", "year": 2023},
            {"title": "Just outside", "year": 2022},
        ]
    checker = NoveltyChecker(tmp_path / "cache.json", fetch=fake_fetch,
                             today=date(2026, 7, 9))
    hits, titles = checker.check("A", "B")
    # Window is inclusive of the cutoff year (2026 - 3 = 2023): deliberately
    # conservative for novelty claims.
    assert hits == 1
    assert titles == ["Exactly at cutoff"]

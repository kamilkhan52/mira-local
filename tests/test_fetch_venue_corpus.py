import importlib
import sys
from types import SimpleNamespace

import pytest


class _July2026:
    @staticmethod
    def today():
        return SimpleNamespace(year=2026)


@pytest.fixture
def mod():
    sys.path.insert(0, "scripts")
    try:
        yield importlib.import_module("fetch_venue_corpus")
    finally:
        sys.path.remove("scripts")


def test_crossref_rows_shapes_items(mod, monkeypatch):
    calls = []

    def fake_get(url, params, *, pacing, tries=6):
        calls.append((url, dict(params)))
        return {"message": {
            "items": [
                {"DOI": "10.1109/IEDM50854.2024.10873587",
                 "title": ["Performance Optimization of  GaN Devices."]},
                {"DOI": "", "title": ["No DOI — skipped"]},
                {"DOI": "10.1109/x.2024.2", "title": []},  # no title — skipped
            ],
            "next-cursor": None,
        }}

    monkeypatch.setattr(mod, "_get", fake_get)
    monkeypatch.setattr(mod, "date", _July2026)
    config = {"venues": [
        {"name": "ISCA", "dblp_stream": "conf/isca"},  # no containers: skipped
        {"name": "IEDM", "crossref_containers": [
            "{year} IEEE International Electron Devices Meeting (IEDM)",
            "{year} International Electron Devices Meeting (IEDM)"]},
    ]}
    rows = mod.crossref_rows(config, since_year=2026)

    assert rows == [{
        "id": "crossref:10.1109/iedm50854.2024.10873587",
        "title": "Performance Optimization of GaN Devices",
        "abstract": "",
        "year": 2026,
        "venue": "IEDM",
        "url": "https://doi.org/10.1109/iedm50854.2024.10873587",
        "doi": "10.1109/iedm50854.2024.10873587",
    }]
    url, params = calls[0]
    assert url.startswith(mod.CROSSREF)
    # both year-formatted containers OR-joined, restricted to proceedings papers
    assert params["filter"] == (
        "container-title:2026 IEEE International Electron Devices Meeting (IEDM),"
        "container-title:2026 International Electron Devices Meeting (IEDM),"
        "type:proceedings-article")


def test_fetch_crossref_year_pages_with_cursor(mod, monkeypatch):
    pages = [
        {"message": {"items": [{"DOI": "10.1/a", "title": ["A"]}], "next-cursor": "c2"}},
        {"message": {"items": [{"DOI": "10.1/b", "title": ["B"]}], "next-cursor": "c3"}},
        {"message": {"items": [], "next-cursor": "c4"}},
    ]
    cursors = []

    def fake_get(url, params, *, pacing, tries=6):
        cursors.append(params["cursor"])
        return pages[len(cursors) - 1]

    monkeypatch.setattr(mod, "_get", fake_get)
    items = mod.fetch_crossref_year(["{year} IEDM"], 2024)
    assert [i["DOI"] for i in items] == ["10.1/a", "10.1/b"]
    assert cursors == ["*", "c2", "c3"]


def test_dedup_by_doi_keeps_first_occurrence(mod):
    dblp = {"id": "dblp:conf/vlsit/X24", "doi": "10.23919/x", "title": "X"}
    crossref = {"id": "crossref:10.23919/x", "doi": "10.23919/x", "title": "X"}
    no_doi_a = {"id": "dblp:conf/isca/A24", "doi": "", "title": "A"}
    no_doi_b = {"id": "dblp:conf/isca/B24", "doi": "", "title": "B"}
    # DOI-less rows are never merged; duplicate DOI keeps the first (DBLP) row
    assert mod.dedup_by_doi([dblp, no_doi_a, crossref, no_doi_b]) == [
        dblp, no_doi_a, no_doi_b]


def test_unresolved_venues_accepts_either_source(mod):
    config = {"venues": [
        {"name": "ISCA", "dblp_stream": "conf/isca"},
        {"name": "IEDM", "crossref_containers": ["{year} IEDM"]},
        {"name": "Broken"},
        {"name": "AlsoBroken", "dblp_stream": ""},
    ]}
    assert mod.unresolved_venues(config) == ["Broken", "AlsoBroken"]


def test_ordinal_covers_suffixes_and_teens(mod):
    cases = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th",
             11: "11th", 12: "12th", 13: "13th",
             21: "21st", 22: "22nd", 23: "23rd",
             73: "73rd", 74: "74th", 75: "75th",
             100: "100th", 111: "111th", 113: "113th"}
    assert {n: mod.ordinal(n) for n in cases} == cases


def test_fetch_crossref_year_fills_ordinal(mod, monkeypatch):
    captured = {}

    def fake_get(url, params, *, pacing, tries=6):
        captured["filter"] = params["filter"]
        return {"message": {"items": [], "next-cursor": None}}

    monkeypatch.setattr(mod, "_get", fake_get)
    mod.fetch_crossref_year(
        ["{year} IEEE {ordinal} Electronic Components and Technology Conference (ECTC)"],
        2024, ordinal_base=1950)
    assert captured["filter"] == (
        "container-title:2024 IEEE 74th Electronic Components and Technology "
        "Conference (ECTC),type:proceedings-article")


def test_fetch_crossref_year_ordinal_without_base_raises(mod, monkeypatch):
    monkeypatch.setattr(mod, "_get", lambda *a, **k: {
        "message": {"items": [], "next-cursor": None}})
    with pytest.raises(KeyError):
        mod.fetch_crossref_year(["{year} {ordinal} X"], 2024)  # no ordinal_base


def test_crossref_rows_passes_ordinal_base_per_venue(mod, monkeypatch):
    seen = []

    def fake_fetch(containers, year, ordinal_base=None):
        seen.append((year, ordinal_base))
        return []

    monkeypatch.setattr(mod, "fetch_crossref_year", fake_fetch)
    monkeypatch.setattr(mod, "date", _July2026)
    config = {"venues": [
        {"name": "ECTC", "crossref_containers": ["{year} IEEE {ordinal} ECTC"],
         "ordinal_base": 1950}]}
    mod.crossref_rows(config, since_year=2024)
    assert seen == [(2024, 1950), (2025, 1950), (2026, 1950)]


def test_novelty_cache_for_matches_graph_target_naming(mod):
    from mira.graph_target import resolve_target
    for name in ("memory", "optical"):
        target = resolve_target(name)
        assert mod.novelty_cache_for(target.venue_db) == target.novelty_cache


def test_main_builds_db_and_clears_novelty_cache(mod, monkeypatch, tmp_path):
    config_path = tmp_path / "venue-corpus.json"
    config_path.write_text('{"years_back": 3, "venues": [{"name": "V", '
                           '"dblp_stream": "conf/v"}]}')
    db_path = tmp_path / "venue_corpus.sqlite"
    cache_path = tmp_path / "venue_novelty.json"       # derived from db_path
    cache_path.write_text("{}")

    monkeypatch.setattr(mod, "dblp_rows", lambda config, since_year: [
        {"id": "dblp:conf/v/1", "title": "T", "abstract": "", "year": 2026,
         "venue": "V", "url": "", "doi": ""}])
    monkeypatch.setattr(mod, "crossref_rows", lambda config, since_year: [])
    monkeypatch.setattr(mod, "hydrate_abstracts", lambda rows: 0)
    monkeypatch.setattr(sys, "argv", ["fetch_venue_corpus.py",
                                      "--config", str(config_path),
                                      "--db", str(db_path)])
    assert mod.main() == 0
    assert db_path.exists()
    # per-pair novelty counts are only valid for the corpus they were
    # computed against — a rebuild must invalidate them
    assert not cache_path.exists()


def test_optical_config_is_valid_and_ectc_formats(mod):
    import json
    from pathlib import Path
    cfg = json.loads(
        Path("configs/venue-corpus-optical.json").read_text())
    assert cfg["years_back"] == 3
    names = {v["name"] for v in cfg["venues"]}
    assert names == {"OFC", "SC", "HOTI", "ECTC"}
    # every venue is resolvable (has a dblp_stream or crossref_containers)
    assert mod.unresolved_venues(cfg) == []
    ectc = next(v for v in cfg["venues"] if v["name"] == "ECTC")
    fmt = ectc["crossref_containers"][0].format(
        year=2024, ordinal=mod.ordinal(2024 - ectc["ordinal_base"]))
    assert fmt == ("2024 IEEE 74th Electronic Components and Technology "
                   "Conference (ECTC)")

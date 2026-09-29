import json
import os
import sqlite3
from datetime import date
from types import SimpleNamespace

import pytest
import requests

import hypothesize
import mira.venue_corpus as venue_corpus
from mira.venue_corpus import (
    VenueCorpusError,
    build_db,
    expand_phrases,
    extract_phrases,
    make_venue_fetch,
    parse_pair_query,
    read_built_at,
    reconstruct_abstract,
    work_to_row,
)


# --- term extraction -------------------------------------------------------

def test_extract_phrases_strips_prefix_and_splits():
    assert extract_phrases("AI/ML Memory: KV Cache Compression & Eviction") == [
        "KV Cache Compression", "Eviction"]


def test_extract_phrases_strips_trailing_parenthetical():
    assert extract_phrases(
        "AI/ML Memory: Performance Optimization (bandwidth/latency/throughput)"
    ) == ["Performance Optimization"]


def test_extract_phrases_drops_generic_single_word():
    assert extract_phrases("3D Stacked DRAM/Memory") == ["3D Stacked DRAM"]


def test_extract_phrases_keeps_generic_when_sole_phrase():
    assert extract_phrases("Memory") == ["Memory"]


def test_extract_phrases_degenerate_falls_back_to_whole_label():
    assert extract_phrases("AI/ML") == ["AI/ML"]


def test_extract_phrases_splits_acronym_tail():
    # Optical taxonomy labels carry an " - ACRONYM" tail the corpus never
    # writes inline ("Microring Modulators - MRM" appears as "microring
    # modulator"); keep both the spelled-out base and the acronym as phrases.
    assert extract_phrases("Microring Modulators - MRM") == [
        "Microring Modulators", "MRM"]


def test_extract_phrases_acronym_tail_preserves_internal_hyphen():
    # Only the spaced-dash tail is an acronym separator; the hyphen inside
    # "Mach-Zehnder" must survive.
    assert extract_phrases("Mach-Zehnder - MZM") == ["Mach-Zehnder", "MZM"]


def test_extract_phrases_keeps_mixed_case_acronym_tail():
    assert extract_phrases("Silicon Photonics - SiPh") == [
        "Silicon Photonics", "SiPh"]


def test_extract_phrases_does_not_split_multiword_dash_tail():
    # A spaced-dash with a multi-word tail is not the acronym convention
    # (affiliation strings, etc.) — leave the label intact.
    assert extract_phrases("Technion - Israel Institute of Technology") == [
        "Technion - Israel Institute of Technology"]


def test_expand_phrases_adds_domain_synonym_variants():
    expanded = [p.casefold() for p in
                expand_phrases(["KV Cache Compression", "Eviction"])]
    assert "kv cache compression" in expanded    # originals always kept
    assert "eviction" in expanded
    assert "kv cache quantization" in expanded   # compression -> quantization
    assert "offloading" in expanded              # eviction -> offloading


def test_expand_phrases_without_synonyms_is_identity():
    assert expand_phrases(["Memory Interface"]) == ["Memory Interface"]


# --- query parsing ---------------------------------------------------------

def test_parse_pair_query_quoted():
    assert parse_pair_query('"CXL" "Processing-in-Memory"') == (
        "CXL", "Processing-in-Memory")


def test_parse_pair_query_unquoted_returns_none():
    assert parse_pair_query("CXL Processing-in-Memory") is None


# --- OpenAlex row shaping --------------------------------------------------

def test_reconstruct_abstract_orders_tokens():
    inv = {"cache": [1], "KV": [0], "compression": [2, 4], "for": [3]}
    assert reconstruct_abstract(inv) == "KV cache compression for compression"


def test_reconstruct_abstract_empty():
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_work_to_row_shapes_fields():
    work = {"id": "https://openalex.org/W1", "title": "T",
            "publication_year": 2025,
            "abstract_inverted_index": {"HBM": [0]},
            "doi": "https://doi.org/10.1/x",
            "primary_location": {"landing_page_url": "https://x"}}
    assert work_to_row(work, "ISCA") == {
        "id": "https://openalex.org/W1", "title": "T", "abstract": "HBM",
        "year": 2025, "venue": "ISCA", "url": "https://doi.org/10.1/x"}


# --- DB build + fetch adapter ----------------------------------------------

FIXTURE_ROWS = [
    {"id": "W1", "title": "HBM-aware KV cache compression",
     "abstract": "We compress the KV cache for HBM GPUs.",
     "year": 2026, "venue": "ISCA", "url": "u1"},
    {"id": "W2", "title": "Prefetching for graph workloads",
     "abstract": "Nothing about caches here.",
     "year": 2025, "venue": "MICRO", "url": "u2"},
    {"id": "W3", "title": "Old KV cache compression on HBM",
     "abstract": "KV cache compression HBM",
     "year": 2019, "venue": "ISCA", "url": "u3"},
]


def _db(tmp_path):
    db = tmp_path / "venue.sqlite"
    build_db(FIXTURE_ROWS, db, years_back=3, venues=["ISCA", "MICRO"],
             today=date(2026, 7, 14))
    return db


def test_multi_fetch_verifies_pair_with_each_side_in_a_different_corpus(tmp_path):
    """Losing the union-level side guard turns cross-domain gaps unverified."""
    from mira.hypothesis.novelty import NoveltyChecker, classify

    memory_db = tmp_path / "memory.sqlite"
    optical_db = tmp_path / "optical.sqlite"
    build_db([{
        "id": "M1", "title": "Memory fabrics for accelerators",
        "abstract": "A scalable memory fabric for accelerators.",
        "year": 2026, "venue": "ISCA", "url": "memory",
    }], memory_db, years_back=3, venues=["ISCA"])
    build_db([{
        "id": "O1", "title": "Low-loss optical switches",
        "abstract": "An integrated optical switch with low insertion loss.",
        "year": 2026, "venue": "OFC", "url": "optical",
    }], optical_db, years_back=3, venues=["OFC"])

    checker = NoveltyChecker(
        tmp_path / "novelty.json",
        fetch=venue_corpus.make_multi_venue_fetch((memory_db, optical_db)),
        today=date(2026, 7, 27),
    )

    hits, titles = checker.check("memory fabric", "optical switch")

    assert hits == 0
    assert titles == []
    assert classify(hits) == "open gap"


def test_multi_fetch_unions_joint_hits_and_deduplicates_normalized_titles(tmp_path):
    """Removing normalized title de-dup double-counts cross-corpus papers."""
    first_db = tmp_path / "first.sqlite"
    second_db = tmp_path / "second.sqlite"
    common = {
        "abstract": "A memory fabric connected through an optical switch.",
        "year": 2026, "venue": "TEST", "url": "u",
    }
    build_db([
        {"id": "A1", "title": " Shared Discovery ", **common},
        {"id": "A2", "title": "First corpus joint hit", **common},
    ], first_db, years_back=3, venues=["ISCA"])
    build_db([
        {"id": "B1", "title": "shared discovery", **common},
        {"id": "B2", "title": "Second corpus joint hit", **common},
    ], second_db, years_back=3, venues=["OFC"])

    hits = venue_corpus.make_multi_venue_fetch((first_db, second_db))(
        '"memory fabric" "optical switch"')

    assert [hit["title"] for hit in hits] == [
        " Shared Discovery ",
        "First corpus joint hit",
        "Second corpus joint hit",
    ]


def test_multi_fetch_dedup_keeps_recent_copy_when_old_copy_is_first(tmp_path):
    """First-row de-dup must not hide a newer copy from recency filtering."""
    first_db = tmp_path / "old.sqlite"
    second_db = tmp_path / "recent.sqlite"
    build_db([{
        "id": "OLD", "title": " Shared Discovery ",
        "abstract": "A memory fabric connected through an optical switch.",
        "year": 2019, "venue": "OLD", "url": "old",
    }], first_db, years_back=3, venues=["OLD"])
    build_db([{
        "id": "NEW", "title": "shared discovery",
        "abstract": "A memory fabric connected through an optical switch.",
        "year": 2026, "venue": "NEW", "url": "new",
    }], second_db, years_back=3, venues=["NEW"])

    hits = venue_corpus.make_multi_venue_fetch((first_db, second_db))(
        '"memory fabric" "optical switch"')

    assert hits == [{"title": "shared discovery", "year": 2026}]


def test_multi_fetch_raises_when_side_is_absent_from_every_corpus(tmp_path):
    """Dropping the union-level absence guard fabricates an open gap."""
    fetch = venue_corpus.make_multi_venue_fetch((_db(tmp_path),))

    with pytest.raises(VenueCorpusError, match="Quantum Blockchain Teleportation"):
        fetch('"Quantum Blockchain Teleportation" "HBM"')


def test_multi_fetch_skips_missing_db_when_another_is_readable(tmp_path):
    """Treating one missing shard as fatal discards usable union coverage."""
    fetch = venue_corpus.make_multi_venue_fetch((
        tmp_path / "missing.sqlite",
        _db(tmp_path),
    ))

    hits = fetch('"KV Cache Compression" "HBM"')

    assert {hit["title"] for hit in hits} == {
        "HBM-aware KV cache compression",
        "Old KV cache compression on HBM",
    }
    assert fetch.notes == [
        f"venue corpus unavailable ({tmp_path / 'missing.sqlite'}): missing"
    ]


def test_multi_fetch_skips_unreadable_db_when_another_is_readable(tmp_path):
    """One corrupt shard must not discard valid coverage from another shard."""
    broken = tmp_path / "broken.sqlite"
    broken.write_text("not a SQLite database")
    fetch = venue_corpus.make_multi_venue_fetch((broken, _db(tmp_path)))

    hits = fetch('"KV Cache Compression" "HBM"')

    assert {hit["title"] for hit in hits} == {
        "HBM-aware KV cache compression",
        "Old KV cache compression on HBM",
    }
    assert len(fetch.notes) == 1
    assert str(broken) in fetch.notes[0]


def test_multi_fetch_raises_when_every_db_is_missing(tmp_path):
    fetch = venue_corpus.make_multi_venue_fetch((
        tmp_path / "missing-memory.sqlite",
        tmp_path / "missing-optical.sqlite",
    ))

    with pytest.raises(VenueCorpusError, match="missing-memory.sqlite"):
        fetch('"memory fabric" "optical switch"')


def test_multi_fetch_unquoted_query_returns_empty_even_when_dbs_are_missing(tmp_path):
    """Parsing after corpus validation would break the established loose-query path."""
    fetch = venue_corpus.make_multi_venue_fetch((tmp_path / "missing.sqlite",))

    assert fetch("memory fabric optical switch") == []


@pytest.mark.parametrize(
    ("hit_count", "expected_label"),
    [(3, "sparsely explored"), (20, "dropped")],
)
def test_multi_fetch_classification_boundaries_match_single_db(
    tmp_path, hit_count, expected_label
):
    """Union adaptation must not shift either established classify boundary."""
    from mira.hypothesis.novelty import NoveltyChecker, classify

    rows = [{
        "id": f"W{index}",
        "title": f"Memory optical integration {index}",
        "abstract": "A memory fabric using an optical switch.",
        "year": 2026, "venue": "TEST", "url": f"u{index}",
    } for index in range(hit_count)]
    db = tmp_path / f"boundary-{hit_count}.sqlite"
    build_db(rows, db, years_back=3, venues=["TEST"])
    single = NoveltyChecker(
        tmp_path / f"single-{hit_count}.json",
        fetch=make_venue_fetch(db),
        today=date(2026, 7, 27),
    )
    multi = NoveltyChecker(
        tmp_path / f"multi-{hit_count}.json",
        fetch=venue_corpus.make_multi_venue_fetch((db,)),
        today=date(2026, 7, 27),
    )

    single_hits, _ = single.check("memory fabric", "optical switch")
    multi_hits, _ = multi.check("memory fabric", "optical switch")

    assert single_hits == multi_hits == hit_count
    assert classify(single_hits) == classify(multi_hits) == expected_label


def test_multi_venue_fetch_reports_isolated_empty_backoffs(tmp_path):
    first_fetch = venue_corpus.make_multi_venue_fetch(
        (tmp_path / "first.sqlite",))
    second_fetch = venue_corpus.make_multi_venue_fetch(
        (tmp_path / "second.sqlite",))

    assert first_fetch.backoffs == {}
    assert isinstance(first_fetch.backoffs, dict)
    assert first_fetch.backoffs is not second_fetch.backoffs


def test_storage_coverage_note_names_storage_only_unverified_topics():
    """Omitting provenance leaves legitimate storage unverifiability unexplained."""
    corpus = SimpleNamespace(
        topic_papers={
            "SSD Endurance": {"P-storage"},
            "Shared Tiering": {"P-shared"},
            "Memory Fabric": {"P-memory"},
        },
        paper_profiles={
            "P-storage": {"storage-innovation"},
            "P-shared": {"storage-innovation", "memory-innovation"},
            "P-memory": {"memory-innovation"},
        },
    )
    candidates = [
        SimpleNamespace(
            topic_a="SSD Endurance",
            topic_c="Memory Fabric",
            novelty_hits=None,
        ),
        SimpleNamespace(
            topic_a="Shared Tiering",
            topic_c="Memory Fabric",
            novelty_hits=None,
        ),
    ]
    note = hypothesize._storage_coverage_note(corpus, candidates)

    assert note == (
        "storage profiles have no venue corpus — novelty unverified for "
        "storage-only topics: SSD Endurance"
    )


def test_storage_only_topic_forces_numeric_candidate_to_unverified():
    """Vocabulary overlap cannot make storage-only coverage authoritative."""
    corpus = SimpleNamespace(
        topic_papers={
            "SSD Caching": {"P-storage"},
            "Memory Fabric": {"P-memory"},
        },
        paper_profiles={
            "P-storage": {"storage-innovation"},
            "P-memory": {"memory-innovation"},
        },
    )
    candidate = SimpleNamespace(
        topic_a="SSD Caching",
        topic_c="Memory Fabric",
        novelty_hits=4,
        external_titles=["Coincidental vocabulary overlap"],
        novelty_label="sparsely explored",
    )
    changed = hypothesize._enforce_storage_unverifiability(corpus, candidate)

    assert changed is True
    assert candidate.novelty_hits is None
    assert candidate.external_titles == []
    assert candidate.novelty_label == "unverified"
    assert hypothesize._storage_coverage_note(corpus, [candidate]) == (
        "storage profiles have no venue corpus — novelty unverified for "
        "storage-only topics: SSD Caching"
    )


def test_storage_coverage_is_applied_without_an_external_checker():
    """The --no-external path still needs the explicit storage coverage reason."""
    corpus = SimpleNamespace(
        topic_papers={"SSD Caching": {"P-storage"}},
        paper_profiles={"P-storage": {"storage-innovation"}},
    )
    candidate = SimpleNamespace(
        topic_a="SSD Caching",
        topic_c="Memory Fabric",
        novelty_hits=None,
        external_titles=[],
        novelty_label="unverified",
    )
    note = hypothesize._apply_storage_coverage(corpus, [candidate])

    assert note == (
        "storage profiles have no venue corpus — novelty unverified for "
        "storage-only topics: SSD Caching"
    )
    assert candidate.novelty_hits is None
    assert candidate.external_titles == []
    assert candidate.novelty_label == "unverified"


def test_partial_union_result_is_not_reused_after_missing_shard_is_repaired(
    tmp_path,
):
    """A degraded numeric result may serve this run but must not poison the next."""
    from mira.hypothesis.novelty import NoveltyChecker

    memory_db = tmp_path / "memory.sqlite"
    optical_db = tmp_path / "optical.sqlite"
    build_db([
        {
            "id": "M1", "title": "Memory fabric mechanisms",
            "abstract": "A memory fabric for accelerators.",
            "year": 2026, "venue": "ISCA", "url": "m1",
        },
        {
            "id": "M2", "title": "Optical switch mechanisms",
            "abstract": "An optical switch for scale-up networks.",
            "year": 2026, "venue": "ISCA", "url": "m2",
        },
    ], memory_db, years_back=3, venues=["ISCA"])
    cache = tmp_path / "combined-novelty.json"
    first = SimpleNamespace(
        topic_a="memory fabric", topic_c="optical switch",
        novelty_hits=None, external_titles=[], novelty_label="unchecked",
    )
    first_checker = NoveltyChecker(
        cache,
        fetch=venue_corpus.make_multi_venue_fetch((memory_db, optical_db)),
        today=date(2026, 7, 27),
    )

    hypothesize._check_candidate_novelty(first_checker, first)

    assert first.novelty_hits == 0
    assert first.novelty_label == "open gap"
    assert first_checker.fetch.notes
    assert "memory fabric || optical switch" not in json.loads(cache.read_text())

    build_db([{
        "id": "O1", "title": "Optical switching for memory fabrics",
        "abstract": "An optical switch connects the memory fabric.",
        "year": 2026, "venue": "OFC", "url": "o1",
    }], optical_db, years_back=3, venues=["OFC"])
    second = SimpleNamespace(
        topic_a="memory fabric", topic_c="optical switch",
        novelty_hits=None, external_titles=[], novelty_label="unchecked",
    )
    second_checker = NoveltyChecker(
        cache,
        fetch=venue_corpus.make_multi_venue_fetch((memory_db, optical_db)),
        today=date(2026, 7, 27),
    )
    hypothesize._check_candidate_novelty(second_checker, second)

    assert second.novelty_hits == 1
    assert second.external_titles == ["Optical switching for memory fabrics"]


def test_newer_union_member_invalidates_combined_novelty_cache(tmp_path):
    """A rebuilt member DB must invalidate results derived from its old contents."""
    from mira.hypothesis.novelty import NoveltyChecker

    memory_db = tmp_path / "memory.sqlite"
    optical_db = tmp_path / "optical.sqlite"
    build_db([{
        "id": "M1", "title": "Memory fabric mechanisms",
        "abstract": "A memory fabric and optical switch overview.",
        "year": 2026, "venue": "ISCA", "url": "m1",
    }], memory_db, years_back=3, venues=["ISCA"])
    build_db([{
        "id": "O1", "title": "Fresh optical memory integration",
        "abstract": "An optical switch connects the memory fabric.",
        "year": 2026, "venue": "OFC", "url": "o1",
    }], optical_db, years_back=3, venues=["OFC"])
    cache = tmp_path / "combined-novelty.json"
    cache.write_text(json.dumps({
        "memory fabric || optical switch": {"hits": 0, "titles": []},
    }))
    os.utime(cache, ns=(1_000_000_000, 1_000_000_000))
    os.utime(optical_db, ns=(2_000_000_000, 2_000_000_000))
    target = SimpleNamespace(
        venue_dbs=(memory_db, optical_db),
        novelty_cache=cache,
    )
    invalidated = hypothesize._invalidate_union_novelty_cache(target)
    checker = NoveltyChecker(
        cache,
        fetch=venue_corpus.make_multi_venue_fetch(target.venue_dbs),
        today=date(2026, 7, 27),
    )
    hits, _ = checker.check("memory fabric", "optical switch")

    assert invalidated is True
    assert hits == 2


def test_query_schema_failure_invalidates_older_combined_cache(tmp_path):
    """Readable metadata cannot make a shard with no FTS query schema usable."""
    from mira.hypothesis.novelty import NoveltyChecker

    memory_db = tmp_path / "memory.sqlite"
    malformed_db = tmp_path / "malformed-optical.sqlite"
    build_db([{
        "id": "M1", "title": "Current joint result",
        "abstract": "An optical switch connects the memory fabric.",
        "year": 2026, "venue": "ISCA", "url": "m1",
    }], memory_db, years_back=3, venues=["ISCA"])
    with sqlite3.connect(malformed_db) as con:
        con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        con.execute(
            "INSERT INTO meta(key, value) VALUES('built_at', '2026-07-27')")
    cache = tmp_path / "combined-novelty.json"
    cache.write_text(json.dumps({
        "memory fabric || optical switch": {
            "hits": 77, "titles": ["obsolete cached result"],
        },
    }))
    os.utime(memory_db, ns=(1_000_000_000, 1_000_000_000))
    os.utime(malformed_db, ns=(2_000_000_000, 2_000_000_000))
    os.utime(cache, ns=(3_000_000_000, 3_000_000_000))
    assert read_built_at(malformed_db) == date(2026, 7, 27)
    target = SimpleNamespace(
        venue_dbs=(memory_db, malformed_db),
        novelty_cache=cache,
    )

    invalidated = hypothesize._invalidate_union_novelty_cache(target)
    checker = NoveltyChecker(
        cache,
        fetch=venue_corpus.make_multi_venue_fetch(target.venue_dbs),
        today=date(2026, 7, 27),
    )
    candidate = SimpleNamespace(
        topic_a="memory fabric", topic_c="optical switch",
        novelty_hits=None, external_titles=[], novelty_label="unchecked",
    )
    hypothesize._check_candidate_novelty(checker, candidate)

    assert invalidated is True
    assert candidate.novelty_hits == 1
    assert candidate.external_titles == ["Current joint result"]
    assert checker.fetch.notes


def test_single_db_target_keeps_single_fetch_threshold_semantics(tmp_path):
    """Routing a singular target through normalized union de-dup shifts classify(3)."""
    from mira.hypothesis.novelty import NoveltyChecker, classify

    common = {
        "abstract": "A memory fabric using an optical switch.",
        "year": 2026, "venue": "TEST", "url": "u",
    }
    db = tmp_path / "single.sqlite"
    build_db([
        {"id": "W1", "title": "Shared Result", **common},
        {"id": "W2", "title": " shared result ", **common},
        {"id": "W3", "title": "Distinct Result", **common},
    ], db, years_back=3, venues=["TEST"])
    target = SimpleNamespace(venue_db=db, venue_dbs=(db,))
    checker = NoveltyChecker(
        tmp_path / "single-target.json",
        fetch=hypothesize._target_venue_fetch(target),
        today=date(2026, 7, 27),
    )

    hits, _ = checker.check("memory fabric", "optical switch")

    assert hits == 3
    assert classify(hits) == "sparsely explored"


def test_fetch_matches_pair_and_passes_year_through(tmp_path):
    fetch = make_venue_fetch(_db(tmp_path))
    hits = fetch('"AI/ML Memory: KV Cache Compression & Eviction" "HBM"')
    titles = {h["title"] for h in hits}
    assert "HBM-aware KV cache compression" in titles
    assert "Old KV cache compression on HBM" in titles  # no recency filter here
    assert "Prefetching for graph workloads" not in titles
    years = {h["title"]: h["year"] for h in hits}
    assert years["Old KV cache compression on HBM"] == 2019


def test_fetch_unquoted_query_returns_empty(tmp_path):
    fetch = make_venue_fetch(_db(tmp_path))
    assert fetch("CXL Processing-in-Memory") == []


def test_venue_fetch_reports_isolated_empty_backoffs(tmp_path):
    first_fetch = make_venue_fetch(tmp_path / "first.sqlite")
    second_fetch = make_venue_fetch(tmp_path / "second.sqlite")

    assert first_fetch.backoffs == {}
    assert isinstance(first_fetch.backoffs, dict)
    assert first_fetch.backoffs is not second_fetch.backoffs
    first_fetch.backoffs["topic"] = "near"
    assert second_fetch.backoffs == {}


def test_fetch_stems_inflected_phrases(tmp_path):
    fetch = make_venue_fetch(_db(tmp_path))
    hits = fetch('"KV Cache Compressions" "HBM GPUs"')  # plural forms
    assert "HBM-aware KV cache compression" in {h["title"] for h in hits}


def test_fetch_matches_synonym_vocabulary(tmp_path):
    # Regression (Oaken, ISCA'25): the paper says 'quantization', the topic
    # label says 'compression' — exact-phrase matching missed it and the
    # pair over-claimed "open gap (0 external papers)".
    rows = FIXTURE_ROWS + [
        {"id": "W4", "title": "Oaken: Hybrid KV Cache Quantization for LLM Serving",
         "abstract": "Online-offline KV cache quantization on HBM GPUs.",
         "year": 2025, "venue": "ISCA", "url": "u4"}]
    db = tmp_path / "venue-syn.sqlite"
    build_db(rows, db, years_back=3, venues=["ISCA"], today=date(2026, 7, 14))
    hits = make_venue_fetch(db)(
        '"AI/ML Memory: KV Cache Compression & Eviction" "HBM"')
    assert "Oaken: Hybrid KV Cache Quantization for LLM Serving" in {
        h["title"] for h in hits}


def test_fetch_matches_base_term_when_label_has_acronym_tail(tmp_path):
    # Regression (optical): "Microring Modulators - MRM" matched nothing —
    # the corpus writes "microring modulator" without the acronym, so the
    # tailed label found 0 standalone hits and the pair falsely degraded to
    # "unverified" despite real coverage.
    rows = [
        {"id": "W1", "title": "A silicon microring modulator for optical links",
         "abstract": "We demonstrate a microring modulator on silicon photonics.",
         "year": 2026, "venue": "OFC", "url": "u1"}]
    db = tmp_path / "venue-mrm.sqlite"
    build_db(rows, db, years_back=3, venues=["OFC"], today=date(2026, 7, 14))
    hits = make_venue_fetch(db)(
        '"Microring Modulators - MRM" "silicon photonics"')
    assert "A silicon microring modulator for optical links" in {
        h["title"] for h in hits}


def test_fetch_raises_when_side_has_no_corpus_vocabulary(tmp_path):
    # A side whose phrases match nothing standalone makes the pair
    # unverifiable — 0 joint hits would fabricate "open gap".
    fetch = make_venue_fetch(_db(tmp_path))
    with pytest.raises(VenueCorpusError):
        fetch('"Quantum Blockchain Teleportation" "HBM"')


def test_fetch_missing_db_raises_requests_compatible(tmp_path):
    fetch = make_venue_fetch(tmp_path / "nope.sqlite")
    with pytest.raises(requests.RequestException):
        fetch('"CXL" "PIM"')
    assert issubclass(VenueCorpusError, requests.RequestException)


def test_build_db_is_atomic_and_records_meta(tmp_path):
    db = _db(tmp_path)
    assert not db.with_suffix(".tmp").exists()
    assert read_built_at(db) == date(2026, 7, 14)


def test_read_built_at_missing_db_returns_none(tmp_path):
    assert read_built_at(tmp_path / "nope.sqlite") is None


# --- end-to-end through the real NoveltyChecker ----------------------------

def test_novelty_checker_end_to_end_with_venue_fetch(tmp_path):
    from mira.hypothesis.novelty import NoveltyChecker
    checker = NoveltyChecker(tmp_path / "cache.json",
                             fetch=make_venue_fetch(_db(tmp_path)),
                             today=date(2026, 7, 14))
    hits, titles = checker.check(
        "AI/ML Memory: KV Cache Compression & Eviction", "HBM")
    assert hits == 1                       # W1 counted; W3 (2019) fails recency
    assert titles == ["HBM-aware KV cache compression"]


def test_novelty_checker_degrades_on_missing_db(tmp_path):
    from mira.hypothesis.novelty import NoveltyChecker, classify
    checker = NoveltyChecker(tmp_path / "cache.json",
                             fetch=make_venue_fetch(tmp_path / "nope.sqlite"))
    hits, titles = checker.check("CXL", "PIM")
    assert hits is None and titles == []
    assert classify(hits) == "unverified"


def test_novelty_checker_unverified_when_side_vocabulary_missing(tmp_path):
    from mira.hypothesis.novelty import NoveltyChecker, classify
    checker = NoveltyChecker(tmp_path / "cache.json",
                             fetch=make_venue_fetch(_db(tmp_path)))
    hits, titles = checker.check("Quantum Blockchain Teleportation", "HBM")
    assert hits is None and titles == []
    assert classify(hits) == "unverified"

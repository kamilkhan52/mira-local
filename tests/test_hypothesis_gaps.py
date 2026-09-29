from datetime import date

from mira.hypothesis.corpus import Corpus
from mira.hypothesis.gaps import (
    GapCandidate,
    gap_facts,
    is_alias_pair,
    mine_gaps,
    score_candidates,
    topic_velocity,
)
from tests.test_hypothesis_corpus import make_fixture_graph


def _corpus() -> Corpus:
    return Corpus.build(make_fixture_graph(), "test")


def test_abc_gap_found_between_disconnected_topics():
    # CXL and PIM: no paper covers both; share related topic HBM4,
    # institution Samsung, author Kim.
    cands = mine_gaps(_corpus(), ["CXL"])
    pairs = {(c.topic_a, c.topic_c) for c in cands}
    assert ("CXL", "PIM") in pairs
    gap = next(c for c in cands if c.topic_c == "PIM")
    assert gap.shared_topics == ["HBM4"]
    assert gap.shared_institutions == ["Samsung"]
    assert gap.shared_authors == ["Kim"]
    assert gap.structural_score > 0


def test_explicit_pair_is_evaluated_before_broader_neighbors():
    corpus = _corpus()

    candidates = mine_gaps(corpus, ["CXL", "PIM"])

    assert (candidates[0].topic_a, candidates[0].topic_c) == ("CXL", "PIM")
    ranked = score_candidates(candidates, {})
    assert (ranked[0].topic_a, ranked[0].topic_c) == ("CXL", "PIM")


def test_expanded_pool_does_not_mark_incidental_pairs_as_explicit():
    corpus = _corpus()

    candidates = mine_gaps(
        corpus,
        ["CXL", "PIM", "ReRAM"],
        explicit_anchor_topics=["CXL", "PIM"],
    )

    explicit = {
        frozenset((candidate.topic_a, candidate.topic_c))
        for candidate in candidates
        if candidate.explicit_anchor_pair
    }
    assert explicit == {frozenset(("CXL", "PIM"))}


def test_explicit_pair_survives_the_broad_neighbor_intermediary_floor():
    corpus = _corpus()
    candidates = mine_gaps(
        corpus,
        ["CXL", "DDR6"],
        explicit_anchor_topics=["CXL", "DDR6"],
    )

    explicit = [
        candidate for candidate in candidates
        if candidate.explicit_anchor_pair
    ]
    assert [
        frozenset((candidate.topic_a, candidate.topic_c))
        for candidate in explicit
    ] == [frozenset(("CXL", "DDR6"))]


def test_pair_with_connecting_paper_is_disqualified():
    # P3 covers both CXL and HBM4 -> not a gap.
    cands = mine_gaps(_corpus(), ["CXL"])
    assert all(c.topic_c != "HBM4" for c in cands)


def test_pair_below_min_intermediaries_is_dropped():
    # CXL vs DDR6 share nothing -> excluded.
    cands = mine_gaps(_corpus(), ["CXL"])
    assert all(c.topic_c != "DDR6" for c in cands)


def test_is_alias_pair_acronym_and_version_variants():
    assert is_alias_pair("HBM", "High Bandwidth Memory")
    assert is_alias_pair("HBM3E", "High Bandwidth Memory")   # versioned acronym
    assert is_alias_pair("PIM", "Processing-in-Memory")


def test_is_alias_pair_token_subset_variants():
    assert is_alias_pair("3D Stacked DRAM/Memory", "3D Stacked Memory")
    assert is_alias_pair("Cache Hierarchy", "Cache Hierarchy & Data Placement")
    assert is_alias_pair("CXL", "CXL Memory Pooling")


def test_is_alias_pair_leaves_real_pairs_alone():
    assert not is_alias_pair("CXL", "PIM")
    assert not is_alias_pair("Memory Interface", "Security")
    assert not is_alias_pair("AI/ML Memory: KV Cache Compression & Eviction", "HBM")


class _AliasCorpus:
    """Minimal Corpus stand-in: three topics, two of which are aliases."""
    def __init__(self):
        self.topic_papers = {"HBM": {"P1"}, "High Bandwidth Memory": {"P2"},
                             "Security": {"P3"}}
        self.topic_institutions = {t: {"Samsung", "SK Hynix"}
                                   for t in self.topic_papers}
        self.topic_authors = {t: {"Kim"} for t in self.topic_papers}
        self.report_dates = []
        self.paper_dates = {}

    def related_topics(self, t):
        return set()


def test_mine_gaps_skips_alias_pairs():
    corpus = _AliasCorpus()
    pairs = {frozenset((c.topic_a, c.topic_c))
             for c in mine_gaps(corpus, sorted(corpus.topic_papers))}
    assert frozenset(("HBM", "High Bandwidth Memory")) not in pairs
    assert frozenset(("HBM", "Security")) in pairs  # real pairs still mined


def test_topic_velocity_measures_acceleration():
    corpus = _corpus()
    # 2 reports only -> window < 4 -> velocity is 0 by definition
    assert topic_velocity(corpus, "CXL") == 0.0
    # Synthetic corpus with 8 report dates and papers loaded onto the recent half
    dates = [date(2026, 5, d) for d in range(1, 9)]
    corpus.report_dates = dates
    corpus.topic_papers["CXL"] = {"P1", "P3", "X1", "X2"}
    corpus.paper_dates.update({
        "P1": dates[0], "P3": dates[5], "X1": dates[6], "X2": dates[7],
    })
    # recent half (dates[4:]) has 3 papers, prior half has 1 -> velocity 2
    assert topic_velocity(corpus, "CXL") == 2.0


def test_score_candidates_normalizes_and_ranks():
    a = GapCandidate("CXL", "PIM", ["HBM4"], ["Samsung"], ["Kim"], 0.0, 10.0)
    b = GapCandidate("CXL", "ReRAM", [], ["Samsung", "SK Hynix"], [], 0.0, 5.0)
    ranked = score_candidates([a, b], {("CXL", "PIM"): 0.2, ("CXL", "ReRAM"): 1.0})
    # a: 0.6*1.0 + 0.4*0.2 = 0.68 ; b: 0.6*0.5 + 0.4*1.0 = 0.70 -> b first
    assert [c.topic_c for c in ranked] == ["ReRAM", "PIM"]
    assert abs(ranked[0].combined_score - 0.70) < 1e-9
    assert abs(ranked[1].combined_score - 0.68) < 1e-9


def test_gap_facts_mentions_the_structural_evidence():
    cand = GapCandidate("CXL", "PIM", ["HBM4"], ["Samsung"], ["Kim"], 2.0, 10.0)
    text = gap_facts(cand)
    assert "CXL" in text and "PIM" in text
    assert "Samsung" in text and "Kim" in text and "HBM4" in text
    assert "no paper" in text.lower()

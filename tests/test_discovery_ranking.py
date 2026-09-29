import numpy as np

from mira.discovery.ranking import (
    HUB_DAMPING,
    NOVELTY_MULTIPLIERS,
    apply_hub_damping,
    final_score,
    semantic_scores,
    topic_degree,
)
from mira.hypothesis.corpus import Corpus
from mira.hypothesis.gaps import GapCandidate
from mira.hypothesis.vectors import EntityVectors
from tests.test_hypothesis_corpus import make_fixture_graph


def _corpus() -> Corpus:
    return Corpus.build(make_fixture_graph(), "test")


def _cand(a: str, c: str, structural: float = 10.0) -> GapCandidate:
    return GapCandidate(a, c, [], [], [], 0.0, structural)


def test_topic_degree_sums_papers_institutions_authors():
    corpus = _corpus()
    # CXL: papers {P1,P3}, institutions {Samsung, SK Hynix}, authors {Kim}
    assert topic_degree(corpus, "CXL") == 5
    # DDR6: papers {P4}, institutions {}, authors {Lee}
    assert topic_degree(corpus, "DDR6") == 2
    assert topic_degree(corpus, "unknown topic") == 0


def test_hub_damping_demotes_high_degree_pairs():
    corpus = _corpus()
    hub = _cand("CXL", "PIM")    # degrees 5 × 3
    niche = _cand("DDR6", "PIM")  # degrees 2 × 3
    apply_hub_damping([hub, niche], corpus)
    assert niche.structural_score > hub.structural_score
    expected_hub = 10.0 / (15 ** 0.5) ** HUB_DAMPING
    assert abs(hub.structural_score - expected_hub) < 1e-9


def test_hub_damping_never_divides_below_one():
    corpus = _corpus()
    c = _cand("unknown a", "unknown b")  # degree 0 → damping denominator clamps to 1
    apply_hub_damping([c], corpus)
    assert c.structural_score == 10.0


def test_semantic_scores_uses_paper_centroids_with_topic_fallback():
    corpus = _corpus()
    vecs = EntityVectors(
        ["P1", "P3", "P2", "DDR6"],
        np.array([[1, 0], [1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32),
    )
    cands = [_cand("CXL", "PIM"), _cand("CXL", "DDR6"), _cand("CXL", "nowhere")]
    scores = semantic_scores(vecs, corpus, cands)
    assert abs(scores[("CXL", "PIM")] - 0.8) < 1e-6     # P1+P3 centroid · P2
    assert scores[("CXL", "DDR6")] == 0.0               # P4 missing → falls back to "DDR6" vector, orthogonal
    assert scores[("CXL", "nowhere")] == 0.0            # no vector at all → 0


def test_final_score_composes_multipliers():
    assert final_score(0.8, "open gap", 0.9, 0.9) == 0.8 * 1.0 * 0.9 * 0.9
    assert final_score(0.8, "sparsely explored", 1.0, 1.0) == 0.8 * NOVELTY_MULTIPLIERS["sparsely explored"]
    assert final_score(0.8, "dropped", 1.0, 1.0) == 0.0
    assert final_score(0.8, "someday new label", 1.0, 1.0) == 0.8 * 0.85  # unknown → unverified treatment

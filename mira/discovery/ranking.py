"""Global ranking adjustments (spec §3 Stage 2 + Stage 5).

Hub damping: in a corpus-wide scan, mega-topics pair with everything and would
dominate on raw shared-neighbor counts, so structural scores are normalized by
the pair's topic degrees. Final score composes combined score with novelty,
feasibility, and ledger multipliers.
"""
from __future__ import annotations

import numpy as np

from mira.hypothesis.corpus import Corpus
from mira.hypothesis.gaps import GapCandidate
from mira.hypothesis.vectors import EntityVectors

HUB_DAMPING = 0.5  # exponent on sqrt(deg_a * deg_c); 0 disables damping

NOVELTY_MULTIPLIERS = {
    "open gap": 1.0,
    "sparsely explored": 0.7,
    "unverified": 0.85,
    "unchecked": 0.85,
    "dropped": 0.0,
}
_UNKNOWN_NOVELTY = 0.85  # treat unknown labels like "unverified"


def topic_degree(corpus: Corpus, topic: str) -> int:
    return (
        len(corpus.topic_papers.get(topic, ()))
        + len(corpus.topic_institutions.get(topic, ()))
        + len(corpus.topic_authors.get(topic, ()))
    )


def apply_hub_damping(candidates: list[GapCandidate], corpus: Corpus) -> None:
    for c in candidates:
        d = (topic_degree(corpus, c.topic_a) * topic_degree(corpus, c.topic_c)) ** 0.5
        c.structural_score /= max(d, 1.0) ** HUB_DAMPING


def semantic_scores(
    vectors: EntityVectors, corpus: Corpus, candidates: list[GapCandidate]
) -> dict[tuple[str, str], float]:
    """Pair cosine from cached per-topic centroids (paper centroid, falling
    back to the topic's own entity vector). One centroid per topic instead of
    one per pair side — a global scan touches each topic many times."""
    cache: dict[str, np.ndarray | None] = {}

    def centroid(topic: str) -> np.ndarray | None:
        if topic not in cache:
            c = vectors.centroid(corpus.topic_papers.get(topic, ()))
            cache[topic] = c if c is not None else vectors.centroid([topic])
        return cache[topic]

    scores: dict[tuple[str, str], float] = {}
    for c in candidates:
        ca, cb = centroid(c.topic_a), centroid(c.topic_c)
        scores[(c.topic_a, c.topic_c)] = (
            max(float(np.dot(ca, cb)), 0.0) if ca is not None and cb is not None else 0.0
        )
    return scores


def final_score(
    combined: float, novelty_label: str, feasibility_mult: float, ledger_mult: float
) -> float:
    return (
        combined
        * NOVELTY_MULTIPLIERS.get(novelty_label, _UNKNOWN_NOVELTY)
        * feasibility_mult
        * ledger_mult
    )

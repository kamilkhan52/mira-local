"""Deterministic gap mining (Swanson ABC linking) over a Corpus.

A gap candidate is a topic pair (A, C) with shared intermediaries but no
single paper covering both. Scores are transparent weighted counts — the
auditability of "why is this a gap" is a design requirement (spec §1, §10).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Corpus

W_SHARED_TOPIC = 1.0
W_SHARED_INSTITUTION = 2.0
W_SHARED_AUTHOR = 3.0
W_BRIDGE_INSTITUTION = 4.0   # flat boost when any institution researches both sides
W_VELOCITY = 1.5
SEMANTIC_WEIGHT = 0.4        # combined = 0.6 * structural_norm + 0.4 * semantic
MIN_SHARED_INTERMEDIARIES = 2
VELOCITY_WINDOW = 8          # trailing reports considered


@dataclass
class GapCandidate:
    topic_a: str
    topic_c: str
    shared_topics: list[str]
    shared_institutions: list[str]
    shared_authors: list[str]
    velocity: float
    structural_score: float
    semantic_score: float = 0.0
    combined_score: float = 0.0
    novelty_hits: int | None = None
    novelty_label: str = "unchecked"
    external_titles: list[str] = field(default_factory=list)
    critic_objection: str | None = None
    explicit_anchor_pair: bool = False
    domains_a: tuple[str, ...] = ()
    domains_c: tuple[str, ...] = ()


def topic_velocity(corpus: Corpus, topic: str) -> float:
    """Paper-count acceleration: papers in the recent half of the trailing
    report window minus papers in the prior half. Positive = accelerating."""
    window = corpus.report_dates[-VELOCITY_WINDOW:]
    if len(window) < 4:
        return 0.0
    recent_cut = window[len(window) // 2]
    dates = [
        corpus.paper_dates[p]
        for p in corpus.topic_papers.get(topic, ())
        if p in corpus.paper_dates and corpus.paper_dates[p] >= window[0]
    ]
    recent = sum(1 for d in dates if d >= recent_cut)
    return float(recent - (len(dates) - recent))


def _tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.casefold())


def _is_acronym(short: list[str], long_: list[str]) -> bool:
    if len(short) != 1 or len(long_) < 2:
        return False
    alpha = re.match(r"[a-z]+", short[0])
    return bool(alpha) and alpha.group() == "".join(w[0] for w in long_)


def is_acronym_pair(a: str, c: str) -> bool:
    """One name is the (possibly versioned) acronym of the other:
    'HBM' / 'HBM3E' vs 'High Bandwidth Memory'. High-precision — safe to merge
    on without vector support, unlike token-subset matches."""
    ta, tc = _tokens(a), _tokens(c)
    return _is_acronym(ta, tc) or _is_acronym(tc, ta)


def is_alias_pair(a: str, c: str) -> bool:
    """Two topic names that plausibly denote the same concept — an acronym,
    version, or qualifier variant. Such pairs are taxonomy noise, not research
    gaps. (Token-subset matches are safe as a pair guard but NOT as a merge
    rule: a bare 'Memory' topic would transitively collapse the taxonomy.)"""
    ta, tc = _tokens(a), _tokens(c)
    if set(ta) <= set(tc) or set(tc) <= set(ta):
        return True
    return is_acronym_pair(a, c)


def mine_gaps(
    corpus: Corpus,
    anchor_topics: list[str],
    *,
    explicit_anchor_topics: list[str] | None = None,
) -> list[GapCandidate]:
    topics = set(corpus.topic_papers)
    pool = [t for t in anchor_topics if t in topics]
    explicit_pool = [
        topic
        for topic in (
            explicit_anchor_topics
            if explicit_anchor_topics is not None
            else anchor_topics
        )
        if topic in topics
    ]
    explicit_pairs = {
        frozenset((left, right))
        for index, left in enumerate(explicit_pool)
        for right in explicit_pool[index + 1:]
    }
    seen: set[frozenset[str]] = set()
    candidates: list[GapCandidate] = []
    for a in pool:
        for c in sorted(topics - {a}):
            key = frozenset((a, c))
            if key in seen:
                continue
            seen.add(key)
            if is_alias_pair(a, c):
                continue  # name variants of one concept, not a gap
            if corpus.topic_papers[a] & corpus.topic_papers[c]:
                continue  # a paper already connects them — provably not a gap
            shared_topics = sorted(corpus.related_topics(a) & corpus.related_topics(c))
            shared_inst = sorted(
                corpus.topic_institutions.get(a, set()) & corpus.topic_institutions.get(c, set())
            )
            shared_auth = sorted(
                corpus.topic_authors.get(a, set()) & corpus.topic_authors.get(c, set())
            )
            is_explicit = key in explicit_pairs
            if (
                not is_explicit
                and len(shared_topics) + len(shared_inst) + len(shared_auth)
                < MIN_SHARED_INTERMEDIARIES
            ):
                continue
            velocity = max(topic_velocity(corpus, a), topic_velocity(corpus, c))
            structural = (
                W_SHARED_TOPIC * len(shared_topics)
                + W_SHARED_INSTITUTION * len(shared_inst)
                + W_SHARED_AUTHOR * len(shared_auth)
                + (W_BRIDGE_INSTITUTION if shared_inst else 0.0)
                + W_VELOCITY * max(velocity, 0.0)
            )
            candidate = GapCandidate(
                a, c, shared_topics, shared_inst, shared_auth, velocity,
                structural,
            )
            candidate.explicit_anchor_pair = is_explicit
            candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda candidate: not candidate.explicit_anchor_pair,
    )


def score_candidates(
    candidates: list[GapCandidate],
    semantic_scores: dict[tuple[str, str], float],
) -> list[GapCandidate]:
    if not candidates:
        return []
    max_structural = max(c.structural_score for c in candidates) or 1.0
    for c in candidates:
        c.semantic_score = semantic_scores.get((c.topic_a, c.topic_c), 0.0)
        c.combined_score = (
            (1 - SEMANTIC_WEIGHT) * (c.structural_score / max_structural)
            + SEMANTIC_WEIGHT * c.semantic_score
        )
    return sorted(
        candidates,
        key=lambda candidate: (
            not candidate.explicit_anchor_pair,
            -candidate.combined_score,
            candidate.topic_a.casefold(),
            candidate.topic_c.casefold(),
        ),
    )


def gap_facts(cand: GapCandidate) -> str:
    """Human/LLM-readable statement of the verified structural facts."""
    lines = [
        f"Gap: '{cand.topic_a}' × '{cand.topic_c}' — no paper in the corpus covers both topics.",
    ]
    if cand.domains_a and cand.domains_c:
        lines.append(
            f"Compiled evidence provenance: {cand.topic_a} "
            f"({', '.join(cand.domains_a)}) × {cand.topic_c} "
            f"({', '.join(cand.domains_c)})."
        )
    if cand.shared_institutions:
        lines.append(
            f"Institutions researching BOTH sides: {', '.join(cand.shared_institutions)} "
            "(capability exists; publication does not)."
        )
    if cand.shared_authors:
        lines.append(f"Authors publishing on both sides: {', '.join(cand.shared_authors)}.")
    if cand.shared_topics:
        lines.append(f"Bridging topics adjacent to both: {', '.join(cand.shared_topics)}.")
    if cand.velocity > 0:
        lines.append(
            f"Momentum: one side gained {cand.velocity:g} net papers in the recent half "
            "of the trailing report window."
        )
    return "\n".join(lines)

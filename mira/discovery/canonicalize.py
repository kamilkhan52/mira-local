"""In-memory Topic node canonicalization (spec §3 Stage 0).

Near-duplicate Topic nodes ("CXL" vs "CXL (Compute Express Link)") flood a
global gap scan with fake pairs, so they are merged at load time. The GraphML
on disk is never modified; merges apply to the loaded nx.Graph only, and the
merge map is emitted in the agenda appendix for auditability.

Merge rules (union-find over all Topic names):
- identical after normalize_topic().casefold()          -> always merge
- one name is the (versioned) acronym of the other      -> always merge
- one name token-extends the other (front or back) AND
  entity-vector cosine >= EXT_SIM_THRESHOLD             -> merge
- entity-vector cosine >= SIM_THRESHOLD                 -> merge
Without vectors only the first two rules apply (safe degradation).
Canonical member = highest graph degree (ties: lexicographic).
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx

from mira.graph_ingest import normalize_topic
from mira.hypothesis.gaps import is_acronym_pair
from mira.hypothesis.vectors import EntityVectors

SIM_THRESHOLD = 0.95      # pure vector-similarity merge
EXT_SIM_THRESHOLD = 0.80  # token-extension merge needs this much vector support


def _key(name: str) -> str:
    return normalize_topic(name).casefold()


def _is_extension(a: str, b: str) -> bool:
    ta, tb = tuple(_key(a).split()), tuple(_key(b).split())
    if len(ta) == len(tb):
        return False
    short, long_ = (ta, tb) if len(ta) < len(tb) else (tb, ta)
    return long_[: len(short)] == short or long_[-len(short):] == short


def _similarity(vectors: EntityVectors | None, a: str, b: str) -> float:
    if vectors is None:
        return 0.0
    return vectors.side_similarity([a], [b])


def build_merge_map(
    topics: Iterable[str], g: nx.Graph, vectors: EntityVectors | None
) -> dict[str, str]:
    """Return alias -> canonical for every mergeable topic group."""
    topics = sorted(topics)
    parent = {t: t for t in topics}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    by_key: dict[str, str] = {}
    for t in topics:
        k = _key(t)
        if k in by_key:
            union(t, by_key[k])
        else:
            by_key[k] = t

    for i, a in enumerate(topics):
        for b in topics[i + 1:]:
            if find(a) == find(b):
                continue
            if is_acronym_pair(a, b):
                union(a, b)
                continue
            sim = _similarity(vectors, a, b)
            if sim >= SIM_THRESHOLD or (_is_extension(a, b) and sim >= EXT_SIM_THRESHOLD):
                union(a, b)

    groups: dict[str, list[str]] = {}
    for t in topics:
        groups.setdefault(find(t), []).append(t)

    merge_map: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = max(members, key=lambda t: (g.degree(t) if t in g else 0, t))
        for m in members:
            if m != canonical:
                merge_map[m] = canonical
    return merge_map


def merge_topics_in_graph(g: nx.Graph, merge_map: dict[str, str]) -> None:
    """Move each alias's edges onto its canonical node, then drop the alias.
    Pre-existing canonical edges win on conflict; self-loops are skipped.
    When several aliases share an edge to the same neighbor, the first alias
    in sorted order wins; later attributes are dropped."""
    for alias, canonical in sorted(merge_map.items()):
        if alias not in g:
            continue
        for nbr in list(g.neighbors(alias)):
            tgt = merge_map.get(nbr, nbr)
            if tgt == canonical or g.has_edge(canonical, tgt):
                continue
            g.add_edge(canonical, tgt, **g[alias][nbr])
        g.remove_node(alias)

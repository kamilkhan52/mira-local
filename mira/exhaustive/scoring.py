"""Exhaustive, threshold-based scoring over all nodes and edges."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from .snapshots import DomainSnapshot
from .types import (
    DOMAIN_NAMES,
    DomainName,
    ScoredEdge,
    ScoredNode,
    SelectedRegion,
)


THRESHOLD_VERSION = "exhaustive-v1"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NODE_TYPE_WEIGHT = {
    "Topic": 1.0,
    "Paper": 0.8,
    "Article": 0.8,
    "Institution": 0.5,
    "Author": 0.35,
    "Report": 0.2,
    "Publication": 0.2,
}


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.casefold()))


def _lexical_score(query_tokens: set[str], value: str) -> float:
    value_tokens = _tokens(value)
    if not query_tokens or not value_tokens:
        return 0.0
    return len(query_tokens & value_tokens) / len(query_tokens | value_tokens)


def _exact_score(
    normalized_query: str,
    query_tokens: set[str],
    node_id: str,
) -> float:
    normalized_node = " ".join(_TOKEN_RE.findall(node_id.casefold()))
    if normalized_node and normalized_node in normalized_query:
        return 1.0
    node_tokens = _TOKEN_RE.findall(node_id)
    initials = "".join(token[0] for token in node_tokens if token)
    return float(len(initials) >= 2 and initials.upper() in query_tokens)


def _normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError("query embedding must have a non-zero norm")
    return array / norm


@dataclass(frozen=True, slots=True)
class ScoringResult:
    nodes: tuple[ScoredNode, ...]
    edges: tuple[ScoredEdge, ...]
    regions: tuple[SelectedRegion, ...]
    nodes_by_domain: Mapping[DomainName, int]
    edges_by_domain: Mapping[DomainName, int]
    threshold_version: str = THRESHOLD_VERSION

    @property
    def nodes_evaluated(self) -> int:
        return len(self.nodes)

    @property
    def edges_evaluated(self) -> int:
        return len(self.edges)

    @property
    def selected_node_keys(self) -> tuple[tuple[DomainName, str], ...]:
        return tuple(
            (node.domain, node.node_id) for node in self.nodes if node.selected
        )

    @property
    def selected_edge_keys(
        self,
    ) -> tuple[tuple[DomainName, str, str], ...]:
        return tuple(
            (edge.domain, edge.source, edge.target)
            for edge in self.edges if edge.selected
        )


class QueryScorer:
    def __init__(
        self,
        embed_query: Callable[[str], np.ndarray],
        *,
        node_threshold: float = 0.42,
        edge_threshold: float = 0.38,
    ):
        if not 0.0 <= node_threshold <= 1.0:
            raise ValueError("node_threshold must be between 0 and 1")
        if not 0.0 <= edge_threshold <= 1.0:
            raise ValueError("edge_threshold must be between 0 and 1")
        self.embed_query = embed_query
        self.node_threshold = node_threshold
        self.edge_threshold = edge_threshold

    def score_all(
        self,
        query: str,
        snapshots: Mapping[DomainName, DomainSnapshot],
    ) -> ScoringResult:
        query_vector = _normalize(self.embed_query(query))
        query_words = _TOKEN_RE.findall(query.casefold())
        query_tokens = set(query_words)
        normalized_query = " ".join(query_words)
        exact_query_tokens = {token.upper() for token in query_words}
        occurrences: dict[str, int] = {}
        for snapshot in snapshots.values():
            for name in snapshot.entity_names:
                occurrences[name] = occurrences.get(name, 0) + 1

        node_records: list[ScoredNode] = []
        edge_records: list[ScoredEdge] = []
        regions: list[SelectedRegion] = []
        nodes_by_domain = {}
        edges_by_domain = {}

        for domain in DOMAIN_NAMES:
            snapshot = snapshots[domain]
            if snapshot.entity_matrix.shape[1] != query_vector.size:
                raise ValueError(
                    f"{domain} vector dimension "
                    f"{snapshot.entity_matrix.shape[1]} does not match query "
                    f"dimension {query_vector.size}"
                )
            if (
                snapshot.relation_matrix.size
                and snapshot.relation_matrix.shape[1] != query_vector.size
            ):
                raise ValueError(
                    f"{domain} relationship vector dimension "
                    f"{snapshot.relation_matrix.shape[1]} does not match query "
                    f"dimension {query_vector.size}"
                )
            semantic_scores = snapshot.entity_matrix @ query_vector
            relationship_semantic = {
                frozenset(key.split("<SEP>", 1)): max(
                    0.0,
                    float(snapshot.relation_matrix[index] @ query_vector),
                )
                for index, key in enumerate(snapshot.relation_keys)
                if "<SEP>" in key
            }
            base_scores: dict[str, float] = {}
            for index, node_id in enumerate(snapshot.entity_names):
                data = snapshot.graph.nodes[node_id]
                node_type = str(data.get("entity_type", ""))
                searchable = " ".join((
                    node_id,
                    node_type,
                    str(data.get("description", "")),
                ))
                semantic = max(float(semantic_scores[index]), 0.0)
                base_scores[node_id] = min(
                    1.0,
                    0.65 * semantic
                    + 0.20 * _lexical_score(query_tokens, searchable)
                    + 0.10 * _exact_score(
                        normalized_query,
                        exact_query_tokens,
                        node_id,
                    )
                    + 0.05 * _NODE_TYPE_WEIGHT.get(node_type, 0.1),
                )

            propagation = {node_id: 0.0 for node_id in snapshot.graph.nodes}
            relationship_relevance: dict[frozenset[str], float] = {}
            for source, target, data in snapshot.graph.edges(data=True):
                relationship = " ".join((
                    str(data.get("keywords", "")),
                    str(data.get("description", "")),
                ))
                edge_key = frozenset((source, target))
                lexical = _lexical_score(query_tokens, relationship)
                relevance = max(
                    lexical,
                    relationship_semantic.get(edge_key, 0.0),
                )
                relationship_relevance[edge_key] = relevance
                relation_factor = 0.5 + 0.5 * relevance
                propagation[source] = max(
                    propagation[source],
                    0.10 * base_scores[target] * relation_factor,
                )
                propagation[target] = max(
                    propagation[target],
                    0.10 * base_scores[source] * relation_factor,
                )

            propagated = {
                node_id: min(1.0, base_scores[node_id] + propagation[node_id])
                for node_id in snapshot.graph.nodes
            }
            selected_nodes = {
                node_id
                for node_id, score in propagated.items()
                if score >= self.node_threshold
            }
            domain_nodes = [
                ScoredNode(
                    domain=domain,
                    node_id=node_id,
                    node_type=str(
                        snapshot.graph.nodes[node_id].get("entity_type", "")
                    ),
                    base_score=base_scores[node_id],
                    propagated_score=propagated[node_id],
                    selected=node_id in selected_nodes,
                )
                for node_id in snapshot.graph.nodes
            ]
            domain_nodes.sort(key=lambda item: item.node_id.casefold())
            node_records.extend(domain_nodes)

            selected_edges: set[tuple[str, str]] = set()
            domain_edges = []
            for source, target, data in snapshot.graph.edges(data=True):
                source, target = sorted((str(source), str(target)))
                relevance = relationship_relevance[
                    frozenset((source, target))
                ]
                bridge = float(
                    occurrences.get(source, 0) > 1
                    or occurrences.get(target, 0) > 1
                )
                score = min(
                    1.0,
                    0.40 * propagated[source]
                    + 0.40 * propagated[target]
                    + 0.15 * relevance
                    + 0.05 * bridge,
                )
                selected = score >= self.edge_threshold
                if selected:
                    selected_edges.add((source, target))
                domain_edges.append(ScoredEdge(
                    domain=domain,
                    source=source,
                    target=target,
                    keywords=str(data.get("keywords", "")),
                    score=score,
                    selected=selected,
                ))
            domain_edges.sort(
                key=lambda item: (item.source.casefold(), item.target.casefold())
            )
            edge_records.extend(domain_edges)
            nodes_by_domain[domain] = snapshot.graph.number_of_nodes()
            edges_by_domain[domain] = snapshot.graph.number_of_edges()
            regions.append(SelectedRegion(
                domain=domain,
                node_ids=tuple(sorted(selected_nodes, key=str.casefold)),
                edge_keys=tuple(sorted(selected_edges)),
            ))

        return ScoringResult(
            nodes=tuple(node_records),
            edges=tuple(edge_records),
            regions=tuple(regions),
            nodes_by_domain=MappingProxyType(nodes_by_domain),
            edges_by_domain=MappingProxyType(edges_by_domain),
        )

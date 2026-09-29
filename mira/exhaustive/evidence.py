"""Typed evidence expansion after exhaustive graph scoring."""

from __future__ import annotations

import hashlib
import logging
from types import MappingProxyType
from typing import Mapping

from .scoring import ScoringResult
from .snapshots import DomainSnapshot
from .types import (
    DOMAIN_NAMES,
    DomainName,
    EvidenceChunk,
    EvidenceProvenance,
    ResearchEvidenceBundle,
)


logger = logging.getLogger(__name__)

PAPER_TOPIC_KEYWORDS = {"primary_topic topic", "also_covers topic"}
SELECTED_IN_KEYWORD = "selected_in report"
RELATED_TOPIC_KEYWORD = "related_to topic"
RESEARCHES_KEYWORD = "researches topic"
AUTHORED_BY_KEYWORD = "authored_by author"
APPROVED_KEYWORDS = (
    PAPER_TOPIC_KEYWORDS
    | {
        SELECTED_IN_KEYWORD,
        RELATED_TOPIC_KEYWORD,
        RESEARCHES_KEYWORD,
        AUTHORED_BY_KEYWORD,
    }
)


class EvidenceIntegrityError(RuntimeError):
    """Evidence cannot be proven complete for the scored snapshots."""


def _has_keyword(raw: str, expected: str) -> bool:
    return expected.casefold() in raw.casefold()


def _approved_keyword(raw: str) -> bool:
    return any(_has_keyword(raw, expected) for expected in APPROVED_KEYWORDS)


def _normalized_content(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


class EvidenceCollector:
    """Expand scored regions into the chunks the compiler will pay to read.

    Scoring itself is unbounded by design. Expansion is not: the compiled
    corpus is billed per token, so `max_papers` and `max_chunks_per_paper` cap
    the fan-out deterministically, keeping the highest-scored evidence first.
    """

    def __init__(
        self,
        *,
        max_papers: int = 400,
        max_chunks_per_paper: int = 40,
    ):
        if max_papers <= 0 or max_chunks_per_paper <= 0:
            raise ValueError("evidence fan-out limits must be positive")
        self.max_papers = max_papers
        self.max_chunks_per_paper = max_chunks_per_paper

    def collect(
        self,
        query: str,
        scoring: ScoringResult,
        snapshots: Mapping[DomainName, DomainSnapshot],
    ) -> ResearchEvidenceBundle:
        self._validate_coverage(scoring, snapshots)
        aggregates: dict[str, dict] = {}
        regions = {region.domain: region for region in scoring.regions}
        node_scores = {
            (node.domain, node.node_id): node.propagated_score
            for node in scoring.nodes
        }
        edge_scores = {
            (edge.domain, edge.source, edge.target): edge.score
            for edge in scoring.edges
        }

        expanded = {
            domain: self._expand_domain(domain, snapshots[domain], regions[domain])
            for domain in DOMAIN_NAMES
        }
        pre_cap_papers = {
            paper
            for (_nodes, _edges, papers) in expanded.values()
            for paper in papers
        }
        all_papers = self._capped_papers(expanded, node_scores)
        # Coverage honesty (review round 2): the bundle must not claim
        # exhaustiveness over a corpus the caps truncated. Track every
        # truncation point so `summarize_coverage().exhaustive` reflects what
        # actually happened and a warning can be emitted.
        papers_dropped_by_cap = len(pre_cap_papers) - len(all_papers)
        chunks_dropped_by_cap = 0

        for domain in DOMAIN_NAMES:
            snapshot = snapshots[domain]
            evidence_nodes, evidence_edges, domain_papers = expanded[domain]
            # Papers cut by the cap take their chunks with them; an edge that
            # only exists to reach one is no longer evidence.
            dropped = domain_papers - all_papers
            if dropped:
                evidence_nodes -= dropped
                evidence_edges = {
                    edge for edge in evidence_edges
                    if not dropped & set(edge)
                }
            paper_nodes = domain_papers & all_papers
            chunk_papers: dict[str, set[str]] = {}
            chunk_scores: dict[str, float] = {}
            selected_chunk_ids: set[str] = set()
            for node_id in evidence_nodes:
                node_type = snapshot.graph.nodes[node_id].get(
                    "entity_type", ""
                )
                if node_type in {"Author", "Institution"}:
                    continue
                score = node_scores.get((domain, node_id), 0.0)
                for chunk_id in snapshot.entity_chunks.get(node_id, ()):
                    selected_chunk_ids.add(chunk_id)
                    chunk_scores[chunk_id] = max(
                        chunk_scores.get(chunk_id, 0.0), score
                    )
                    if node_id in paper_nodes:
                        chunk_papers.setdefault(chunk_id, set()).add(node_id)

            for source, target in evidence_edges:
                chunk_ids = set(snapshot.relation_chunks.get(
                    f"{source}<SEP>{target}", ()
                ))
                chunk_ids.update(snapshot.relation_chunks.get(
                    f"{target}<SEP>{source}", ()
                ))
                related_papers = {source, target} & paper_nodes
                score = edge_scores.get((domain, source, target), 0.0)
                for chunk_id in chunk_ids:
                    selected_chunk_ids.add(chunk_id)
                    chunk_scores[chunk_id] = max(
                        chunk_scores.get(chunk_id, 0.0), score
                    )
                    chunk_papers.setdefault(chunk_id, set()).update(
                        related_papers
                    )

            pre_cap_count = len(selected_chunk_ids)
            selected_chunk_ids = self._capped_chunks(
                selected_chunk_ids, chunk_papers, chunk_scores
            )
            chunks_dropped_by_cap += pre_cap_count - len(selected_chunk_ids)

            for chunk_id in sorted(selected_chunk_ids):
                record = snapshot.text_chunks.get(chunk_id)
                if record is None:
                    raise EvidenceIntegrityError(
                        f"{domain}: selected evidence references missing "
                        f"chunk {chunk_id}"
                    )
                content = record.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise EvidenceIntegrityError(
                        f"{domain}: selected chunk {chunk_id} has no text"
                    )
                normalized = _normalized_content(content)
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                content_hash = f"sha256:{digest}"
                aggregate = aggregates.setdefault(content_hash, {
                    "content": normalized,
                    "domains": set(),
                    "file_paths": set(),
                    "paper_names": set(),
                    "source_chunk_ids": set(),
                    "provenance": set(),
                })
                aggregate["domains"].add(domain)
                file_path = record.get("file_path") or record.get("full_doc_id")
                if file_path:
                    aggregate["file_paths"].add(str(file_path))
                papers = chunk_papers.get(chunk_id) or {""}
                if len(papers) > 1:
                    # One chunk attributed to several papers means the citation
                    # it produces cannot say which paper the text came from.
                    logger.warning(
                        "%s: chunk %s is attributed to %d papers (%s); its "
                        "provenance names all of them",
                        domain,
                        chunk_id,
                        len(papers),
                        ", ".join(sorted(papers)),
                    )
                aggregate["paper_names"].update(
                    paper for paper in papers if paper
                )
                aggregate["source_chunk_ids"].add(chunk_id)
                aggregate["provenance"].update(
                    (
                        chunk_id,
                        domain,
                        str(file_path or ""),
                        paper,
                    )
                    for paper in papers
                )

        chunks = []
        for content_hash, aggregate in sorted(aggregates.items()):
            source_ids = tuple(sorted(aggregate["source_chunk_ids"]))
            chunks.append(EvidenceChunk(
                chunk_id=source_ids[0],
                content_hash=content_hash,
                content=aggregate["content"],
                domains=tuple(
                    domain for domain in DOMAIN_NAMES
                    if domain in aggregate["domains"]
                ),
                file_paths=tuple(sorted(aggregate["file_paths"])),
                paper_names=tuple(sorted(
                    aggregate["paper_names"], key=str.casefold
                )),
                source_chunk_ids=source_ids,
                provenance=tuple(
                    EvidenceProvenance(
                        source_chunk_id=source_chunk_id,
                        domain=domain,
                        file_path=file_path,
                        paper_name=paper_name,
                    )
                    for (
                        source_chunk_id,
                        domain,
                        file_path,
                        paper_name,
                    ) in sorted(aggregate["provenance"])
                ),
            ))

        bundle = ResearchEvidenceBundle(
            query=query,
            snapshot_fingerprints=MappingProxyType({
                domain: snapshots[domain].fingerprint for domain in DOMAIN_NAMES
            }),
            nodes_scanned=MappingProxyType({
                domain: snapshots[domain].graph.number_of_nodes()
                for domain in DOMAIN_NAMES
            }),
            edges_scanned=MappingProxyType({
                domain: snapshots[domain].graph.number_of_edges()
                for domain in DOMAIN_NAMES
            }),
            selected_regions=scoring.regions,
            paper_names=tuple(sorted(all_papers, key=str.casefold)),
            chunks=tuple(chunks),
            threshold_version=scoring.threshold_version,
            exhaustive=(papers_dropped_by_cap == 0 and chunks_dropped_by_cap == 0),
        )
        if papers_dropped_by_cap or chunks_dropped_by_cap:
            logger.warning(
                "evidence capped — %d paper(s), %d chunk(s) dropped by "
                "max_papers/max_chunks_per_paper; coverage is NOT exhaustive",
                papers_dropped_by_cap, chunks_dropped_by_cap,
            )
        return bundle

    def _expand_domain(
        self,
        domain: DomainName,
        snapshot: DomainSnapshot,
        region,
    ) -> tuple[set[str], set[tuple[str, str]], set[str]]:
        """Grow one domain's selected region into nodes, edges, and papers."""
        selected = set(region.node_ids)
        selected_edge_endpoints = {
            node_id
            for edge in region.edge_keys
            for node_id in edge
        }
        missing_endpoints = (
            selected_edge_endpoints - set(snapshot.graph.nodes)
        )
        if missing_endpoints:
            raise EvidenceIntegrityError(
                f"{domain}: selected edge references missing nodes "
                f"{sorted(missing_endpoints)}"
            )
        evidence_nodes = {
            node_id
            for node_id in selected | selected_edge_endpoints
            if snapshot.graph.nodes[node_id].get("entity_type")
            in {
                "Topic",
                "Paper",
                "Article",
                "Report",
                "Author",
                "Institution",
            }
        }
        evidence_edges: set[tuple[str, str]] = set()
        for source, target in region.edge_keys:
            data = snapshot.graph.get_edge_data(source, target)
            if data and _approved_keyword(str(data.get("keywords", ""))):
                evidence_edges.add(tuple(sorted((source, target))))

        for selected_node in selected | selected_edge_endpoints:
            node_type = snapshot.graph.nodes[selected_node].get(
                "entity_type", ""
            )
            for neighbor in snapshot.graph.neighbors(selected_node):
                neighbor_type = snapshot.graph.nodes[neighbor].get(
                    "entity_type", ""
                )
                keywords = str(
                    snapshot.graph[selected_node][neighbor].get(
                        "keywords", ""
                    )
                )
                is_topic_paper = (
                    node_type == "Topic"
                    and neighbor_type in {"Paper", "Article"}
                    and any(
                        _has_keyword(keywords, expected)
                        for expected in PAPER_TOPIC_KEYWORDS
                    )
                )
                is_report_paper = (
                    {node_type, neighbor_type} == {"Report", "Paper"}
                    and _has_keyword(keywords, SELECTED_IN_KEYWORD)
                )
                if is_topic_paper or is_report_paper:
                    evidence_nodes.add(str(neighbor))
                    evidence_edges.add(tuple(sorted(
                        (str(selected_node), str(neighbor))
                    )))

        for source, target, data in snapshot.graph.edges(data=True):
            source, target = str(source), str(target)
            keywords = str(data.get("keywords", ""))
            if (
                _approved_keyword(keywords)
                and source in evidence_nodes
                and target in evidence_nodes
            ):
                evidence_edges.add(tuple(sorted((source, target))))

        paper_nodes = {
            node_id
            for node_id in evidence_nodes
            if snapshot.graph.nodes[node_id].get("entity_type")
            in {"Paper", "Article"}
        }
        return evidence_nodes, evidence_edges, paper_nodes

    def _capped_papers(
        self,
        expanded: Mapping[DomainName, tuple[set, set, set[str]]],
        node_scores: Mapping[tuple[DomainName, str], float],
    ) -> set[str]:
        """Keep the highest-scored papers up to `max_papers`, ties by name."""
        best: dict[str, float] = {}
        for domain, (_nodes, _edges, papers) in expanded.items():
            for paper in papers:
                score = node_scores.get((domain, paper), 0.0)
                best[paper] = max(best.get(paper, 0.0), score)
        if len(best) <= self.max_papers:
            return set(best)
        ranked = sorted(best, key=lambda paper: (-best[paper], paper.casefold()))
        return set(ranked[:self.max_papers])

    def _capped_chunks(
        self,
        selected_chunk_ids: set[str],
        chunk_papers: Mapping[str, set[str]],
        chunk_scores: Mapping[str, float],
    ) -> set[str]:
        """Keep each paper's highest-scored chunks up to the per-paper cap.

        A chunk survives if it is within any of its papers' quotas. Chunks
        attributed to no paper are not covered by a per-paper cap and are
        kept as-is.
        """
        by_paper: dict[str, list[str]] = {}
        for chunk_id in selected_chunk_ids:
            for paper in chunk_papers.get(chunk_id, ()):  # noqa: SIM118
                by_paper.setdefault(paper, []).append(chunk_id)
        if all(
            len(chunk_ids) <= self.max_chunks_per_paper
            for chunk_ids in by_paper.values()
        ):
            return selected_chunk_ids
        kept = {
            chunk_id
            for chunk_id in selected_chunk_ids
            if not chunk_papers.get(chunk_id)
        }
        for chunk_ids in by_paper.values():
            chunk_ids.sort(key=lambda item: (-chunk_scores.get(item, 0.0), item))
            kept.update(chunk_ids[:self.max_chunks_per_paper])
        return kept

    def _validate_coverage(
        self,
        scoring: ScoringResult,
        snapshots: Mapping[DomainName, DomainSnapshot],
    ) -> None:
        missing = [domain for domain in DOMAIN_NAMES if domain not in snapshots]
        if missing:
            raise EvidenceIntegrityError(
                f"missing scored domain snapshots: {', '.join(missing)}"
            )
        region_domains = {region.domain for region in scoring.regions}
        if region_domains != set(DOMAIN_NAMES):
            raise EvidenceIntegrityError(
                "scoring result does not contain one region for every domain"
            )
        for domain in DOMAIN_NAMES:
            expected_nodes = snapshots[domain].graph.number_of_nodes()
            expected_edges = snapshots[domain].graph.number_of_edges()
            if scoring.nodes_by_domain.get(domain) != expected_nodes:
                raise EvidenceIntegrityError(
                    f"{domain}: scoring covered "
                    f"{scoring.nodes_by_domain.get(domain)} nodes; "
                    f"snapshot has {expected_nodes}"
                )
            if scoring.edges_by_domain.get(domain) != expected_edges:
                raise EvidenceIntegrityError(
                    f"{domain}: scoring covered "
                    f"{scoring.edges_by_domain.get(domain)} edges; "
                    f"snapshot has {expected_edges}"
                )

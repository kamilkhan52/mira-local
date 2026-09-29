"""Immutable contracts for exhaustive multi-graph retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Any, Literal, Mapping


DomainName = Literal["memory", "optical", "storage"]
DOMAIN_NAMES: tuple[DomainName, ...] = ("memory", "optical", "storage")


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class SnapshotFingerprint:
    files: tuple[FileFingerprint, ...]


@dataclass(frozen=True, slots=True)
class ScoredNode:
    domain: DomainName
    node_id: str
    node_type: str
    base_score: float
    propagated_score: float
    selected: bool


@dataclass(frozen=True, slots=True)
class ScoredEdge:
    domain: DomainName
    source: str
    target: str
    keywords: str
    score: float
    selected: bool


@dataclass(frozen=True, slots=True)
class SelectedRegion:
    domain: DomainName
    node_ids: tuple[str, ...]
    edge_keys: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    source_chunk_id: str
    domain: DomainName
    file_path: str
    paper_name: str


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    chunk_id: str
    content_hash: str
    content: str
    domains: tuple[DomainName, ...]
    file_paths: tuple[str, ...]
    paper_names: tuple[str, ...]
    source_chunk_ids: tuple[str, ...] = ()
    provenance: tuple[EvidenceProvenance, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchEvidenceBundle:
    query: str
    snapshot_fingerprints: Mapping[DomainName, Any]
    nodes_scanned: Mapping[DomainName, int]
    edges_scanned: Mapping[DomainName, int]
    selected_regions: tuple[Any, ...]
    paper_names: tuple[str, ...]
    chunks: tuple[EvidenceChunk, ...]
    threshold_version: str
    exhaustive: bool

    @classmethod
    def empty(cls, query: str) -> "ResearchEvidenceBundle":
        zero_counts = MappingProxyType({domain: 0 for domain in DOMAIN_NAMES})
        return cls(
            query=query,
            snapshot_fingerprints=MappingProxyType({}),
            nodes_scanned=zero_counts,
            edges_scanned=zero_counts,
            selected_regions=(),
            paper_names=(),
            chunks=(),
            threshold_version="unscored",
            exhaustive=False,
        )


@dataclass(frozen=True, slots=True)
class SnapshotCoverageSummary:
    file_count: int
    total_bytes: int
    latest_mtime_ns: int


@dataclass(frozen=True, slots=True)
class ResearchCoverageSummary:
    snapshot_fingerprints: Mapping[DomainName, SnapshotCoverageSummary]
    nodes_scanned: Mapping[DomainName, int]
    edges_scanned: Mapping[DomainName, int]
    selected_nodes: Mapping[DomainName, int]
    selected_edges: Mapping[DomainName, int]
    paper_count: int
    chunk_count: int
    threshold_version: str
    exhaustive: bool


def summarize_coverage(
    bundle: ResearchEvidenceBundle,
) -> ResearchCoverageSummary:
    selected_nodes = {domain: set() for domain in DOMAIN_NAMES}
    selected_edges = {domain: set() for domain in DOMAIN_NAMES}
    for region in bundle.selected_regions:
        domain = region.domain
        selected_nodes[domain].update(region.node_ids)
        selected_edges[domain].update(region.edge_keys)

    fingerprints = {}
    for domain, fingerprint in bundle.snapshot_fingerprints.items():
        files = tuple(getattr(fingerprint, "files", ()))
        fingerprints[domain] = SnapshotCoverageSummary(
            file_count=len(files),
            total_bytes=sum(file.size for file in files),
            latest_mtime_ns=max(
                (file.mtime_ns for file in files),
                default=0,
            ),
        )
    return ResearchCoverageSummary(
        snapshot_fingerprints=MappingProxyType(fingerprints),
        nodes_scanned=MappingProxyType(dict(bundle.nodes_scanned)),
        edges_scanned=MappingProxyType(dict(bundle.edges_scanned)),
        selected_nodes=MappingProxyType({
            domain: len(nodes) for domain, nodes in selected_nodes.items()
        }),
        selected_edges=MappingProxyType({
            domain: len(edges) for domain, edges in selected_edges.items()
        }),
        paper_count=len(bundle.paper_names),
        chunk_count=len(bundle.chunks),
        threshold_version=bundle.threshold_version,
        exhaustive=bundle.exhaustive,
    )


@dataclass(frozen=True, slots=True)
class UsageRecord:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    total_cost: float | None
    generation_id: str | None = None
    reasoning_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True, slots=True)
class CostSummary:
    actual_cost_usd: float
    cost_status: Literal["actual", "calculated", "cost_pending"]
    by_stage: Mapping[str, float]
    pending_generation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    sequence: int
    name: str
    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "name": self.name,
            "data": dict(self.data),
        }

from types import MappingProxyType

import networkx as nx
import numpy as np
import pytest

from mira.exhaustive.evidence import EvidenceCollector, EvidenceIntegrityError
from mira.exhaustive.scoring import ScoringResult
from mira.exhaustive.snapshots import DomainSnapshot
from mira.exhaustive.types import (
    ScoredNode,
    SelectedRegion,
    SnapshotFingerprint,
)


def _readonly_matrix(rows):
    matrix = np.zeros((rows, 2), dtype=np.float32)
    matrix.flags.writeable = False
    return matrix


def _snapshot(
    domain,
    graph,
    text_chunks,
    entity_chunks,
    relation_chunks=None,
):
    frozen = nx.freeze(graph)
    names = tuple(str(node) for node in frozen.nodes)
    return DomainSnapshot(
        domain=domain,
        fingerprint=SnapshotFingerprint(()),
        graph=frozen,
        entity_names=names,
        entity_matrix=_readonly_matrix(len(names)),
        relation_keys=tuple((relation_chunks or {}).keys()),
        relation_matrix=_readonly_matrix(len(relation_chunks or {})),
        text_chunks=MappingProxyType(text_chunks),
        entity_chunks=MappingProxyType({
            key: tuple(value) for key, value in entity_chunks.items()
        }),
        relation_chunks=MappingProxyType({
            key: tuple(value)
            for key, value in (relation_chunks or {}).items()
        }),
    )


@pytest.fixture()
def evidence_fixture():
    memory = nx.Graph()
    memory.add_node("Relevant Topic", entity_type="Topic")
    memory.add_node("Relevant Paper", entity_type="Paper")
    memory.add_node("Shared Author", entity_type="Author")
    memory.add_node("Unrelated Paper By Same Author", entity_type="Paper")
    memory.add_edge(
        "Relevant Topic", "Relevant Paper", keywords="primary_topic topic"
    )
    memory.add_edge(
        "Relevant Paper", "Shared Author", keywords="authored_by author"
    )
    memory.add_edge(
        "Unrelated Paper By Same Author",
        "Shared Author",
        keywords="authored_by author",
    )

    optical = nx.Graph()
    optical.add_node("Optical Topic", entity_type="Topic")
    optical.add_node("Optical Paper", entity_type="Paper")
    optical.add_edge(
        "Optical Topic", "Optical Paper", keywords="also_covers topic"
    )

    storage = nx.Graph()
    storage.add_node("Storage Topic", entity_type="Topic")
    storage.add_node("Storage Paper", entity_type="Paper")
    storage.add_edge(
        "Storage Topic", "Storage Paper", keywords="primary_topic topic"
    )

    shared_content = "A cross-domain measured result."
    snapshots = {
        "memory": _snapshot(
            "memory",
            memory,
            {
                "chunk-memory": {
                    "content": shared_content,
                    "file_path": "memory.md",
                },
                "chunk-unrelated": {
                    "content": "Unrelated author paper.",
                    "file_path": "unrelated.md",
                },
            },
            {
                "Relevant Paper": ["chunk-memory"],
                "Unrelated Paper By Same Author": ["chunk-unrelated"],
            },
            {
                "Relevant Topic<SEP>Relevant Paper": ["chunk-memory"],
            },
        ),
        "optical": _snapshot(
            "optical",
            optical,
            {
                "chunk-optical": {
                    "content": shared_content,
                    "file_path": "optical.md",
                }
            },
            {"Optical Paper": ["chunk-optical"]},
        ),
        "storage": _snapshot(
            "storage",
            storage,
            {
                "chunk-storage": {
                    "content": "A storage endurance result.",
                    "file_path": "storage.md",
                }
            },
            {"Storage Paper": ["chunk-storage"]},
        ),
    }
    regions = (
        SelectedRegion(
            "memory",
            ("Relevant Topic", "Shared Author"),
            (("Relevant Paper", "Relevant Topic"),),
        ),
        SelectedRegion("optical", ("Optical Topic",), ()),
        SelectedRegion("storage", ("Storage Topic",), ()),
    )
    scoring = ScoringResult(
        nodes=(),
        edges=(),
        regions=regions,
        nodes_by_domain=MappingProxyType({
            domain: snapshot.graph.number_of_nodes()
            for domain, snapshot in snapshots.items()
        }),
        edges_by_domain=MappingProxyType({
            domain: snapshot.graph.number_of_edges()
            for domain, snapshot in snapshots.items()
        }),
    )
    return snapshots, scoring


def test_shared_author_does_not_flood_unselected_papers(evidence_fixture):
    snapshots, scoring = evidence_fixture

    bundle = EvidenceCollector().collect("question", scoring, snapshots)

    assert "Relevant Paper" in bundle.paper_names
    assert "Unrelated Paper By Same Author" not in bundle.paper_names
    assert all(
        "Unrelated author paper." != chunk.content for chunk in bundle.chunks
    )


def test_all_qualifying_chunks_are_deduplicated_with_union_provenance(
    evidence_fixture,
):
    snapshots, scoring = evidence_fixture

    bundle = EvidenceCollector().collect("question", scoring, snapshots)

    assert len(bundle.chunks) == 2
    shared = next(
        chunk for chunk in bundle.chunks
        if chunk.content == "A cross-domain measured result."
    )
    assert shared.domains == ("memory", "optical")
    assert shared.file_paths == ("memory.md", "optical.md")
    assert shared.source_chunk_ids == ("chunk-memory", "chunk-optical")
    assert shared.paper_names == ("Optical Paper", "Relevant Paper")
    assert {
        (
            item.source_chunk_id,
            item.domain,
            item.file_path,
            item.paper_name,
        )
        for item in shared.provenance
    } == {
        ("chunk-memory", "memory", "memory.md", "Relevant Paper"),
        ("chunk-optical", "optical", "optical.md", "Optical Paper"),
    }
    assert bundle.exhaustive is True


def test_bundle_reports_exact_scanned_counts(evidence_fixture):
    snapshots, scoring = evidence_fixture

    bundle = EvidenceCollector().collect("question", scoring, snapshots)

    assert dict(bundle.nodes_scanned) == {
        "memory": 4,
        "optical": 2,
        "storage": 2,
    }
    assert dict(bundle.edges_scanned) == {
        "memory": 3,
        "optical": 1,
        "storage": 1,
    }


def test_missing_referenced_chunk_aborts_instead_of_truncating(evidence_fixture):
    snapshots, scoring = evidence_fixture
    memory = snapshots["memory"]
    broken = DomainSnapshot(
        domain=memory.domain,
        fingerprint=memory.fingerprint,
        graph=memory.graph,
        entity_names=memory.entity_names,
        entity_matrix=memory.entity_matrix,
        relation_keys=memory.relation_keys,
        relation_matrix=memory.relation_matrix,
        text_chunks=memory.text_chunks,
        entity_chunks=MappingProxyType({
            **memory.entity_chunks,
            "Relevant Paper": ("chunk-memory", "chunk-missing"),
        }),
        relation_chunks=memory.relation_chunks,
    )
    snapshots = {**snapshots, "memory": broken}

    with pytest.raises(EvidenceIntegrityError, match="chunk-missing"):
        EvidenceCollector().collect("question", scoring, snapshots)


def test_incomplete_scoring_counts_cannot_be_labeled_exhaustive(
    evidence_fixture,
):
    snapshots, scoring = evidence_fixture
    incomplete = ScoringResult(
        nodes=scoring.nodes,
        edges=scoring.edges,
        regions=scoring.regions,
        nodes_by_domain=MappingProxyType({
            **scoring.nodes_by_domain,
            "memory": scoring.nodes_by_domain["memory"] - 1,
        }),
        edges_by_domain=scoring.edges_by_domain,
    )

    with pytest.raises(EvidenceIntegrityError, match="memory"):
        EvidenceCollector().collect("question", incomplete, snapshots)


def test_selected_edge_contributes_evidence_when_endpoints_are_unselected(
    evidence_fixture,
):
    snapshots, scoring = evidence_fixture
    regions = tuple(
        SelectedRegion(
            region.domain,
            (),
            region.edge_keys if region.domain == "memory" else (),
        )
        for region in scoring.regions
    )
    edge_only = ScoringResult(
        nodes=scoring.nodes,
        edges=scoring.edges,
        regions=regions,
        nodes_by_domain=scoring.nodes_by_domain,
        edges_by_domain=scoring.edges_by_domain,
    )

    bundle = EvidenceCollector().collect("question", edge_only, snapshots)

    assert "Relevant Paper" in bundle.paper_names
    assert any(
        chunk.chunk_id == "chunk-memory" for chunk in bundle.chunks
    )


def test_relation_chunks_are_unioned_across_both_index_directions(
    evidence_fixture,
):
    snapshots, scoring = evidence_fixture
    memory = snapshots["memory"]
    text_chunks = MappingProxyType({
        **memory.text_chunks,
        "chunk-reverse": {
            "content": "Reverse-index relation evidence.",
            "file_path": "reverse.md",
        },
    })
    relation_chunks = MappingProxyType({
        **memory.relation_chunks,
        "Relevant Paper<SEP>Relevant Topic": ("chunk-reverse",),
    })
    snapshots = {
        **snapshots,
        "memory": DomainSnapshot(
            domain=memory.domain,
            fingerprint=memory.fingerprint,
            graph=memory.graph,
            entity_names=memory.entity_names,
            entity_matrix=memory.entity_matrix,
            relation_keys=memory.relation_keys,
            relation_matrix=memory.relation_matrix,
            text_chunks=text_chunks,
            entity_chunks=memory.entity_chunks,
            relation_chunks=relation_chunks,
        ),
    }

    bundle = EvidenceCollector().collect("question", scoring, snapshots)

    assert {chunk.chunk_id for chunk in bundle.chunks} >= {
        "chunk-memory",
        "chunk-reverse",
    }


def _paper_cap_fixture(paper_count, chunks_per_paper):
    graph = nx.Graph()
    graph.add_node("Topic", entity_type="Topic")
    text_chunks = {}
    entity_chunks = {}
    nodes = []
    for index in range(paper_count):
        paper = f"Paper {index:02d}"
        graph.add_node(paper, entity_type="Paper")
        graph.add_edge("Topic", paper, keywords="primary_topic topic")
        chunk_ids = []
        for position in range(chunks_per_paper):
            chunk_id = f"chunk-{index:02d}-{position:02d}"
            text_chunks[chunk_id] = {
                "content": f"Result {index}-{position}",
                "file_path": f"{paper}.md",
            }
            chunk_ids.append(chunk_id)
        entity_chunks[paper] = chunk_ids
        nodes.append((paper, 1.0 - index / 100.0))

    empty = nx.Graph()
    empty.add_node("Other Topic", entity_type="Topic")
    snapshots = {
        "memory": _snapshot("memory", graph, text_chunks, entity_chunks),
        "optical": _snapshot("optical", empty.copy(), {}, {}),
        "storage": _snapshot("storage", empty.copy(), {}, {}),
    }
    scored = tuple(
        ScoredNode(
            domain="memory",
            node_id=paper,
            node_type="Paper",
            base_score=score,
            propagated_score=score,
            selected=True,
        )
        for paper, score in nodes
    )
    regions = (
        SelectedRegion("memory", ("Topic",), ()),
        SelectedRegion("optical", (), ()),
        SelectedRegion("storage", (), ()),
    )
    scoring = ScoringResult(
        nodes=scored,
        edges=(),
        regions=regions,
        nodes_by_domain=MappingProxyType({
            domain: snapshot.graph.number_of_nodes()
            for domain, snapshot in snapshots.items()
        }),
        edges_by_domain=MappingProxyType({
            domain: snapshot.graph.number_of_edges()
            for domain, snapshot in snapshots.items()
        }),
    )
    return snapshots, scoring


def test_paper_fan_out_is_capped_highest_scored_first(caplog):
    snapshots, scoring = _paper_cap_fixture(6, 1)

    bundle = EvidenceCollector(max_papers=2).collect(
        "question", scoring, snapshots
    )

    assert bundle.paper_names == ("Paper 00", "Paper 01")
    assert len(bundle.chunks) == 2
    # Coverage honesty (round-3): a capped corpus must not claim exhaustive.
    assert bundle.exhaustive is False
    assert "coverage is NOT exhaustive" in caplog.text


def test_chunks_per_paper_are_capped_and_the_cap_is_deterministic():
    snapshots, scoring = _paper_cap_fixture(1, 9)

    first = EvidenceCollector(max_chunks_per_paper=4).collect(
        "question", scoring, snapshots
    )
    second = EvidenceCollector(max_chunks_per_paper=4).collect(
        "question", scoring, snapshots
    )

    assert len(first.chunks) == 4
    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]
    assert first.exhaustive is False


def test_uncapped_expansion_keeps_every_chunk():
    snapshots, scoring = _paper_cap_fixture(3, 5)

    bundle = EvidenceCollector().collect("question", scoring, snapshots)

    assert len(bundle.paper_names) == 3
    assert len(bundle.chunks) == 15
    assert bundle.exhaustive is True


def test_chunk_attributed_to_two_papers_is_warned_about(caplog):
    graph = nx.Graph()
    graph.add_node("Topic", entity_type="Topic")
    graph.add_node("Paper A", entity_type="Paper")
    graph.add_node("Paper B", entity_type="Paper")
    graph.add_edge("Topic", "Paper A", keywords="primary_topic topic")
    graph.add_edge("Topic", "Paper B", keywords="primary_topic topic")
    empty = nx.Graph()
    empty.add_node("Other Topic", entity_type="Topic")
    snapshots = {
        "memory": _snapshot(
            "memory",
            graph,
            {"shared-chunk": {"content": "Shared", "file_path": "both.md"}},
            {"Paper A": ["shared-chunk"], "Paper B": ["shared-chunk"]},
        ),
        "optical": _snapshot("optical", empty.copy(), {}, {}),
        "storage": _snapshot("storage", empty.copy(), {}, {}),
    }
    scoring = ScoringResult(
        nodes=(),
        edges=(),
        regions=(
            SelectedRegion("memory", ("Topic",), ()),
            SelectedRegion("optical", (), ()),
            SelectedRegion("storage", (), ()),
        ),
        nodes_by_domain=MappingProxyType({
            domain: snapshot.graph.number_of_nodes()
            for domain, snapshot in snapshots.items()
        }),
        edges_by_domain=MappingProxyType({
            domain: snapshot.graph.number_of_edges()
            for domain, snapshot in snapshots.items()
        }),
    )

    with caplog.at_level("WARNING", logger="mira.exhaustive.evidence"):
        bundle = EvidenceCollector().collect("question", scoring, snapshots)

    assert "attributed to 2 papers" in caplog.text
    assert bundle.chunks[0].paper_names == ("Paper A", "Paper B")

import base64
import inspect
import json
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

import mira.exhaustive.scoring as scoring
from mira.exhaustive.evidence import EvidenceCollector
from mira.exhaustive.scoring import QueryScorer
from mira.exhaustive.snapshots import GraphSnapshotStore
from tests.test_exhaustive_snapshots import write_snapshot


def _snapshots(tmp_path, extra_irrelevant=0):
    directories = {
        domain: tmp_path / f"{domain}-{extra_irrelevant}"
        for domain in ("memory", "optical", "storage")
    }
    for domain, root in directories.items():
        write_snapshot(root, domain)
        if not extra_irrelevant:
            continue
        graph_path = root / "graph_chunk_entity_relation.graphml"
        graph = nx.read_graphml(graph_path)
        vector_path = root / "vdb_entities.json"
        vectors = json.loads(vector_path.read_text())
        matrix = np.frombuffer(
            base64.b64decode(vectors["matrix"]), dtype=np.float32
        ).reshape(len(vectors["data"]), vectors["embedding_dim"])
        new_rows = []
        for index in range(extra_irrelevant):
            name = f"{domain} unrelated {index}"
            graph.add_node(
                name, entity_type="Author", description="unrelated biography"
            )
            vectors["data"].append({
                "__id__": f"unrelated-{index}",
                "entity_name": name,
            })
            new_rows.append([0.0, 1.0])
        matrix = np.vstack([matrix, np.asarray(new_rows, dtype=np.float32)])
        vectors["matrix"] = base64.b64encode(matrix.tobytes()).decode("ascii")
        nx.write_graphml(graph, graph_path)
        vector_path.write_text(json.dumps(vectors))
    return GraphSnapshotStore(directories).pin_all()


def _embed_query(_text):
    return np.asarray([1.0, 0.0], dtype=np.float32)


def test_every_node_and_edge_is_scored_exactly_once(tmp_path):
    snapshots = _snapshots(tmp_path)

    result = QueryScorer(_embed_query).score_all(
        "memory optical storage systems", snapshots
    )

    assert result.nodes_evaluated == 9
    assert result.edges_evaluated == 6
    assert len(result.nodes) == 9
    assert len(result.edges) == 6
    assert len({(node.domain, node.node_id) for node in result.nodes}) == 9
    assert len({
        (edge.domain, frozenset((edge.source, edge.target)))
        for edge in result.edges
    }) == 6
    assert dict(result.nodes_by_domain) == {
        "memory": 3,
        "optical": 3,
        "storage": 3,
    }
    assert dict(result.edges_by_domain) == {
        "memory": 2,
        "optical": 2,
        "storage": 2,
    }


def test_relevant_addition_never_evicts_existing_selection(tmp_path):
    before = QueryScorer(_embed_query, node_threshold=0.42).score_all(
        "systems", _snapshots(tmp_path / "before")
    )
    after = QueryScorer(_embed_query, node_threshold=0.42).score_all(
        "systems", _snapshots(tmp_path / "after", extra_irrelevant=40)
    )

    assert set(before.selected_node_keys) <= set(after.selected_node_keys)


def test_selection_is_independent_of_graph_insertion_order(tmp_path):
    snapshots = _snapshots(tmp_path)
    scorer = QueryScorer(_embed_query)
    first = scorer.score_all("systems", snapshots)
    reversed_snapshots = {}
    for domain, snapshot in snapshots.items():
        reordered = snapshot.graph.__class__()
        for node, data in reversed(list(snapshot.graph.nodes(data=True))):
            reordered.add_node(node, **data)
        for source, target, data in reversed(list(snapshot.graph.edges(data=True))):
            reordered.add_edge(source, target, **data)
        reversed_snapshots[domain] = replace(snapshot, graph=nx.freeze(reordered))

    second = scorer.score_all("systems", reversed_snapshots)

    assert first.selected_node_keys == second.selected_node_keys
    assert first.selected_edge_keys == second.selected_edge_keys


def test_scoring_is_unlimited_but_evidence_fan_out_is_capped():
    """Scoring stays exhaustive; the billable expansion does not.

    Every node and edge is still scored -- no top_k, no result truncation. The
    chunks that expansion then hands to the compiler are billed per token, so
    that step carries explicit ceilings instead of none."""
    parameters = set(inspect.signature(QueryScorer).parameters)
    parameters |= set(inspect.signature(QueryScorer.score_all).parameters)
    assert not parameters & {
        "top_k",
        "chunk_top_k",
        "max_nodes",
        "max_edges",
    }

    collector_parameters = set(
        inspect.signature(EvidenceCollector).parameters
    )
    assert {"max_papers", "max_chunks_per_paper"} <= collector_parameters
    collector = EvidenceCollector()
    assert (collector.max_papers, collector.max_chunks_per_paper) == (400, 40)
    with pytest.raises(ValueError, match="fan-out"):
        EvidenceCollector(max_papers=0)
    with pytest.raises(ValueError, match="fan-out"):
        EvidenceCollector(max_chunks_per_paper=0)


def test_long_query_is_tokenized_once_not_once_per_node(tmp_path, monkeypatch):
    snapshots = _snapshots(tmp_path)
    query = "User history " * 2_000
    original = scoring._TOKEN_RE

    class CountingPattern:
        def __init__(self):
            self.query_scans = 0

        def findall(self, value):
            if value == query or value == query.casefold():
                self.query_scans += 1
            return original.findall(value)

    counter = CountingPattern()
    monkeypatch.setattr(scoring, "_TOKEN_RE", counter)

    QueryScorer(_embed_query).score_all(query, snapshots)

    assert counter.query_scans <= 3


def test_edge_relationship_lexical_score_is_computed_once(
    tmp_path, monkeypatch
):
    snapshots = _snapshots(tmp_path)
    relationships = {
        " ".join((
            str(data.get("keywords", "")),
            str(data.get("description", "")),
        ))
        for snapshot in snapshots.values()
        for _source, _target, data in snapshot.graph.edges(data=True)
    }
    edge_count = sum(
        snapshot.graph.number_of_edges() for snapshot in snapshots.values()
    )
    original = scoring._lexical_score
    relationship_calls = 0

    def counting_score(query_tokens, value):
        nonlocal relationship_calls
        if value in relationships:
            relationship_calls += 1
        return original(query_tokens, value)

    monkeypatch.setattr(scoring, "_lexical_score", counting_score)

    QueryScorer(_embed_query).score_all("systems", snapshots)

    assert relationship_calls == edge_count

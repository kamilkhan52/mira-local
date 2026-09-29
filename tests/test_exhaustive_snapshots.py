import base64
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from mira.exhaustive.snapshots import (
    GraphSnapshotStore,
    SnapshotValidationError,
)


STORAGE_FILES = (
    "graph_chunk_entity_relation.graphml",
    "vdb_entities.json",
    "vdb_relationships.json",
    "kv_store_text_chunks.json",
    "kv_store_entity_chunks.json",
    "kv_store_relation_chunks.json",
)


def _matrix_payload(rows, matrix):
    array = np.asarray(matrix, dtype=np.float32)
    return {
        "embedding_dim": array.shape[1],
        "data": rows,
        "matrix": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def write_snapshot(root: Path, domain: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    graph = nx.Graph()
    topic = f"{domain} topic"
    paper = f"{domain} paper"
    report = f"{domain}-profile-2026-07-29"
    graph.add_node(topic, entity_type="Topic", description=f"{domain} systems")
    graph.add_node(paper, entity_type="Paper", description=f"{domain} result")
    graph.add_node(report, entity_type="Report", description=f"{domain} report")
    graph.add_edge(
        topic,
        paper,
        keywords="primary_topic topic",
        description="topic evidence",
    )
    graph.add_edge(
        paper,
        report,
        keywords="selected_in report",
        description="report evidence",
    )
    nx.write_graphml(graph, root / "graph_chunk_entity_relation.graphml")

    entity_rows = [
        {"__id__": f"ent-{i}", "entity_name": name}
        for i, name in enumerate((topic, paper, report))
    ]
    entity_matrix = [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]
    (root / "vdb_entities.json").write_text(json.dumps(
        _matrix_payload(entity_rows, entity_matrix)
    ))

    relation_rows = [
        {"__id__": "rel-0", "src_id": topic, "tgt_id": paper},
        {"__id__": "rel-1", "src_id": paper, "tgt_id": report},
    ]
    relation_matrix = [[1.0, 0.0], [0.0, 1.0]]
    (root / "vdb_relationships.json").write_text(json.dumps(
        _matrix_payload(relation_rows, relation_matrix)
    ))

    chunk_id = f"chunk-{domain}"
    (root / "kv_store_text_chunks.json").write_text(json.dumps({
        chunk_id: {
            "content": f"{domain} chunk content",
            "file_path": f"{domain}.md",
        }
    }))
    (root / "kv_store_entity_chunks.json").write_text(json.dumps({
        paper: {"chunk_ids": [chunk_id], "count": 1}
    }))
    (root / "kv_store_relation_chunks.json").write_text(json.dumps({
        f"{topic}<SEP>{paper}": {"chunk_ids": [chunk_id], "count": 1}
    }))


@pytest.fixture()
def snapshot_dirs(tmp_path):
    directories = {
        domain: tmp_path / domain
        for domain in ("memory", "optical", "storage")
    }
    for domain, root in directories.items():
        write_snapshot(root, domain)
    return directories


def test_pin_all_loads_every_domain_and_exact_counts(snapshot_dirs):
    store = GraphSnapshotStore(snapshot_dirs)

    pinned = store.pin_all()

    assert tuple(pinned) == ("memory", "optical", "storage")
    assert {d: s.graph.number_of_nodes() for d, s in pinned.items()} == {
        "memory": 3,
        "optical": 3,
        "storage": 3,
    }
    assert {d: s.graph.number_of_edges() for d, s in pinned.items()} == {
        "memory": 2,
        "optical": 2,
        "storage": 2,
    }
    assert all(snapshot.entity_matrix.flags.writeable is False
               for snapshot in pinned.values())


def test_unchanged_snapshots_are_reused_by_identity(snapshot_dirs):
    store = GraphSnapshotStore(snapshot_dirs)
    first = store.pin_all()

    second = store.pin_all()

    assert all(first[domain] is second[domain] for domain in first)


def test_changed_snapshot_reloads_only_changed_domain(snapshot_dirs):
    store = GraphSnapshotStore(snapshot_dirs)
    first = store.pin_all()
    graph_path = (
        snapshot_dirs["memory"] / "graph_chunk_entity_relation.graphml"
    )
    stat = graph_path.stat()
    os.utime(graph_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    second = store.pin_all()

    assert second["memory"] is not first["memory"]
    assert second["optical"] is first["optical"]
    assert second["storage"] is first["storage"]


def test_invalid_changed_snapshot_fails_without_replacing_cache(snapshot_dirs):
    store = GraphSnapshotStore(snapshot_dirs)
    first = store.pin_all()["memory"]
    vector_path = snapshot_dirs["memory"] / "vdb_entities.json"
    payload = json.loads(vector_path.read_text())
    payload["matrix"] = base64.b64encode(
        np.asarray([[1.0, 0.0]], dtype=np.float32).tobytes()
    ).decode("ascii")
    vector_path.write_text(json.dumps(payload))

    with pytest.raises(SnapshotValidationError, match="memory"):
        store.pin_all()

    assert store.cached("memory") is first


def test_missing_referenced_chunk_rejects_snapshot(snapshot_dirs):
    broken = snapshot_dirs["storage"] / "kv_store_entity_chunks.json"
    payload = json.loads(broken.read_text())
    only_record = next(iter(payload.values()))
    only_record["chunk_ids"].append("chunk-missing")
    broken.write_text(json.dumps(payload))

    with pytest.raises(SnapshotValidationError, match="chunk-missing"):
        GraphSnapshotStore(snapshot_dirs).pin_all()


def test_missing_domain_file_rejects_the_complete_pin(snapshot_dirs):
    (snapshot_dirs["optical"] / STORAGE_FILES[0]).unlink()
    store = GraphSnapshotStore(snapshot_dirs)

    with pytest.raises(SnapshotValidationError, match="optical"):
        store.pin_all()

    assert store.cached("memory") is None
    assert store.cached("optical") is None
    assert store.cached("storage") is None


def test_snapshot_change_during_load_is_rejected(snapshot_dirs, monkeypatch):
    import mira.exhaustive.snapshots as snapshots_module

    original = snapshots_module._load_snapshot
    changed = False

    def mutating_load(domain, fingerprint):
        nonlocal changed
        snapshot = original(domain, fingerprint)
        if domain == "memory" and not changed:
            changed = True
            path = snapshot_dirs["memory"] / "kv_store_text_chunks.json"
            path.write_text(path.read_text() + " ")
        return snapshot

    monkeypatch.setattr(snapshots_module, "_load_snapshot", mutating_load)

    with pytest.raises(SnapshotValidationError, match="changed during load"):
        GraphSnapshotStore(snapshot_dirs).pin_all()


def test_cache_hit_is_refingerprinted_before_the_pin_is_returned(
    snapshot_dirs, monkeypatch
):
    """A reused snapshot is checked again after the slow loads finish.

    Fingerprinting all three domains up front and then loading one of them can
    take minutes; without the recheck, a graph edited in that window would be
    pinned as if it still matched disk."""
    import mira.exhaustive.snapshots as snapshots_module

    store = GraphSnapshotStore(snapshot_dirs)
    store.pin_all()
    optical_graph = (
        snapshot_dirs["optical"] / "graph_chunk_entity_relation.graphml"
    )
    memory_graph = (
        snapshot_dirs["memory"] / "graph_chunk_entity_relation.graphml"
    )
    # Force one domain to reload, and change a *different*, cache-hit domain
    # while that load is in progress.
    stat = memory_graph.stat()
    os.utime(memory_graph, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    original_load = snapshots_module._load_snapshot

    def slow_load(domain, fingerprint):
        result = original_load(domain, fingerprint)
        optical_stat = optical_graph.stat()
        os.utime(
            optical_graph,
            ns=(optical_stat.st_atime_ns, optical_stat.st_mtime_ns + 5_000_000),
        )
        return result

    monkeypatch.setattr(snapshots_module, "_load_snapshot", slow_load)

    with pytest.raises(SnapshotValidationError, match="changed while pinning"):
        store.pin_all()

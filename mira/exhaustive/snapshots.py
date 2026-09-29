"""Validated immutable snapshots of the three live LightRAG graphs."""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import networkx as nx
import numpy as np

from .types import (
    DOMAIN_NAMES,
    DomainName,
    FileFingerprint,
    SnapshotFingerprint,
)


STORAGE_FILES = (
    "graph_chunk_entity_relation.graphml",
    "vdb_entities.json",
    "vdb_relationships.json",
    "kv_store_text_chunks.json",
    "kv_store_entity_chunks.json",
    "kv_store_relation_chunks.json",
)


class SnapshotValidationError(RuntimeError):
    """A domain snapshot is missing, unreadable, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class DomainSnapshot:
    domain: DomainName
    fingerprint: SnapshotFingerprint
    graph: nx.Graph
    entity_names: tuple[str, ...]
    entity_matrix: np.ndarray
    relation_keys: tuple[str, ...]
    relation_matrix: np.ndarray
    text_chunks: Mapping[str, dict]
    entity_chunks: Mapping[str, tuple[str, ...]]
    relation_chunks: Mapping[str, tuple[str, ...]]


def _fingerprint(root: Path, domain: DomainName) -> SnapshotFingerprint:
    files = []
    for filename in STORAGE_FILES:
        path = (root / filename).resolve()
        try:
            info = path.stat()
        except OSError as exc:
            raise SnapshotValidationError(
                f"{domain}: required snapshot file unavailable: {path}"
            ) from exc
        files.append(FileFingerprint(
            path=path,
            device=info.st_dev,
            inode=info.st_ino,
            size=info.st_size,
            mtime_ns=info.st_mtime_ns,
        ))
    return SnapshotFingerprint(tuple(files))


def _read_json(path: Path, domain: DomainName) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(
            f"{domain}: cannot read valid JSON from {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise SnapshotValidationError(
            f"{domain}: {path.name} must contain a JSON object"
        )
    return value


def _decode_vectors(
    payload: dict,
    *,
    row_name,
    domain: DomainName,
    filename: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    try:
        rows = payload["data"]
        dimension = int(payload["embedding_dim"])
        raw = base64.b64decode(payload["matrix"], validate=True)
        matrix = np.frombuffer(raw, dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotValidationError(
            f"{domain}: malformed {filename}"
        ) from exc
    if not isinstance(rows, list) or dimension <= 0:
        raise SnapshotValidationError(f"{domain}: malformed {filename}")
    expected = len(rows) * dimension
    if matrix.size != expected:
        raise SnapshotValidationError(
            f"{domain}: {filename} matrix has {matrix.size} floats; "
            f"expected {expected}"
        )
    try:
        names = tuple(str(row_name(row)) for row in rows)
    except (KeyError, TypeError) as exc:
        raise SnapshotValidationError(
            f"{domain}: malformed row in {filename}"
        ) from exc
    matrix = matrix.reshape(len(rows), dimension)
    matrix.flags.writeable = False
    return names, matrix


def _chunk_index(
    payload: dict, *, domain: DomainName, filename: str
) -> Mapping[str, tuple[str, ...]]:
    index = {}
    for key, record in payload.items():
        if not isinstance(record, dict) or not isinstance(
            record.get("chunk_ids", []), list
        ):
            raise SnapshotValidationError(
                f"{domain}: malformed chunk index in {filename}: {key}"
            )
        index[str(key)] = tuple(str(item) for item in record.get("chunk_ids", []))
    return MappingProxyType(index)


def _load_snapshot(
    domain: DomainName,
    fingerprint: SnapshotFingerprint,
) -> DomainSnapshot:
    paths = {item.path.name: item.path for item in fingerprint.files}
    try:
        graph = nx.read_graphml(paths["graph_chunk_entity_relation.graphml"])
    except Exception as exc:
        raise SnapshotValidationError(
            f"{domain}: cannot load graph_chunk_entity_relation.graphml"
        ) from exc
    nx.freeze(graph)

    entity_payload = _read_json(paths["vdb_entities.json"], domain)
    entity_names, entity_matrix = _decode_vectors(
        entity_payload,
        row_name=lambda row: row["entity_name"],
        domain=domain,
        filename="vdb_entities.json",
    )
    graph_names = {str(node) for node in graph.nodes}
    vector_names = set(entity_names)
    if graph_names != vector_names:
        missing_vectors = sorted(graph_names - vector_names)[:3]
        missing_nodes = sorted(vector_names - graph_names)[:3]
        raise SnapshotValidationError(
            f"{domain}: graph/vector entity mismatch; "
            f"missing vectors={missing_vectors}, missing nodes={missing_nodes}"
        )

    relation_payload = _read_json(paths["vdb_relationships.json"], domain)
    relation_keys, relation_matrix = _decode_vectors(
        relation_payload,
        row_name=lambda row: f"{row['src_id']}<SEP>{row['tgt_id']}",
        domain=domain,
        filename="vdb_relationships.json",
    )

    text_chunks = _read_json(paths["kv_store_text_chunks.json"], domain)
    entity_chunks = _chunk_index(
        _read_json(paths["kv_store_entity_chunks.json"], domain),
        domain=domain,
        filename="kv_store_entity_chunks.json",
    )
    relation_chunks = _chunk_index(
        _read_json(paths["kv_store_relation_chunks.json"], domain),
        domain=domain,
        filename="kv_store_relation_chunks.json",
    )
    available_chunks = set(text_chunks)
    for index_name, index in (
        ("entity", entity_chunks),
        ("relation", relation_chunks),
    ):
        for owner, chunk_ids in index.items():
            for chunk_id in chunk_ids:
                if chunk_id not in available_chunks:
                    raise SnapshotValidationError(
                        f"{domain}: {index_name} index {owner!r} references "
                        f"missing chunk {chunk_id}"
                    )

    return DomainSnapshot(
        domain=domain,
        fingerprint=fingerprint,
        graph=graph,
        entity_names=entity_names,
        entity_matrix=entity_matrix,
        relation_keys=relation_keys,
        relation_matrix=relation_matrix,
        text_chunks=MappingProxyType(text_chunks),
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
    )


class GraphSnapshotStore:
    """Cache complete validated snapshots and pin all domains atomically."""

    def __init__(self, directories: Mapping[str, Path]):
        self.directories = {
            domain: Path(directories[domain])
            for domain in DOMAIN_NAMES
        }
        self._cache: dict[DomainName, DomainSnapshot] = {}
        self._lock = threading.RLock()

    def cached(self, domain: DomainName) -> DomainSnapshot | None:
        with self._lock:
            return self._cache.get(domain)

    def pin_all(self) -> Mapping[DomainName, DomainSnapshot]:
        fingerprints = {
            domain: _fingerprint(self.directories[domain], domain)
            for domain in DOMAIN_NAMES
        }
        with self._lock:
            current = dict(self._cache)

        candidates: dict[DomainName, DomainSnapshot] = {}
        reused: list[DomainName] = []
        for domain in DOMAIN_NAMES:
            cached = current.get(domain)
            fingerprint = fingerprints[domain]
            candidates[domain] = (
                cached
                if cached is not None and cached.fingerprint == fingerprint
                else _load_snapshot(domain, fingerprint)
            )
            if candidates[domain] is not cached:
                after = _fingerprint(self.directories[domain], domain)
                if after != fingerprint:
                    raise SnapshotValidationError(
                        f"{domain}: snapshot changed during load"
                    )
            else:
                reused.append(domain)

        # A cache hit was fingerprinted before the other domains were loaded,
        # which can take minutes. Re-check it here so the pin proves all three
        # graphs matched disk at the same moment, not at three moments.
        for domain in reused:
            if _fingerprint(self.directories[domain], domain) != (
                fingerprints[domain]
            ):
                raise SnapshotValidationError(
                    f"{domain}: snapshot changed while pinning"
                )

        with self._lock:
            self._cache.update(candidates)
            return MappingProxyType({
                domain: self._cache[domain] for domain in DOMAIN_NAMES
            })

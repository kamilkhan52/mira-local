"""Builders for tiny synthetic LightRAG working dirs used by merge tests.

File formats mirror the real stores verified in
docs/superpowers/specs/2026-07-22-combined-graph-chatbot-design.md §2.
"""
import base64
import hashlib
import json
import zlib
from pathlib import Path

import networkx as nx
import numpy as np

DIM = 8  # small dim keeps fixtures readable; merge code must not hardcode 1536
SEP = "<SEP>"


def det_vector(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic pseudo-embedding: seeded by md5 of text, L2-normalized."""
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def fake_embed(texts: list[str]) -> np.ndarray:
    return np.stack([det_vector(t) for t in texts])


def _row_vector(vec: np.ndarray) -> str:
    # Per-row vectors are float16 in the real stores; the matrix is float32.
    return base64.b64encode(zlib.compress(vec.astype(np.float16).tobytes())).decode()


def _write_vdb(path: Path, rows: list[dict], vectors: list[np.ndarray]) -> None:
    data = []
    for row, vec in zip(rows, vectors):
        r = dict(row)
        r["vector"] = _row_vector(vec)
        data.append(r)
    matrix = base64.b64encode(np.stack(vectors).astype(np.float32).tobytes()).decode()
    path.write_text(json.dumps({"embedding_dim": DIM, "data": data, "matrix": matrix}))


def ent_id(name: str) -> str:
    return "ent-" + hashlib.md5(name.encode()).hexdigest()


def rel_id(src: str, tgt: str) -> str:
    return "rel-" + hashlib.md5(f"{src}{tgt}".encode()).hexdigest()


def chunk_id(content: str) -> str:
    return "chunk-" + hashlib.md5(content.encode()).hexdigest()


DEFAULT_CREATED_AT = 1700000000


def build_working_dir(path: Path, spec: dict) -> None:
    """spec = {"entities": [{name, entity_type, description, chunk_ids}],
               "relations": [{src, tgt, description, keywords, weight, chunk_ids}],
               "chunks": {chunk_id: content},
               "created_at": int (optional, defaults to DEFAULT_CREATED_AT)}

    `created_at` is a per-working-dir timestamp, not a per-entity one: real
    LightRAG dirs are built in one ingest run, so every node, edge and store row
    in a dir shares it. It is a spec field rather than a hardcoded constant
    because the merge's `created_at`-min rule is only testable when the two dirs
    carry *different* timestamps — with both sides on the same value, `min` and
    a last-wins overwrite are indistinguishable.
    """
    path.mkdir(parents=True, exist_ok=True)
    created_at = spec.get("created_at", DEFAULT_CREATED_AT)

    g = nx.Graph()
    for e in spec["entities"]:
        g.add_node(e["name"], entity_id=e["name"], entity_type=e["entity_type"],
                   description=e["description"], source_id=SEP.join(e["chunk_ids"]),
                   file_path="custom_kg", created_at=created_at)
    for r in spec["relations"]:
        g.add_edge(r["src"], r["tgt"], weight=r["weight"], description=r["description"],
                   keywords=r["keywords"], source_id=SEP.join(r["chunk_ids"]),
                   file_path="custom_kg", created_at=created_at)
    nx.write_graphml(g, path / "graph_chunk_entity_relation.graphml")

    text_chunks = {cid: {"content": content, "source_id": cid, "tokens": 10,
                         "chunk_order_index": 0, "full_doc_id": "doc-1",
                         "file_path": "custom_kg", "status": "processed",
                         "llm_cache_list": [], "create_time": created_at,
                         "update_time": created_at, "_id": cid}
                   for cid, content in spec["chunks"].items()}
    (path / "kv_store_text_chunks.json").write_text(json.dumps(text_chunks))

    entity_chunks = {e["name"]: {"chunk_ids": list(e["chunk_ids"]),
                                 "count": len(e["chunk_ids"]), "create_time": created_at}
                     for e in spec["entities"]}
    (path / "kv_store_entity_chunks.json").write_text(json.dumps(entity_chunks))

    relation_chunks = {f'{r["src"]}{SEP}{r["tgt"]}': {"chunk_ids": list(r["chunk_ids"]),
                       "count": len(r["chunk_ids"]), "create_time": created_at,
                       "update_time": created_at, "_id": rel_id(r["src"], r["tgt"])}
                       for r in spec["relations"]}
    (path / "kv_store_relation_chunks.json").write_text(json.dumps(relation_chunks))

    (path / "kv_store_llm_response_cache.json").write_text(json.dumps({"stale": "cache"}))

    ent_rows, ent_vecs = [], []
    for e in spec["entities"]:
        content = f'{e["name"]}\n{e["description"]}'
        ent_rows.append({"__id__": ent_id(e["name"]), "__created_at__": created_at,
                         "content": content, "entity_name": e["name"],
                         "source_id": "UNKNOWN", "file_path": "custom_kg"})
        ent_vecs.append(det_vector(content))
    _write_vdb(path / "vdb_entities.json", ent_rows, ent_vecs)

    rel_rows, rel_vecs = [], []
    for r in spec["relations"]:
        content = f'{r["keywords"]}\t{r["src"]}\n{r["tgt"]}\n{r["description"]}'
        rel_rows.append({"__id__": rel_id(r["src"], r["tgt"]), "__created_at__": created_at,
                         "src_id": r["src"], "tgt_id": r["tgt"],
                         "source_id": "UNKNOWN", "content": content, "file_path": "custom_kg"})
        rel_vecs.append(det_vector(content))
    _write_vdb(path / "vdb_relationships.json", rel_rows, rel_vecs)

    ch_rows, ch_vecs = [], []
    for cid, content in spec["chunks"].items():
        ch_rows.append({"__id__": cid, "__created_at__": created_at, "content": content,
                        "full_doc_id": "doc-1", "file_path": "custom_kg"})
        ch_vecs.append(det_vector(content))
    _write_vdb(path / "vdb_chunks.json", ch_rows, ch_vecs)


# Shared across both: entity "Shared Topic", relation Shared Topic--Common Org,
# chunk CH_SHARED. Everything else is domain-unique.
CH_SHARED = chunk_id("shared chunk text")
CH_MEM = chunk_id("memory-only chunk")
CH_OPT = chunk_id("optical-only chunk")

FIXTURE_MEMORY = {
    # Older than the optical dir on purpose: created_at is merged with `min`, and
    # memory is folded in first, so equal timestamps would make `min` and a
    # last-wins overwrite (the nx.compose regression) produce the same answer.
    # Memory strictly earlier => min gives 1700000000, last-wins gives 1700000001.
    "created_at": 1700000000,
    "entities": [
        # entity_type on the two shared entities exercises both halves of the
        # node rule (memory wins unless empty, then optical fills it in), and
        # both halves need the two sides to *disagree* to be observable:
        #   "Shared Topic": memory "topic"  vs optical "concept"      -> "topic"
        #   "Common Org":   memory ""       vs optical "organization" -> "organization"
        {"name": "Shared Topic", "entity_type": "topic",
         "description": "memory view of shared topic", "chunk_ids": [CH_SHARED, CH_MEM]},
        {"name": "Common Org", "entity_type": "",
         "description": "org seen from memory", "chunk_ids": [CH_MEM]},
        {"name": "Mem Entity", "entity_type": "topic",
         "description": "hbm memory wall", "chunk_ids": [CH_MEM]},
    ],
    "relations": [
        # weight 3.0 on the memory side and 2.0 on the optical side is deliberate:
        # optical is folded in second, so with the weights the other way round a
        # plain last-wins overwrite (the nx.compose regression) would also yield
        # 3.0 and the max-rule assertion could not tell the two apart.
        {"src": "Shared Topic", "tgt": "Common Org", "description": "mem rel desc",
         "keywords": "collaboration", "weight": 3.0, "chunk_ids": [CH_MEM]},
        {"src": "Shared Topic", "tgt": "Mem Entity", "description": "mem-only rel",
         "keywords": "memory", "weight": 1.0, "chunk_ids": [CH_MEM]},
    ],
    "chunks": {CH_SHARED: "shared chunk text", CH_MEM: "memory-only chunk"},
}

FIXTURE_OPTICAL = {
    "created_at": 1700000001,  # strictly later than memory's; see FIXTURE_MEMORY
    "entities": [
        {"name": "Shared Topic", "entity_type": "concept",
         "description": "optical view of shared topic", "chunk_ids": [CH_SHARED, CH_OPT]},
        {"name": "Common Org", "entity_type": "organization",
         "description": "org seen from optical", "chunk_ids": [CH_OPT]},
        {"name": "Opt Entity", "entity_type": "topic",
         "description": "co-packaged optics", "chunk_ids": [CH_OPT]},
    ],
    "relations": [
        {"src": "Shared Topic", "tgt": "Common Org", "description": "opt rel desc",
         "keywords": "photonics", "weight": 2.0, "chunk_ids": [CH_OPT]},
        {"src": "Shared Topic", "tgt": "Opt Entity", "description": "opt-only rel",
         "keywords": "optics", "weight": 1.0, "chunk_ids": [CH_OPT]},
    ],
    "chunks": {CH_SHARED: "shared chunk text", CH_OPT: "optical-only chunk"},
}

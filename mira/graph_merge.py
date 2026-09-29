"""Mechanical merge of two LightRAG working dirs into a combined one.

Storage formats documented and verified in
docs/superpowers/specs/2026-07-22-combined-graph-chatbot-design.md §2.
"""
from __future__ import annotations

import base64
import json
import shutil
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np

GRAPH_FIELD_SEP = "<SEP>"

STORAGE_FILES = [
    "graph_chunk_entity_relation.graphml",
    "vdb_entities.json", "vdb_relationships.json", "vdb_chunks.json",
    "kv_store_text_chunks.json", "kv_store_entity_chunks.json",
    "kv_store_relation_chunks.json", "kv_store_llm_response_cache.json",
]


class MergeError(Exception):
    pass


def encode_row_vector(vec: np.ndarray) -> str:
    """Per-row 'vector' field: base64(zlib(float16)). NOT the same encoding as
    the store-level 'matrix' field, which is plain base64(float32)."""
    return base64.b64encode(zlib.compress(vec.astype(np.float16).tobytes())).decode()


def decode_row_vector(s: str) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(s))
    return np.frombuffer(raw, dtype=np.float16).astype(np.float32)


class VectorStore:
    def __init__(self, embedding_dim: int, data: list[dict], matrix: np.ndarray):
        self.embedding_dim = embedding_dim
        self.data = data
        self.matrix = matrix

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        raw = json.loads(Path(path).read_text())
        dim = raw["embedding_dim"]
        matrix = np.frombuffer(base64.b64decode(raw["matrix"]),
                               dtype=np.float32).reshape(-1, dim).copy()
        if len(raw["data"]) != matrix.shape[0]:
            raise MergeError(f"{path}: matrix rows != data rows")
        # Self-check against format drift: per-row vector must match matrix row.
        if raw["data"]:
            probe = decode_row_vector(raw["data"][0]["vector"])
            # atol, not rtol: the row vector is float16, the matrix float32.
            if probe.shape != (dim,) or not np.allclose(probe, matrix[0], atol=1e-3):
                raise MergeError(f"{path}: row-vector/matrix mismatch — format drift?")
        return cls(dim, raw["data"], matrix)

    def save(self, path: Path) -> None:
        raw = {"embedding_dim": self.embedding_dim, "data": self.data,
               "matrix": base64.b64encode(
                   self.matrix.astype(np.float32).tobytes()).decode()}
        Path(path).write_text(json.dumps(raw, ensure_ascii=False))


@dataclass
class WorkingDir:
    path: Path
    graph: nx.Graph
    vdb_entities: VectorStore
    vdb_relationships: VectorStore
    vdb_chunks: VectorStore
    text_chunks: dict
    entity_chunks: dict
    relation_chunks: dict


def load_working_dir(path: Path) -> WorkingDir:
    path = Path(path)
    missing = [f for f in STORAGE_FILES if not (path / f).exists()]
    if missing:
        raise MergeError(f"{path}: missing storage files: {missing}")
    # Unknown stores must abort: the merge is destructive by omission, so a
    # store we have no merge rule for would be silently dropped.
    unexpected = [p.name for p in path.iterdir()
                  if (p.name.startswith("kv_store_") or p.name.startswith("vdb_"))
                  and p.name.endswith(".json") and p.name not in STORAGE_FILES]
    if unexpected:
        raise MergeError(
            f"{path}: unexpected storage files (merge rules unknown): {unexpected}")

    def _kv(name):
        return json.loads((path / name).read_text())

    wd = WorkingDir(
        path=path,
        graph=nx.read_graphml(path / "graph_chunk_entity_relation.graphml"),
        vdb_entities=VectorStore.load(path / "vdb_entities.json"),
        vdb_relationships=VectorStore.load(path / "vdb_relationships.json"),
        vdb_chunks=VectorStore.load(path / "vdb_chunks.json"),
        text_chunks=_kv("kv_store_text_chunks.json"),
        entity_chunks=_kv("kv_store_entity_chunks.json"),
        relation_chunks=_kv("kv_store_relation_chunks.json"),
    )
    dims = {wd.vdb_entities.embedding_dim, wd.vdb_relationships.embedding_dim,
            wd.vdb_chunks.embedding_dim}
    if len(dims) != 1:
        raise MergeError(f"{path}: inconsistent embedding dims: {dims}")
    return wd


def sep_union(*values) -> str:
    """Union of <SEP>-joined multi-valued attributes, first-seen order, deduped."""
    parts: list[str] = []
    for v in values:
        if not v:
            continue
        for p in str(v).split(GRAPH_FIELD_SEP):
            if p and p not in parts:
                parts.append(p)
    return GRAPH_FIELD_SEP.join(parts)


_UNION_ATTRS = ("description", "source_id", "file_path")
_EDGE_UNION_ATTRS = _UNION_ATTRS + ("keywords",)


def _fold_union_attrs(target: dict, incoming: dict,
                      union_keys: tuple[str, ...]) -> None:
    """Union each multi-valued attribute of `incoming` into `target`, in place.

    Assigns unconditionally, so a shared entity gains the key with value "" even
    when neither side carries it — preserved deliberately; see merge_graphs.
    """
    for key in union_keys:
        target[key] = sep_union(target.get(key), incoming.get(key))


def _fold_created_at(target: dict, incoming: dict) -> None:
    """Keep the earliest creation time, in place.

    The default matters: `target`'s own value is what wins when both sides carry
    one, so a missing `target` value must fall back to `incoming`'s rather than
    to 0 or now() — either of which would pin every merge to a bogus minimum.
    """
    if "created_at" in incoming:
        target["created_at"] = min(target.get("created_at", incoming["created_at"]),
                                   incoming["created_at"])


def merge_graphs(g_mem: nx.Graph, g_opt: nx.Graph
                 ) -> tuple[nx.Graph, set[str], set[frozenset]]:
    """Union the two graphs, merging attributes of entities/relations in both.

    Returns (merged graph, shared node names, shared edges as frozensets).

    NOT nx.compose: for a node in both graphs compose keeps only the second
    graph's attribute values, silently dropping the first graph's description,
    source_id and file_path. Shared entities are exactly the cross-domain
    bridges this merge exists to create, so losing one side's attributes would
    gut the result while still looking like a successful merge.
    """
    shared_nodes = set(g_mem.nodes) & set(g_opt.nodes)
    shared_edges = ({frozenset(e) for e in g_mem.edges}
                    & {frozenset(e) for e in g_opt.edges})
    g = nx.Graph()
    for n, attrs in g_mem.nodes(data=True):
        g.add_node(n, **attrs)
    for n, attrs in g_opt.nodes(data=True):
        if n not in g:
            g.add_node(n, **attrs)
            continue
        a = g.nodes[n]
        _fold_union_attrs(a, attrs, _UNION_ATTRS)
        if not a.get("entity_type"):
            a["entity_type"] = attrs.get("entity_type", "")
        _fold_created_at(a, attrs)
    for u, v, attrs in g_mem.edges(data=True):
        g.add_edge(u, v, **attrs)
    for u, v, attrs in g_opt.edges(data=True):
        if not g.has_edge(u, v):
            g.add_edge(u, v, **attrs)
            continue
        a = g.edges[u, v]
        _fold_union_attrs(a, attrs, _EDGE_UNION_ATTRS)
        a["weight"] = max(float(a.get("weight", 0)), float(attrs.get("weight", 0)))
        _fold_created_at(a, attrs)
    return g, shared_nodes, shared_edges


def merge_text_chunks(mem: dict, opt: dict) -> dict:
    """Union the two chunk stores, keyed by chunk id.

    Chunk ids are content hashes (`chunk-<md5 of content>`), so identical text in
    both dirs dedupes for free. That assumption is asserted rather than trusted:
    a shared id carrying different text means the hash assumption is broken and
    the merge must abort instead of silently picking a side. Only `content` is
    compared — bookkeeping fields (create_time, update_time) legitimately differ
    between two independently ingested dirs.
    """
    out = dict(mem)
    for k, v in opt.items():
        if k in out:
            if out[k]["content"] != v["content"]:
                raise MergeError(
                    f"text chunk id collision with different content: {k}")
        else:
            out[k] = v
    return out


def merge_chunk_lists(mem: dict, opt: dict) -> dict:
    """Union the chunk-id lists of entity_chunks or relation_chunks.

    Keys are entity names, or "<src><SEP><tgt>" for relations. For a key in both
    dirs the lists are unioned with memory's ids first and `count` recomputed:
    keeping one side's list would leave a shared entity unable to cite the other
    domain's papers, which retrieval surfaces as missing sources, not as an error.
    """
    out = {k: dict(v) for k, v in mem.items()}
    for k, v in opt.items():
        if k not in out:
            out[k] = dict(v)
            continue
        ids = list(out[k]["chunk_ids"])
        ids += [c for c in v["chunk_ids"] if c not in ids]
        out[k]["chunk_ids"] = ids
        out[k]["count"] = len(ids)
    return out


def _concat_dedupe(mem: VectorStore, opt: VectorStore):
    """Concat rows keeping memory's row on __id__ collision.

    Returns (data, matrix, colliding_ids). `data` and `matrix` are built from the
    same index sequence, which is the invariant the whole file rests on: matrix
    row i must describe data[i], and a mismatch here would misroute every search
    hit without ever raising.
    """
    if mem.embedding_dim != opt.embedding_dim:
        raise MergeError(
            f"embedding dim mismatch between sources: "
            f"{mem.embedding_dim} vs {opt.embedding_dim}")
    mem_ids = {r["__id__"] for r in mem.data}
    colliding = [r["__id__"] for r in opt.data if r["__id__"] in mem_ids]
    keep = [i for i, r in enumerate(opt.data) if r["__id__"] not in mem_ids]
    data = [dict(r) for r in mem.data] + [dict(opt.data[i]) for i in keep]
    matrix = np.vstack([mem.matrix, opt.matrix[keep]]) if keep else mem.matrix.copy()
    return data, matrix, set(colliding)


def merge_vdb_chunks(mem: VectorStore, opt: VectorStore) -> VectorStore:
    """Union the chunk vectors, keeping memory's row when ids collide.

    Chunk ids are `chunk-<md5 of content>`, so a colliding id provably carries
    identical text and its embedding is identical too — nothing to re-embed.
    """
    data, matrix, _ = _concat_dedupe(mem, opt)
    return VectorStore(mem.embedding_dim, data, matrix)


def _reembed(store: VectorStore, indices: list[int], texts: list[str], embed_fn) -> None:
    """Rewrite the given rows from `texts`, in place, in one embedder batch.

    Both encodings of a vector are written together: the row's own
    base64(zlib(float16)) `vector` field and the float32 `matrix` row at the same
    index. Updating one and not the other leaves the store self-inconsistent in a
    way that only shows up as wrong search results.
    """
    if not indices:
        return
    vecs = np.asarray(embed_fn(texts), dtype=np.float32)
    if vecs.shape != (len(texts), store.embedding_dim):
        raise MergeError(f"embed_fn returned shape {vecs.shape}, "
                         f"expected {(len(texts), store.embedding_dim)}")
    # Normalization is the merge's job, not the embedder's: the stores hold unit
    # vectors so retrieval's dot product is a cosine similarity.
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    for i, text, vec in zip(indices, texts, vecs):
        store.data[i]["content"] = text
        store.data[i]["vector"] = encode_row_vector(vec)
        store.matrix[i] = vec


def merge_vdb_entities(mem: VectorStore, opt: VectorStore, merged_graph: nx.Graph,
                       embed_fn) -> VectorStore:
    """Union the entity vectors, re-embedding entities present in both graphs.

    A colliding id means the entity exists in both dirs, so merge_graphs joined
    the two descriptions into text neither stored embedding encodes. Copying
    either side's vector would leave that entity findable only from its own
    domain — the opposite of what the merge is for.
    """
    data, matrix, colliding = _concat_dedupe(mem, opt)
    store = VectorStore(mem.embedding_dim, data, matrix)
    indices, texts = [], []
    for i, row in enumerate(store.data):
        if row["__id__"] in colliding:
            name = row["entity_name"]
            if name not in merged_graph.nodes:
                raise MergeError(f"shared entity missing from merged graph: {name}")
            desc = merged_graph.nodes[name].get("description", "")
            indices.append(i)
            texts.append(f"{name}\n{desc}")
    _reembed(store, indices, texts, embed_fn)
    return store


def merge_vdb_relationships(mem: VectorStore, opt: VectorStore,
                            merged_graph: nx.Graph, embed_fn) -> VectorStore:
    """Union the relation vectors, re-embedding relations present in both graphs.

    Same reasoning as merge_vdb_entities; the embed text mirrors what LightRAG
    stores for a relation: keywords, endpoints, then description.
    """
    data, matrix, colliding = _concat_dedupe(mem, opt)
    store = VectorStore(mem.embedding_dim, data, matrix)
    indices, texts = [], []
    for i, row in enumerate(store.data):
        if row["__id__"] in colliding:
            src, tgt = row["src_id"], row["tgt_id"]
            if not merged_graph.has_edge(src, tgt):
                raise MergeError(f"shared relation missing from merged graph: "
                                 f"{src} -- {tgt}")
            e = merged_graph.edges[src, tgt]
            indices.append(i)
            texts.append(f'{e.get("keywords", "")}\t{src}\n{tgt}\n'
                         f'{e.get("description", "")}')
    _reembed(store, indices, texts, embed_fn)
    return store


def _domain(key, in_mem: bool, in_opt: bool) -> str:
    return "both" if (in_mem and in_opt) else ("memory" if in_mem else "optical")


def build_provenance(mem: WorkingDir, opt: WorkingDir, generated_at: str) -> dict:
    """Record, for every entity/chunk/relation, whether it came from memory,
    optical, or both. Powers the chatbot's domain badges."""
    entities = {}
    for name in set(mem.graph.nodes) | set(opt.graph.nodes):
        entities[name] = _domain(name, name in mem.graph.nodes,
                                 name in opt.graph.nodes)
    chunks = {}
    for cid in set(mem.text_chunks) | set(opt.text_chunks):
        chunks[cid] = _domain(cid, cid in mem.text_chunks, cid in opt.text_chunks)
    relations = {}
    for key in set(mem.relation_chunks) | set(opt.relation_chunks):
        relations[key] = _domain(key, key in mem.relation_chunks,
                                 key in opt.relation_chunks)
    return {"generated_at": generated_at, "entities": entities,
            "chunks": chunks, "relations": relations}


def build_provenance_multi(sources: dict[str, WorkingDir], generated_at: str) -> dict:
    """Record each key's sorted set of original source domains.

    Unlike the legacy pairwise writer, shared keys use labels such as
    ``memory+optical`` rather than ``both`` so this works for any number of
    original working directories.
    """
    def labels(keys_by_domain: dict[str, set]) -> dict:
        keys = set().union(*keys_by_domain.values()) if keys_by_domain else set()
        return {
            key: "+".join(sorted(
                domain for domain, source_keys in keys_by_domain.items()
                if key in source_keys))
            for key in keys
        }

    return {
        "generated_at": generated_at,
        "entities": labels({domain: set(wd.graph.nodes)
                             for domain, wd in sources.items()}),
        "chunks": labels({domain: set(wd.text_chunks)
                           for domain, wd in sources.items()}),
        "relations": labels({domain: set(wd.relation_chunks)
                              for domain, wd in sources.items()}),
    }


def verify_merge(mem: WorkingDir, opt: WorkingDir, combined: WorkingDir) -> None:
    """Abort-before-swap gate: collect EVERY failed invariant into one MergeError
    rather than stopping at the first, so a failed run reports everything wrong."""
    errors: list[str] = []

    def expect(actual, expected, label):
        if actual != expected:
            errors.append(f"{label}: got {actual}, expected {expected}")

    shared_nodes = set(mem.graph.nodes) & set(opt.graph.nodes)
    shared_edges = ({frozenset(e) for e in mem.graph.edges}
                    & {frozenset(e) for e in opt.graph.edges})
    expect(combined.graph.number_of_nodes(),
           mem.graph.number_of_nodes() + opt.graph.number_of_nodes()
           - len(shared_nodes), "graph nodes")
    expect(combined.graph.number_of_edges(),
           mem.graph.number_of_edges() + opt.graph.number_of_edges()
           - len(shared_edges), "graph edges")

    for label, get in [("vdb_entities", lambda w: w.vdb_entities),
                       ("vdb_relationships", lambda w: w.vdb_relationships),
                       ("vdb_chunks", lambda w: w.vdb_chunks)]:
        m, o, c = get(mem), get(opt), get(combined)
        shared = {r["__id__"] for r in m.data} & {r["__id__"] for r in o.data}
        expect(len(c.data), len(m.data) + len(o.data) - len(shared),
               f"{label} rows")
        if c.matrix.shape[0] != len(c.data):
            errors.append(f"{label}: matrix rows != data rows")
        norms = np.linalg.norm(c.matrix, axis=1)
        if len(norms) and not np.allclose(norms, 1.0, rtol=1e-3):
            errors.append(f"{label}: vectors not L2-normalized")

    for label, m, o, c in [
        ("text_chunks", mem.text_chunks, opt.text_chunks, combined.text_chunks),
        ("entity_chunks", mem.entity_chunks, opt.entity_chunks,
         combined.entity_chunks),
        ("relation_chunks", mem.relation_chunks, opt.relation_chunks,
         combined.relation_chunks),
    ]:
        expect(len(c), len(set(m) | set(o)), f"{label} keys")

    graph_names = set(combined.graph.nodes)
    vdb_names = {r["entity_name"] for r in combined.vdb_entities.data}
    if graph_names != vdb_names:
        errors.append(f"graph<->vdb_entities mismatch: "
                      f"only-graph={sorted(graph_names - vdb_names)[:5]}, "
                      f"only-vdb={sorted(vdb_names - graph_names)[:5]}")

    # Dangling-chunk check is BASELINE-RELATIVE, not absolute. Live sources
    # carry legacy non-chunk tags in their chunk_id lists ('UNKNOWN' placeholders
    # and profile-run ids like 'memory-innovation-<timestamp>') that were never
    # real `chunk-<md5>` ids and have no text_chunks entry — the LightRAG servers
    # operate fine with them. A mechanical merge faithfully unions those lists, so
    # it preserves that pre-existing dangling set unchanged. Flagging the absolute
    # dangling set would abort on normal operating data; only dangling refs the
    # MERGE INTRODUCED (present in combined but in neither source's baseline)
    # indicate a real bug. Pre-existing dangling passing through is silent success.
    def _dangling(wd: WorkingDir) -> set:
        referenced = {c for v in wd.entity_chunks.values() for c in v["chunk_ids"]}
        referenced |= {c for v in wd.relation_chunks.values() for c in v["chunk_ids"]}
        return referenced - set(wd.text_chunks)

    dangling_new = _dangling(combined) - (_dangling(mem) | _dangling(opt))
    if dangling_new:
        errors.append(f"chunk ids referenced but missing from text_chunks: "
                      f"{sorted(dangling_new)[:5]}")

    if errors:
        raise MergeError("merge verification failed:\n  " + "\n  ".join(errors))


@dataclass
class MergeReport:
    nodes: int
    edges: int
    shared_entities: int
    shared_relations: int
    shared_chunks: int
    reembedded: int
    output_dir: Path | None


def run_merge(memory_dir, optical_dir, output_dir, embed_fn,
              dry_run: bool = False, now: datetime | None = None) -> MergeReport:
    """Merge two LightRAG working dirs into `output_dir`, atomically.

    Writes into a sibling temp dir, verifies a fresh re-load of it (which
    re-runs every VectorStore self-check on the bytes just written), and only
    then swaps: the existing output dir becomes `<output>.bak-<YYYY-MM-DD>`
    (at most 2 baks kept) and the temp dir is renamed into place. On
    verification failure the temp dir is left for inspection and the existing
    output dir is never touched. Sources are read-only. `now` is injectable so
    two runs with identical inputs produce byte-identical output.
    """
    now = now or datetime.now(timezone.utc)
    output_dir = Path(output_dir)
    mem = load_working_dir(memory_dir)
    opt = load_working_dir(optical_dir)

    graph, shared_nodes, shared_edges = merge_graphs(mem.graph, opt.graph)
    shared_chunks = set(mem.text_chunks) & set(opt.text_chunks)
    report = MergeReport(
        nodes=graph.number_of_nodes(), edges=graph.number_of_edges(),
        shared_entities=len(shared_nodes), shared_relations=len(shared_edges),
        shared_chunks=len(shared_chunks),
        reembedded=len(shared_nodes) + len(shared_edges),
        output_dir=None,
    )
    if dry_run:
        return report

    text_chunks = merge_text_chunks(mem.text_chunks, opt.text_chunks)
    entity_chunks = merge_chunk_lists(mem.entity_chunks, opt.entity_chunks)
    relation_chunks = merge_chunk_lists(mem.relation_chunks, opt.relation_chunks)
    vdb_entities = merge_vdb_entities(mem.vdb_entities, opt.vdb_entities,
                                      graph, embed_fn)
    vdb_relationships = merge_vdb_relationships(
        mem.vdb_relationships, opt.vdb_relationships, graph, embed_fn)
    vdb_chunks = merge_vdb_chunks(mem.vdb_chunks, opt.vdb_chunks)
    provenance = build_provenance(mem, opt, generated_at=now.isoformat())

    tmp = output_dir.parent / f"{output_dir.name}.tmp-{now.strftime('%Y%m%d%H%M%S')}"
    tmp.mkdir(parents=True)
    try:
        nx.write_graphml(graph, tmp / "graph_chunk_entity_relation.graphml")
        vdb_entities.save(tmp / "vdb_entities.json")
        vdb_relationships.save(tmp / "vdb_relationships.json")
        vdb_chunks.save(tmp / "vdb_chunks.json")
        for name, obj in [("kv_store_text_chunks.json", text_chunks),
                          ("kv_store_entity_chunks.json", entity_chunks),
                          ("kv_store_relation_chunks.json", relation_chunks),
                          ("kv_store_llm_response_cache.json", {}),
                          ("provenance.json", provenance)]:
            (tmp / name).write_text(
                json.dumps(obj, ensure_ascii=False, sort_keys=True))
        # Verify a fresh re-load of what was written, not the in-memory objects:
        # this re-runs every VectorStore row-vector/matrix self-check on the
        # actual bytes. Any failure raises before the swap below, leaving `tmp`
        # for inspection and the existing output dir untouched.
        verify_merge(mem, opt, load_working_dir(tmp))
    except Exception:
        # Leave tmp for inspection on verification/write failure, per spec.
        raise

    if output_dir.exists():
        bak = output_dir.parent / f"{output_dir.name}.bak-{now.strftime('%Y-%m-%d')}"
        if bak.exists():
            shutil.rmtree(bak)
        output_dir.rename(bak)
        baks = sorted(output_dir.parent.glob(f"{output_dir.name}.bak-*"))
        for old in baks[:-2]:
            shutil.rmtree(old)
    tmp.rename(output_dir)
    report.output_dir = output_dir
    return report

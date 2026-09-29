from pathlib import Path

import numpy as np
import pytest

from tests.lightrag_fixtures import (
    FIXTURE_MEMORY, FIXTURE_OPTICAL, build_working_dir, fake_embed,
    det_vector, CH_SHARED, CH_MEM, CH_OPT, ent_id,
)
from mira.graph_merge import (
    MergeError, VectorStore, WorkingDir, build_provenance, build_provenance_multi,
    decode_row_vector,
    load_working_dir, merge_chunk_lists, merge_graphs, merge_text_chunks,
    merge_vdb_chunks, merge_vdb_entities, merge_vdb_relationships, sep_union,
    verify_merge,
)


@pytest.fixture()
def dirs(tmp_path):
    mem, opt = tmp_path / "mem", tmp_path / "opt"
    build_working_dir(mem, FIXTURE_MEMORY)
    build_working_dir(opt, FIXTURE_OPTICAL)
    return mem, opt


def test_vectorstore_roundtrip(dirs):
    mem, _ = dirs
    vs = VectorStore.load(mem / "vdb_entities.json")
    assert vs.embedding_dim == 8
    assert vs.matrix.shape == (3, 8)
    assert len(vs.data) == 3
    # matrix row i corresponds to data[i], and the per-row compressed
    # 'vector' field decodes to the same values (float16 round-trip => atol)
    row0 = decode_row_vector(vs.data[0]["vector"])
    np.testing.assert_allclose(vs.matrix[0], row0, atol=1e-3)
    out = mem / "roundtrip.json"
    vs.save(out)
    vs2 = VectorStore.load(out)
    np.testing.assert_allclose(vs.matrix, vs2.matrix, rtol=1e-6)
    assert vs.data == vs2.data


def test_load_working_dir(dirs):
    mem, _ = dirs
    wd = load_working_dir(mem)
    assert wd.graph.number_of_nodes() == 3
    assert wd.graph.number_of_edges() == 2
    assert set(wd.text_chunks) == {CH_SHARED, CH_MEM}
    assert wd.vdb_entities.matrix.shape[0] == 3
    # read_graphml honours the file's edgedefault, so a directed graphml would
    # yield a DiGraph and make the nx.Graph annotation a lie. Every merge rule
    # downstream assumes an undirected simple graph.
    assert not wd.graph.is_directed()
    assert not wd.graph.is_multigraph()


def test_load_missing_file_aborts(dirs):
    mem, _ = dirs
    (mem / "vdb_chunks.json").unlink()
    with pytest.raises(MergeError, match="vdb_chunks.json"):
        load_working_dir(mem)


def test_load_unexpected_storage_file_aborts(dirs):
    mem, _ = dirs
    (mem / "kv_store_full_docs.json").write_text("{}")
    with pytest.raises(MergeError, match="kv_store_full_docs.json"):
        load_working_dir(mem)


def test_load_tolerates_non_storage_clutter(dirs):
    """The unexpected-store guard must reject only kv_store_*/vdb_* .json files.

    Real working dirs carry payload dumps, loader scripts and backups alongside
    the eight stores. Broadening the rule to "any .json not in STORAGE_FILES"
    would reject every real dir, starting with _bulk_payload.json.
    """
    mem, _ = dirs
    clutter = [
        "_bulk_payload.json", "_fulltext_payload.json", "_rename_plan.json.done",
        "bulk_load.py", "bulk_load_fulltext.py", ".gitkeep",
        "graph_chunk_entity_relation.graphml.bak-pre-author-cleanup",
    ]
    for name in clutter:
        (mem / name).write_text("{}" if name.endswith(".json") else "junk")

    wd = load_working_dir(mem)
    assert wd.graph.number_of_nodes() == 3
    assert set(wd.text_chunks) == {CH_SHARED, CH_MEM}


def test_sep_union_dedupes_and_preserves_order():
    assert sep_union("a<SEP>b", "b<SEP>c", None) == "a<SEP>b<SEP>c"


def test_merge_graphs_counts_and_shared_attrs(dirs):
    mem, opt = dirs
    g, shared_nodes, shared_edges = merge_graphs(
        load_working_dir(mem).graph, load_working_dir(opt).graph)
    # 3 + 3 nodes, 2 shared names ("Shared Topic", "Common Org")
    assert shared_nodes == {"Shared Topic", "Common Org"}
    assert g.number_of_nodes() == 4
    # 2 + 2 edges, 1 shared (Shared Topic -- Common Org)
    assert shared_edges == {frozenset({"Shared Topic", "Common Org"})}
    assert g.number_of_edges() == 3
    # Every shared-node/shared-edge merge rule is asserted below. A partial
    # compose regression (e.g. source_id switched from union to overwrite)
    # leaves the description assertions green, so each rule needs its own check.
    # The fixtures deliberately disagree across the two sides on every attribute
    # whose rule picks a winner (description, source_id, keywords, weight,
    # entity_type, created_at), so each assertion below fails under a last-wins
    # overwrite. file_path is the one intentional exception, noted at its checks.
    node = g.nodes["Shared Topic"]
    assert node["description"] == "memory view of shared topic<SEP>optical view of shared topic"
    # memory chunk ids first ([CH_SHARED, CH_MEM]), then optical's new one
    assert node["source_id"] == f"{CH_SHARED}<SEP>{CH_MEM}<SEP>{CH_OPT}"
    # Both sides carry the identical "custom_kg", which is what real dirs look
    # like. That makes this a dedupe check and nothing more: it catches
    # "custom_kg<SEP>custom_kg" (union without dedupe), but it CANNOT catch a
    # union-replaced-by-overwrite regression, because overwriting also yields
    # "custom_kg". The union-vs-overwrite proof lives on description/source_id,
    # where the two sides differ.
    assert node["file_path"] == "custom_kg"
    # memory "topic" vs optical "concept": memory wins, last-wins gives "concept"
    assert node["entity_type"] == "topic"
    # memory 1700000000 vs optical 1700000001: last-wins gives 1700000001
    assert node["created_at"] == 1700000000           # min
    # graphml types created_at as long; a float/str would break the writer
    assert isinstance(node["created_at"], int)

    edge = g.edges["Shared Topic", "Common Org"]
    assert edge["description"] == "mem rel desc<SEP>opt rel desc"
    # memory weight is 3.0 and optical 2.0, so last-wins would give 2.0 here
    assert edge["weight"] == 3.0                      # max
    assert edge["keywords"] == "collaboration<SEP>photonics"
    assert edge["source_id"] == f"{CH_MEM}<SEP>{CH_OPT}"
    # dedupe proof only, same caveat as the node file_path assertion above
    assert edge["file_path"] == "custom_kg"
    assert edge["created_at"] == 1700000000           # min
    assert isinstance(edge["created_at"], int)
    # memory-only and optical-only attrs survive untouched
    assert g.nodes["Mem Entity"]["description"] == "hbm memory wall"
    assert g.nodes["Opt Entity"]["description"] == "co-packaged optics"


def test_merge_graphs_empty_entity_type_filled_from_optical(dirs):
    """The other half of the entity_type rule: memory empty => optical fills it.

    "Common Org" carries entity_type="" on the memory side and "organization" on
    the optical side. Unconditional memory-wins would leave "" here; only the
    fill-if-empty branch produces "organization".
    """
    mem, opt = dirs
    g, _, _ = merge_graphs(load_working_dir(mem).graph, load_working_dir(opt).graph)
    # the empty value must survive the graphml round-trip for this to mean anything
    assert load_working_dir(mem).graph.nodes["Common Org"]["entity_type"] == ""
    assert g.nodes["Common Org"]["entity_type"] == "organization"


def test_merge_text_chunks_union_and_collision(dirs):
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    merged = merge_text_chunks(mem_wd.text_chunks, opt_wd.text_chunks)
    assert set(merged) == {CH_SHARED, CH_MEM, CH_OPT}          # 2 + 2 - 1 shared
    # The shared chunk carries identical `content` but a different create_time on
    # each side (1700000000 vs 1700000001). The union call above therefore only
    # survives a collision check that compares `content` alone: a whole-dict
    # comparison would abort on these perfectly valid fixtures.
    assert merged[CH_SHARED]["content"] == "shared chunk text"
    assert merged[CH_SHARED]["create_time"] == 1700000000       # memory row kept
    assert merged[CH_OPT]["content"] == "optical-only chunk"
    bad = dict(opt_wd.text_chunks)
    bad[CH_SHARED] = dict(bad[CH_SHARED], content="DIFFERENT")
    with pytest.raises(MergeError, match="collision"):
        merge_text_chunks(mem_wd.text_chunks, bad)


def test_merge_chunk_lists_unions_ids(dirs):
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    merged = merge_chunk_lists(mem_wd.entity_chunks, opt_wd.entity_chunks)
    shared = merged["Shared Topic"]
    assert shared["chunk_ids"] == [CH_SHARED, CH_MEM, CH_OPT]  # memory order first
    assert shared["count"] == 3
    assert merged["Mem Entity"]["chunk_ids"] == [CH_MEM]
    # Optical-only keys must survive too: folding in only the keys memory already
    # has would leave every assertion above green.
    assert merged["Opt Entity"]["chunk_ids"] == [CH_OPT]
    # Inputs stay untouched — a shallow copy would write the unioned list back
    # into the memory working dir that later merge steps still read from.
    assert mem_wd.entity_chunks["Shared Topic"]["chunk_ids"] == [CH_SHARED, CH_MEM]
    assert mem_wd.entity_chunks["Shared Topic"]["count"] == 2
    rel = merge_chunk_lists(mem_wd.relation_chunks, opt_wd.relation_chunks)
    assert rel["Shared Topic<SEP>Common Org"]["count"] == 2
    assert rel["Shared Topic<SEP>Common Org"]["chunk_ids"] == [CH_MEM, CH_OPT]


def _merged(dirs):
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    g, _, _ = merge_graphs(mem_wd.graph, opt_wd.graph)
    return mem_wd, opt_wd, g


def test_merge_vdb_chunks_dedupes(dirs):
    mem_wd, opt_wd, _ = _merged(dirs)
    vs = merge_vdb_chunks(mem_wd.vdb_chunks, opt_wd.vdb_chunks)
    assert vs.matrix.shape[0] == len(vs.data) == 3          # 2 + 2 - 1
    ids = [r["__id__"] for r in vs.data]
    assert len(ids) == len(set(ids))


def test_merge_vdb_chunks_keeps_memory_row_and_stays_aligned(dirs):
    """Counting rows cannot tell which side of a collision was kept, nor whether
    the surviving rows still line up with the matrix they index into."""
    mem_wd, opt_wd, _ = _merged(dirs)
    vs = merge_vdb_chunks(mem_wd.vdb_chunks, opt_wd.vdb_chunks)
    by_id = {r["__id__"]: r for r in vs.data}
    assert set(by_id) == {CH_SHARED, CH_MEM, CH_OPT}
    # The colliding id carries identical content by construction (the id is its
    # md5), so __created_at__ — 1700000000 on memory, 1700000001 on optical — is
    # the only field that reveals which row survived.
    assert by_id[CH_SHARED]["__created_at__"] == 1700000000
    for i, row in enumerate(vs.data):
        # data[i] must still describe matrix[i], for the appended optical row too
        np.testing.assert_allclose(vs.matrix[i], decode_row_vector(row["vector"]),
                                   atol=1e-3)
        np.testing.assert_allclose(vs.matrix[i], det_vector(row["content"]), atol=1e-3)


def test_merge_vdb_entities_reembeds_shared(dirs):
    mem_wd, opt_wd, g = _merged(dirs)
    calls = []

    def spy_embed(texts):
        calls.append(list(texts))
        return fake_embed(texts)

    vs = merge_vdb_entities(mem_wd.vdb_entities, opt_wd.vdb_entities, g, spy_embed)
    assert vs.matrix.shape[0] == len(vs.data) == 4          # 3 + 3 - 2 shared
    # only the 2 shared entities were re-embedded, with merged descriptions
    embedded = [t for batch in calls for t in batch]
    assert len(embedded) == 2
    assert ("Shared Topic\nmemory view of shared topic<SEP>optical view of shared topic"
            in embedded)
    # re-embedded row: content updated, matrix row == decoded row vector,
    # unit norm, and data[i] still aligns with matrix[i]
    idx = next(i for i, r in enumerate(vs.data) if r["entity_name"] == "Shared Topic")
    assert "<SEP>" in vs.data[idx]["content"]
    np.testing.assert_allclose(
        vs.matrix[idx], decode_row_vector(vs.data[idx]["vector"]), atol=1e-3)
    assert np.isclose(np.linalg.norm(vs.matrix[idx]), 1.0, rtol=1e-5)
    # untouched row keeps its original vector
    j = next(i for i, r in enumerate(vs.data) if r["entity_name"] == "Mem Entity")
    np.testing.assert_allclose(vs.matrix[j], det_vector("Mem Entity\nhbm memory wall"),
                               atol=1e-3)


def test_merge_vdb_entities_preserves_appended_rows_and_inputs(dirs):
    """Every assertion about a re-embedded row survives a misaligned append,
    because those rows get rewritten at whatever index they land on. Only a row
    that is copied through — the optical-only one — pins the append order."""
    mem_wd, opt_wd, g = _merged(dirs)
    before_data = [dict(r) for r in mem_wd.vdb_entities.data]
    before_matrix = mem_wd.vdb_entities.matrix.copy()

    vs = merge_vdb_entities(mem_wd.vdb_entities, opt_wd.vdb_entities, g, fake_embed)

    k = next(i for i, r in enumerate(vs.data) if r["entity_name"] == "Opt Entity")
    assert vs.data[k]["content"] == "Opt Entity\nco-packaged optics"
    np.testing.assert_allclose(vs.matrix[k],
                               det_vector("Opt Entity\nco-packaged optics"), atol=1e-3)
    for i, row in enumerate(vs.data):
        np.testing.assert_allclose(vs.matrix[i], decode_row_vector(row["vector"]),
                                   atol=1e-3)
    # The source store must come out untouched: later merge steps read it, and a
    # merged row that aliases it would rewrite memory's own vectors in place.
    assert [dict(r) for r in mem_wd.vdb_entities.data] == before_data
    np.testing.assert_array_equal(mem_wd.vdb_entities.matrix, before_matrix)


def test_reembedded_vectors_are_normalized_by_the_merge(dirs):
    """fake_embed already returns unit vectors, so deleting the normalization
    step leaves every other test green. Only a non-unit embedder can tell a
    normalizing merge from one that trusts its embedder."""
    mem_wd, opt_wd, g = _merged(dirs)

    def unnormalized_embed(texts):
        return fake_embed(texts) * 7.5

    vs = merge_vdb_entities(mem_wd.vdb_entities, opt_wd.vdb_entities, g,
                            unnormalized_embed)
    idx = next(i for i, r in enumerate(vs.data) if r["entity_name"] == "Shared Topic")
    assert np.isclose(np.linalg.norm(vs.matrix[idx]), 1.0, rtol=1e-5)
    np.testing.assert_allclose(vs.matrix[idx], det_vector(vs.data[idx]["content"]),
                               atol=1e-3)
    assert np.isclose(np.linalg.norm(decode_row_vector(vs.data[idx]["vector"])),
                      1.0, atol=1e-3)


def test_embed_fn_wrong_dim_aborts(dirs):
    """A short embedding would otherwise surface as a raw numpy broadcast error
    halfway through rewriting the store."""
    mem_wd, opt_wd, g = _merged(dirs)
    with pytest.raises(MergeError, match="shape"):
        merge_vdb_entities(mem_wd.vdb_entities, opt_wd.vdb_entities, g,
                           lambda texts: np.zeros((len(texts), 3), dtype=np.float32))


def test_merge_vdb_relationships_reembeds_shared(dirs):
    mem_wd, opt_wd, g = _merged(dirs)
    vs = merge_vdb_relationships(mem_wd.vdb_relationships, opt_wd.vdb_relationships,
                                 g, fake_embed)
    assert vs.matrix.shape[0] == len(vs.data) == 3          # 2 + 2 - 1 shared
    idx = next(i for i, r in enumerate(vs.data)
               if {r["src_id"], r["tgt_id"]} == {"Shared Topic", "Common Org"})
    expected = ("collaboration<SEP>photonics\tShared Topic\nCommon Org\n"
                "mem rel desc<SEP>opt rel desc")
    assert vs.data[idx]["content"] == expected


def test_merge_vdb_relationships_rewrites_both_encodings(dirs):
    """The shared relation's stored vector must encode the merged text in both
    the row field and the matrix row — one without the other silently returns
    the wrong relation for every query that hits it."""
    mem_wd, opt_wd, g = _merged(dirs)
    vs = merge_vdb_relationships(mem_wd.vdb_relationships, opt_wd.vdb_relationships,
                                 g, fake_embed)
    idx = next(i for i, r in enumerate(vs.data)
               if {r["src_id"], r["tgt_id"]} == {"Shared Topic", "Common Org"})
    merged_vec = det_vector(vs.data[idx]["content"])
    np.testing.assert_allclose(vs.matrix[idx], merged_vec, atol=1e-3)
    np.testing.assert_allclose(decode_row_vector(vs.data[idx]["vector"]), merged_vec,
                               atol=1e-3)
    # the optical-only relation is copied through untouched, at its own index
    k = next(i for i, r in enumerate(vs.data) if r["tgt_id"] == "Opt Entity")
    assert vs.data[k]["content"] == "optics\tShared Topic\nOpt Entity\nopt-only rel"
    np.testing.assert_allclose(vs.matrix[k], det_vector(vs.data[k]["content"]),
                               atol=1e-3)


# ---------------------------------------------------------------------------
# Task 6: provenance + post-merge verification
# ---------------------------------------------------------------------------

def test_provenance_domains(dirs):
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    prov = build_provenance(mem_wd, opt_wd, generated_at="2026-07-22T00:00:00Z")
    assert prov["entities"]["Shared Topic"] == "both"
    assert prov["entities"]["Mem Entity"] == "memory"
    assert prov["entities"]["Opt Entity"] == "optical"
    assert prov["chunks"][CH_SHARED] == "both"
    assert prov["chunks"][CH_OPT] == "optical"
    assert prov["relations"]["Shared Topic<SEP>Common Org"] == "both"
    assert prov["relations"]["Shared Topic<SEP>Mem Entity"] == "memory"
    assert prov["generated_at"] == "2026-07-22T00:00:00Z"


def test_provenance_multi_labels_entities_chunks_and_relations(tmp_path):
    """Multi-source labels use sorted domains without changing `both` legacy labels."""
    def source_spec(domain, shared=False, triple=False):
        entities = [{"name": f"{domain.title()} Entity", "entity_type": "topic",
                     "description": domain, "chunk_ids": [f"chunk-{domain}"]}]
        chunks = {f"chunk-{domain}": domain}
        relations = [{"src": "Triple Entity", "tgt": f"{domain.title()} Entity",
                      "description": domain, "keywords": domain, "weight": 1.0,
                      "chunk_ids": [f"chunk-{domain}"]}]
        if shared:
            entities.append({"name": "Shared Entity", "entity_type": "topic",
                             "description": "shared", "chunk_ids": ["chunk-shared"]})
            chunks["chunk-shared"] = "shared"
            relations.append({"src": "Triple Entity", "tgt": "Shared Entity",
                              "description": "shared", "keywords": "shared", "weight": 1.0,
                              "chunk_ids": ["chunk-shared"]})
        if triple:
            entities.extend([
                {"name": "Triple Entity", "entity_type": "topic",
                 "description": "triple", "chunk_ids": ["chunk-triple"]},
                {"name": "Triple Partner", "entity_type": "topic",
                 "description": "triple", "chunk_ids": ["chunk-triple"]},
            ])
            chunks["chunk-triple"] = "triple"
            relations.append({"src": "Triple Entity", "tgt": "Triple Partner",
                              "description": "triple", "keywords": "triple", "weight": 1.0,
                              "chunk_ids": ["chunk-triple"]})
        return {"entities": entities, "chunks": chunks, "relations": relations}

    paths = {domain: tmp_path / domain for domain in ("memory", "optical", "storage")}
    build_working_dir(paths["memory"], source_spec("memory", shared=True, triple=True))
    build_working_dir(paths["optical"], source_spec("optical", shared=True, triple=True))
    build_working_dir(paths["storage"], source_spec("storage", triple=True))
    # Deliberately unsorted input: labels must be sorted independently of the
    # orchestrator's source-dictionary insertion order.
    sources = {domain: load_working_dir(paths[domain])
               for domain in ("storage", "memory", "optical")}

    prov = build_provenance_multi(sources, generated_at="2026-07-23T00:00:00Z")

    assert prov["entities"] == {
        "Memory Entity": "memory", "Optical Entity": "optical",
        "Shared Entity": "memory+optical", "Storage Entity": "storage",
        "Triple Entity": "memory+optical+storage",
        "Triple Partner": "memory+optical+storage",
    }
    assert prov["chunks"] == {
        "chunk-memory": "memory", "chunk-optical": "optical",
        "chunk-shared": "memory+optical", "chunk-storage": "storage",
        "chunk-triple": "memory+optical+storage",
    }
    assert prov["relations"] == {
        "Triple Entity<SEP>Memory Entity": "memory",
        "Triple Entity<SEP>Optical Entity": "optical",
        "Triple Entity<SEP>Shared Entity": "memory+optical",
        "Triple Entity<SEP>Storage Entity": "storage",
        "Triple Entity<SEP>Triple Partner": "memory+optical+storage",
    }
    assert "Missing Entity" not in prov["entities"]
    assert "chunk-missing" not in prov["chunks"]
    assert "Missing Source<SEP>Missing Target" not in prov["relations"]
    assert prov["generated_at"] == "2026-07-23T00:00:00Z"


def _build_combined(mem_wd, opt_wd, path):
    """Run the Task 3/4/5 merge functions by hand into an in-memory WorkingDir.

    Deliberately does NOT call run_merge (Task 7): these verify_merge tests must
    run before the orchestrator exists, so they assemble a correct combined dir
    directly and then corrupt one store at a time.
    """
    g, _, _ = merge_graphs(mem_wd.graph, opt_wd.graph)
    return WorkingDir(
        path=path,
        graph=g,
        vdb_entities=merge_vdb_entities(mem_wd.vdb_entities, opt_wd.vdb_entities,
                                        g, fake_embed),
        vdb_relationships=merge_vdb_relationships(
            mem_wd.vdb_relationships, opt_wd.vdb_relationships, g, fake_embed),
        vdb_chunks=merge_vdb_chunks(mem_wd.vdb_chunks, opt_wd.vdb_chunks),
        text_chunks=merge_text_chunks(mem_wd.text_chunks, opt_wd.text_chunks),
        entity_chunks=merge_chunk_lists(mem_wd.entity_chunks, opt_wd.entity_chunks),
        relation_chunks=merge_chunk_lists(mem_wd.relation_chunks,
                                          opt_wd.relation_chunks),
    )


def test_verify_merge_accepts_clean_merge(dirs, tmp_path):
    """A correctly assembled combined dir passes. Without this baseline, a
    verify_merge that raised unconditionally would satisfy every corruption test
    below while being useless."""
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    combined = _build_combined(mem_wd, opt_wd, tmp_path / "combined")
    assert verify_merge(mem_wd, opt_wd, combined) is None


def test_verify_merge_catches_edge_loss(dirs, tmp_path):
    """Dropping an edge leaves node count untouched, so a node-count-only check
    passes. Only an independent edge-count check catches it."""
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    combined = _build_combined(mem_wd, opt_wd, tmp_path / "combined")
    combined.graph.remove_edge("Shared Topic", "Mem Entity")
    with pytest.raises(MergeError, match="graph edges"):
        verify_merge(mem_wd, opt_wd, combined)


def test_verify_merge_catches_vdb_row_loss(dirs, tmp_path):
    """Popping a vdb_entities row keeps graph node/edge counts correct, so a
    check that stops at the graph never notices the vdb store is short a row."""
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    combined = _build_combined(mem_wd, opt_wd, tmp_path / "combined")
    combined.vdb_entities.data.pop()
    combined.vdb_entities.matrix = combined.vdb_entities.matrix[:-1]
    with pytest.raises(MergeError, match="vdb_entities"):
        verify_merge(mem_wd, opt_wd, combined)


def test_verify_merge_bijection_uses_names_not_counts(dirs, tmp_path):
    """Rename one vdb entity: row COUNT still matches the graph node count, but
    the name SETS no longer agree. A bijection check that compares counts passes;
    only comparing the name sets catches the swap."""
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    combined = _build_combined(mem_wd, opt_wd, tmp_path / "combined")
    combined.vdb_entities.data[0]["entity_name"] = "GHOST ENTITY"
    assert combined.vdb_entities.matrix.shape[0] == len(combined.vdb_entities.data)
    with pytest.raises(MergeError, match="vdb_entities mismatch"):
        verify_merge(mem_wd, opt_wd, combined)


def _dangling_refs(wd):
    """Chunk ids referenced by a dir's entity/relation chunk lists that have no
    text_chunks entry — mirrors verify_merge's own per-dir dangling computation."""
    refs = {c for v in wd.entity_chunks.values() for c in v["chunk_ids"]}
    refs |= {c for v in wd.relation_chunks.values() for c in v["chunk_ids"]}
    return refs - set(wd.text_chunks)


def test_verify_merge_dangling_check_flags_merge_introduced_relation_chunk(dirs, tmp_path):
    """A chunk id referenced only by relation_chunks (not entity_chunks) must
    still be flagged when it is missing from text_chunks. A dangling check that
    inspects only entity_chunks would let this through.

    This is also the MERGE-INTRODUCED case for the baseline-relative check: the
    ghost id dangles in NEITHER source, so `dangling(combined) - (dangling(mem) |
    dangling(opt))` still contains it. Asserting that baseline property here (the
    ghost is absent from both sources' dangling sets) makes this test kill the
    mutation that skips the check entirely (dangling_new = set()) as well as the
    entity-chunks-only mutation. The complementary pass-through case — legacy tags
    that DO dangle in a source and must survive silently — is covered by
    test_verify_merge_preexisting_dangling_passes_through."""
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    combined = _build_combined(mem_wd, opt_wd, tmp_path / "combined")
    key = "Shared Topic<SEP>Common Org"
    ghost = "chunk-ghost-relation"
    assert ghost not in _dangling_refs(mem_wd)
    assert ghost not in _dangling_refs(opt_wd)
    combined.relation_chunks[key]["chunk_ids"].append(ghost)
    with pytest.raises(MergeError, match="referenced but missing") as exc:
        verify_merge(mem_wd, opt_wd, combined)
    assert ghost in str(exc.value)                 # the new dangling id is named


def test_verify_merge_preexisting_dangling_passes_through(tmp_path):
    """Legacy non-chunk tags that already dangle in the SOURCES ('UNKNOWN'
    placeholders, profile-run ids like 'memory-innovation-<timestamp>') are
    preserved unchanged by a mechanical union and must NOT abort the merge — this
    is the normal operating state of the live dirs, not a bug.

    Both sides carry a DISTINCT pre-existing dangling tag. A baseline that
    subtracted only one source's dangling set (dropping the other) would still see
    the other side's tag as merge-introduced and abort, so this single test kills
    both single-side mutants (dangling(mem)-only and dangling(opt)-only) at once.

    Uses in-test deep copies of the spec dicts (never mutates the shared
    fixtures); build_working_dir weaves each appended tag into the entity's
    graphml source_id too, mirroring how the real dirs carry these tags."""
    import copy
    mem_spec = copy.deepcopy(FIXTURE_MEMORY)
    opt_spec = copy.deepcopy(FIXTURE_OPTICAL)
    MEM_TAG = "memory-innovation-2025-11-17T144231"     # profile-run id (memory side)
    OPT_TAG = "UNKNOWN"                                  # placeholder (optical side)
    mem_spec["entities"][2]["chunk_ids"].append(MEM_TAG)   # "Mem Entity"
    opt_spec["entities"][2]["chunk_ids"].append(OPT_TAG)   # "Opt Entity"

    mem, opt = tmp_path / "mem", tmp_path / "opt"
    build_working_dir(mem, mem_spec)
    build_working_dir(opt, opt_spec)
    # Preconditions: each tag dangles in exactly its own source and neither is a
    # real chunk id, so the only thing keeping the merge green is baseline-relativity.
    assert MEM_TAG in _dangling_refs(load_working_dir(mem))
    assert OPT_TAG in _dangling_refs(load_working_dir(opt))

    out = tmp_path / "combined"
    report = run_merge(mem, opt, out, embed_fn=fake_embed)   # must SUCCEED
    assert report.output_dir == out
    assert out.exists()
    combined = load_working_dir(out)
    all_refs = {c for v in combined.entity_chunks.values() for c in v["chunk_ids"]}
    all_refs |= {c for v in combined.relation_chunks.values() for c in v["chunk_ids"]}
    assert MEM_TAG in all_refs and OPT_TAG in all_refs      # both survived the union
    assert MEM_TAG not in combined.text_chunks
    assert OPT_TAG not in combined.text_chunks


def test_verify_merge_collects_all_failures(dirs, tmp_path):
    """Two independent corruptions must both be named in a single MergeError.
    A verify that raised on the first failure would report only one."""
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    combined = _build_combined(mem_wd, opt_wd, tmp_path / "combined")
    combined.graph.remove_edge("Shared Topic", "Mem Entity")   # first failure
    key = "Shared Topic<SEP>Common Org"
    combined.relation_chunks[key]["chunk_ids"].append("chunk-ghost-relation")  # later failure
    with pytest.raises(MergeError) as exc:
        verify_merge(mem_wd, opt_wd, combined)
    msg = str(exc.value)
    assert "graph edges" in msg
    assert "referenced but missing" in msg


def test_verify_merge_catches_missing_vdb_row(dirs, tmp_path):
    # run the real merge into a dir, then corrupt it and expect verify to fail.
    # run_merge is Task 7's deliverable; imported lazily so its absence xfails
    # this test alone rather than breaking collection of the whole module.
    from mira.graph_merge import run_merge
    out = tmp_path / "combined"
    run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed)
    combined = load_working_dir(out)
    combined.vdb_entities.data.pop()          # break node<->vdb bijection
    combined.vdb_entities.matrix = combined.vdb_entities.matrix[:-1]
    mem_wd, opt_wd = (load_working_dir(d) for d in dirs)
    with pytest.raises(MergeError):
        verify_merge(mem_wd, opt_wd, combined)


# ---------------------------------------------------------------------------
# Task 7: run_merge orchestration — tmp dir, verify, atomic swap, bak rotation
# ---------------------------------------------------------------------------

import json as _json
from datetime import datetime, timezone

import mira.graph_merge as _gm
from mira.graph_merge import run_merge, STORAGE_FILES


def test_run_merge_end_to_end(dirs, tmp_path):
    out = tmp_path / "combined"
    report = run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed)
    assert report.nodes == 4 and report.edges == 3
    assert report.shared_entities == 2 and report.shared_relations == 1
    assert report.shared_chunks == 1
    assert report.reembedded == 3          # 2 entities + 1 relation
    assert report.output_dir == out
    for f in STORAGE_FILES:
        assert (out / f).exists(), f
    assert _json.loads((out / "kv_store_llm_response_cache.json").read_text()) == {}
    prov = _json.loads((out / "provenance.json").read_text())
    assert prov["entities"]["Shared Topic"] == "both"
    combined = load_working_dir(out)       # loads + passes VectorStore self-checks
    assert combined.graph.number_of_nodes() == 4


def test_run_merge_dry_run_writes_nothing(dirs, tmp_path):
    out = tmp_path / "combined"
    report = run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed, dry_run=True)
    assert report.nodes == 4
    assert not out.exists()
    assert not list(tmp_path.glob("combined.tmp-*"))


def test_run_merge_dry_run_skips_embed(dirs, tmp_path):
    """The CLI passes embed_fn=None for dry runs, so a dry run that still calls
    the embedder would crash. Passing None here makes that failure observable
    instead of silently harmless with a real fake_embed."""
    out = tmp_path / "combined"
    report = run_merge(dirs[0], dirs[1], out, embed_fn=None, dry_run=True)
    assert report.nodes == 4
    assert report.output_dir is None
    assert not out.exists()


def test_run_merge_is_deterministic(dirs, tmp_path):
    out1, out2 = tmp_path / "c1", tmp_path / "c2"
    fixed = datetime(2026, 7, 22, tzinfo=timezone.utc)
    run_merge(dirs[0], dirs[1], out1, embed_fn=fake_embed, now=fixed)
    run_merge(dirs[0], dirs[1], out2, embed_fn=fake_embed, now=fixed)
    for f in [*STORAGE_FILES, "provenance.json"]:
        assert (out1 / f).read_bytes() == (out2 / f).read_bytes(), f


def test_run_merge_rotates_baks_and_sources_untouched(dirs, tmp_path):
    out = tmp_path / "combined"
    before = {p: p.stat().st_mtime_ns for d in dirs for p in d.iterdir()}
    run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed,
              now=datetime(2026, 7, 20, tzinfo=timezone.utc))
    run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed,
              now=datetime(2026, 7, 21, tzinfo=timezone.utc))
    run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed,
              now=datetime(2026, 7, 22, tzinfo=timezone.utc))
    run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed,
              now=datetime(2026, 7, 23, tzinfo=timezone.utc))
    baks = sorted(p.name for p in tmp_path.glob("combined.bak-*"))
    assert baks == ["combined.bak-2026-07-22", "combined.bak-2026-07-23"]
    after = {p: p.stat().st_mtime_ns for d in dirs for p in d.iterdir()}
    assert before == after                 # sources strictly read-only


def test_run_merge_verify_failure_leaves_tmp_and_no_output(dirs, tmp_path, monkeypatch):
    """A failed verification must abort loudly, leave the temp dir for
    inspection, and never write into the output dir. Kills three mutations at
    once: writing directly into output_dir (out would exist), deleting the tmp
    dir on failure (tmp would be gone), and swapping before verifying on a
    first run (out would exist)."""
    out = tmp_path / "combined"
    monkeypatch.setattr(
        _gm, "verify_merge",
        lambda *a, **k: (_ for _ in ()).throw(MergeError("injected verify failure")))
    with pytest.raises(MergeError, match="injected verify failure"):
        run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed,
                  now=datetime(2026, 7, 22, tzinfo=timezone.utc))
    assert not out.exists()                                # output never written
    tmps = list(tmp_path.glob("combined.tmp-*"))
    assert len(tmps) == 1 and tmps[0].is_dir()             # tmp left for inspection


def test_run_merge_verify_failure_preserves_existing_output(dirs, tmp_path, monkeypatch):
    """When an output dir already exists, a later failed merge must leave it
    exactly as it was and create no bak. Kills swap-before-verify: swapping
    first would have renamed the good output away into a bak."""
    out = tmp_path / "combined"
    run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed,
              now=datetime(2026, 7, 20, tzinfo=timezone.utc))
    sentinel = (out / "graph_chunk_entity_relation.graphml").read_bytes()
    before = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    monkeypatch.setattr(
        _gm, "verify_merge",
        lambda *a, **k: (_ for _ in ()).throw(MergeError("injected verify failure")))
    with pytest.raises(MergeError):
        run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed,
                  now=datetime(2026, 7, 21, tzinfo=timezone.utc))
    assert out.exists()
    assert (out / "graph_chunk_entity_relation.graphml").read_bytes() == sentinel
    assert not list(tmp_path.glob("combined.bak-*"))       # swap never happened
    after = {p.name: p.stat().st_mtime_ns for p in out.iterdir()}
    assert before == after


def test_run_merge_verifies_reloaded_temp_dir(dirs, tmp_path, monkeypatch):
    """verify_merge must run on a WorkingDir re-loaded from the temp dir, not on
    the in-memory merge products — the reload is what re-runs every VectorStore
    row-vector/matrix self-check on the bytes actually written. Kills the
    mutation that verifies in-memory objects (path is not a discriminator, since
    an in-memory WorkingDir can carry any path): only reloading the temp dir
    actually calls load_working_dir on it, so we assert that call happened and
    that its result is exactly what verify_merge received."""
    loaded_paths = []
    real_load = _gm.load_working_dir

    def spy_load(path):
        wd = real_load(path)
        loaded_paths.append(Path(path))
        return wd

    verified = {}
    real_verify = _gm.verify_merge

    def spy_verify(mem, opt, combined):
        verified["combined"] = combined
        return real_verify(mem, opt, combined)

    monkeypatch.setattr(_gm, "load_working_dir", spy_load)
    monkeypatch.setattr(_gm, "verify_merge", spy_verify)
    out = tmp_path / "combined"
    run_merge(dirs[0], dirs[1], out, embed_fn=fake_embed,
              now=datetime(2026, 7, 22, tzinfo=timezone.utc))
    tmp_loads = [p for p in loaded_paths if p.name.startswith("combined.tmp-")]
    assert len(tmp_loads) == 1                      # the temp dir was re-loaded
    # and the object verify_merge saw is that reload, not an in-memory build
    assert verified["combined"].path == tmp_loads[0]

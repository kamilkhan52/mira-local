import networkx as nx
import numpy as np

from mira.discovery.canonicalize import build_merge_map, merge_topics_in_graph
from mira.hypothesis.vectors import EntityVectors


def _vectors(spec: dict[str, list[float]]) -> EntityVectors:
    names = list(spec)
    m = np.array([spec[n] for n in names], dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return EntityVectors(names, m)


def _graph(degrees: dict[str, int]) -> nx.Graph:
    """Topic nodes with the requested degree (via dummy neighbor nodes)."""
    g = nx.Graph()
    for t, deg in degrees.items():
        g.add_node(t, entity_type="Topic")
        for i in range(deg):
            g.add_edge(t, f"{t}-nbr-{i}", keywords="related_to topic")
    return g


def test_identical_normalized_names_merge_without_vectors():
    g = _graph({"CXL": 3, "CXL (Compute Express Link)": 1})
    mm = build_merge_map(["CXL", "CXL (Compute Express Link)"], g, vectors=None)
    assert mm == {"CXL (Compute Express Link)": "CXL"}  # higher degree wins


def test_token_extension_merges_only_with_vector_support():
    g = _graph({"CXL": 3, "CXL memory pooling": 1, "memory": 3, "memory pooling": 1})
    vecs = _vectors({
        "CXL": [1.0, 0.0, 0.0],
        "CXL memory pooling": [0.9, 0.435, 0.0],   # cos ≈ 0.90 ≥ EXT_SIM_THRESHOLD
        "memory": [0.0, 1.0, 0.0],
        "memory pooling": [0.0, 0.0, 1.0],          # cos = 0 with "memory"
    })
    mm = build_merge_map(["CXL", "CXL memory pooling", "memory", "memory pooling"], g, vecs)
    assert mm == {"CXL memory pooling": "CXL"}
    assert "memory pooling" not in mm  # extension without vector support: no merge


def test_acronym_pairs_merge_without_vectors():
    g = _graph({"HBM": 5, "High Bandwidth Memory": 2,
                "PIM": 3, "Processing-in-Memory": 4})
    mm = build_merge_map(sorted(["HBM", "High Bandwidth Memory",
                                 "PIM", "Processing-in-Memory"]), g, vectors=None)
    assert mm == {"High Bandwidth Memory": "HBM",       # higher degree wins
                  "PIM": "Processing-in-Memory"}


def test_versioned_acronyms_join_the_family_transitively():
    # HBM3E ~ High Bandwidth Memory (versioned acronym) and HBM ~ High
    # Bandwidth Memory (acronym) -> one group, canonical by degree.
    g = _graph({"HBM": 5, "HBM3E": 1, "High Bandwidth Memory": 2})
    mm = build_merge_map(sorted(["HBM", "HBM3E", "High Bandwidth Memory"]),
                         g, vectors=None)
    assert mm == {"HBM3E": "HBM", "High Bandwidth Memory": "HBM"}


def test_acronym_rule_does_not_merge_non_acronyms():
    g = _graph({"CXL": 2, "Cache Hierarchy & Data Placement": 2})
    assert build_merge_map(["CXL", "Cache Hierarchy & Data Placement"],
                           g, vectors=None) == {}


def test_high_vector_similarity_merges_unrelated_names():
    g = _graph({"PIM": 2, "processing in memory": 1})
    vecs = _vectors({"PIM": [1.0, 0.0], "processing in memory": [0.99, 0.141]})  # cos ≈ 0.99
    mm = build_merge_map(["PIM", "processing in memory"], g, vecs)
    assert mm == {"processing in memory": "PIM"}


def test_distinct_topics_survive():
    g = _graph({"CXL": 2, "DDR6": 2})
    vecs = _vectors({"CXL": [1.0, 0.0], "DDR6": [0.0, 1.0]})
    assert build_merge_map(["CXL", "DDR6"], g, vecs) == {}


def test_merge_topics_in_graph_moves_edges_and_drops_alias():
    g = nx.Graph()
    for n, t in [("CXL", "Topic"), ("cxl pooling", "Topic"),
                 ("P1", "Paper"), ("Samsung", "Institution")]:
        g.add_node(n, entity_type=t)
    g.add_edge("P1", "cxl pooling", keywords="primary_topic topic")
    g.add_edge("Samsung", "cxl pooling", keywords="researches topic")
    g.add_edge("Samsung", "CXL", keywords="researches topic")  # pre-existing: must survive
    merge_topics_in_graph(g, {"cxl pooling": "CXL"})
    assert "cxl pooling" not in g
    assert g["P1"]["CXL"]["keywords"] == "primary_topic topic"
    assert g["Samsung"]["CXL"]["keywords"] == "researches topic"


def test_merge_topics_in_graph_remaps_alias_neighbors():
    # alias A related_to alias B (different groups) → edge lands canonical-to-canonical
    g = nx.Graph()
    for n in ("A", "a-alias", "B", "b-alias"):
        g.add_node(n, entity_type="Topic")
    g.add_edge("a-alias", "b-alias", keywords="related_to topic")
    merge_topics_in_graph(g, {"a-alias": "A", "b-alias": "B"})
    assert g.has_edge("A", "B")
    assert "a-alias" not in g and "b-alias" not in g


def test_merge_topics_alias_collision_is_deterministic():
    # Two aliases in one group both link to P1 with different keywords;
    # sorted alias order means "a1"'s attributes win.
    g = nx.Graph()
    for n, t in [("CXL", "Topic"), ("a1", "Topic"), ("a2", "Topic"), ("P1", "Paper")]:
        g.add_node(n, entity_type=t)
    g.add_edge("a1", "P1", keywords="primary_topic topic")
    g.add_edge("a2", "P1", keywords="also_covers topic")
    merge_topics_in_graph(g, {"a1": "CXL", "a2": "CXL"})
    assert g["CXL"]["P1"]["keywords"] == "primary_topic topic"
    assert "a1" not in g and "a2" not in g

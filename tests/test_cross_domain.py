"""Cross-domain topic discovery: layer extraction, shared entities, payload.

Synthetic graphs mirror the real GraphML shape (undirected, `entity_type` on
nodes, `keywords` + `description` on edges) so no live working_dir is needed.
"""
import sys
from types import SimpleNamespace

import networkx as nx

import cross_domain as cross_domain_cli
from mira.discovery.cross_domain import (
    build_payload, estimate_tokens, load_topic_layer, shared_entities, shared_topics,
)
from mira.hypothesis.synthesis import SYNTHESIS_MODEL


def test_cross_domain_cli_uses_the_supported_hypothesis_model():
    assert cross_domain_cli.CROSS_DOMAIN_MODEL == SYNTHESIS_MODEL


def _memory_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("HBM4", entity_type="Topic", description="High bandwidth memory gen 4")
    g.add_node("PIM", entity_type="Topic", description="Processing in memory")
    g.add_node("2D Materials", entity_type="Topic", description="Atomically thin crystals")
    g.add_node("Kim", entity_type="Author", description="")
    g.add_node("Samsung", entity_type="Institution", description="")
    g.add_node("Unknown", entity_type="Author", description="")
    g.add_node("P1", entity_type="Paper", description="A memory paper")
    g.add_edge("HBM4", "PIM", keywords="related_to topic",
               description="both target bandwidth")
    g.add_edge("P1", "HBM4", keywords="primary_topic topic", description="")
    g.add_edge("Kim", "P1", keywords="authored_by author", description="")
    return g


def _optical_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("Silicon Photonics", entity_type="Topic",
               description="Photonic ICs in silicon")
    g.add_node("Ring Modulator", entity_type="Topic", description="Resonant modulator")
    g.add_node("2D Materials", entity_type="Topic",
               description="Used for optical modulation")
    g.add_node("Kim", entity_type="Author", description="")
    g.add_node("Unknown", entity_type="Institution", description="")
    g.add_node("P2", entity_type="Paper", description="An optical paper")
    g.add_edge("Silicon Photonics", "Ring Modulator", keywords="related_to topic",
               description="modulators built in SiP")
    g.add_edge("P2", "Silicon Photonics", keywords="primary_topic topic", description="")
    g.add_edge("Kim", "P2", keywords="authored_by author", description="")
    return g


def test_topic_layer_keeps_only_topic_nodes():
    layer = load_topic_layer(_memory_graph(), "memory")
    assert set(layer.topics) == {"HBM4", "PIM", "2D Materials"}
    assert layer.topics["HBM4"] == "High bandwidth memory gen 4"
    assert layer.domain == "memory"


def test_topic_layer_keeps_only_topic_to_topic_links():
    layer = load_topic_layer(_memory_graph(), "memory")
    # P1->HBM4 and Kim->P1 have a non-Topic endpoint and must be dropped.
    assert layer.links == [("HBM4", "PIM", "related_to topic", "both target bandwidth")]


def test_topic_layer_link_endpoints_are_sorted_for_determinism():
    g = nx.Graph()
    g.add_node("Zeta", entity_type="Topic", description="")
    g.add_node("Alpha", entity_type="Topic", description="")
    g.add_edge("Zeta", "Alpha", keywords="related_to topic", description="d")
    layer = load_topic_layer(g, "memory")
    assert layer.links == [("Alpha", "Zeta", "related_to topic", "d")]


def test_shared_topics_are_those_in_both_graphs():
    mem = load_topic_layer(_memory_graph(), "memory")
    opt = load_topic_layer(_optical_graph(), "optical")
    assert shared_topics(mem, opt) == ["2D Materials"]


def test_shared_entities_exclude_topics():
    # "2D Materials" is in both graphs but is a Topic; it belongs to the
    # shared-topics section, not the shared-entities section.
    result = shared_entities(_memory_graph(), _optical_graph())
    assert ("2D Materials", "Topic") not in result


def test_shared_entities_finds_cross_domain_author():
    assert shared_entities(_memory_graph(), _optical_graph()) == [("Kim", "Author")]


def test_shared_entities_drops_placeholder_names():
    # "Unknown" is in both real graphs (corrupted-author fallout). Presenting it
    # as a researcher publishing across domains would be a fabricated signal.
    names = [n for n, _ in shared_entities(_memory_graph(), _optical_graph())]
    assert "Unknown" not in names


def test_shared_entities_prefers_first_graph_type_on_disagreement():
    # "Unknown" is Author in memory and Institution in optical. Exactly one such
    # disagreement exists in the real graphs. Drop the placeholder, but prove the
    # tie-break rule on a non-placeholder name.
    mem, opt = _memory_graph(), _optical_graph()
    mem.add_node("Ambiguous", entity_type="Author", description="")
    opt.add_node("Ambiguous", entity_type="Institution", description="")
    assert ("Ambiguous", "Author") in shared_entities(mem, opt)


def _payload() -> str:
    mem = load_topic_layer(_memory_graph(), "memory")
    opt = load_topic_layer(_optical_graph(), "optical")
    return build_payload(mem, opt, shared_topics(mem, opt),
                         shared_entities(_memory_graph(), _optical_graph()))


def test_payload_has_all_five_sections_with_counts():
    p = _payload()
    # Domain counts are EXCLUSIVE topics: "2D Materials" is shared, so each
    # domain shows 2 of its 3 topics here and the shared one is listed once
    # in its own section. Both domains' single link is the only link in that
    # layer, so it trivially has one distinct (keywords, description) pair and
    # the links header carries the "(all: ...)" summary.
    assert "# MEMORY DOMAIN — 2 topics" in p
    assert "# MEMORY DOMAIN — 1 topic links (all: related_to topic / both target bandwidth)" in p
    assert "# OPTICAL DOMAIN — 2 topics" in p
    assert "# OPTICAL DOMAIN — 1 topic links (all: related_to topic / modulators built in SiP)" in p
    assert "# TOPICS PRESENT IN BOTH DOMAINS (1)" in p
    assert "# ENTITIES PRESENT IN BOTH DOMAINS (1)" in p


def test_shared_topic_listed_once_in_shared_section_only():
    p = _payload()
    # Shared section renders the bare name...
    assert "- 2D Materials\n" in p
    # ...and neither domain section repeats it with its description.
    assert "- 2D Materials: " not in p


def test_links_may_still_reference_a_shared_topic():
    # Excluding shared topics from domain lists must not drop links that touch
    # them — the name still resolves against the shared section.
    mem = _memory_graph()
    mem.add_edge("HBM4", "2D Materials", keywords="related_to topic",
                 description="stacked layers")
    layer = load_topic_layer(mem, "memory")
    opt = load_topic_layer(_optical_graph(), "optical")
    p = build_payload(layer, opt, shared_topics(layer, opt), [])
    assert "- 2D Materials -> HBM4 [related_to topic]: stacked layers" in p


def test_uniform_link_metadata_renders_bare_with_shared_value_in_header():
    # Every real graph link shares one (keywords, description) pair. When a
    # layer's links are uniform, spelling out [kw]: desc on every line is pure
    # token waste — the value belongs once, in the header.
    mem = load_topic_layer(_memory_graph(), "memory")
    opt = load_topic_layer(_optical_graph(), "optical")
    p = build_payload(mem, opt, shared_topics(mem, opt), [])
    assert "# MEMORY DOMAIN — 1 topic links (all: related_to topic / both target bandwidth)" in p
    assert "- HBM4 -> PIM\n" in p
    assert "[related_to topic]" not in p.split("# OPTICAL DOMAIN")[0]


def test_heterogeneous_link_metadata_keeps_per_link_format():
    # As soon as a layer has two or more distinct (keywords, description)
    # pairs, the shared-header shortcut is unsafe — fall back to spelling out
    # every link so no distinct meaning is lost.
    mem = _memory_graph()
    mem.add_edge("HBM4", "2D Materials", keywords="related_to topic",
                 description="stacked layers")
    layer = load_topic_layer(mem, "memory")
    opt = load_topic_layer(_optical_graph(), "optical")
    p = build_payload(layer, opt, shared_topics(layer, opt), [])
    assert "# MEMORY DOMAIN — 2 topic links\n" in p
    assert "(all:" not in p.split("# OPTICAL DOMAIN")[0]
    assert "- HBM4 -> PIM [related_to topic]: both target bandwidth" in p
    assert "- 2D Materials -> HBM4 [related_to topic]: stacked layers" in p


def test_sep_repeated_values_do_not_block_the_uniform_shortcut():
    # LightRAG joins multi-valued attributes with <SEP>, so "X<SEP>X" records
    # the same fact twice rather than two facts. In the real memory graph 10 of
    # 608 links carry such repeats — enough, if taken literally, to force the
    # verbose format on the other 598 for no informational gain.
    mem = _memory_graph()
    mem.add_edge("HBM4", "2D Materials", keywords="related_to topic",
                 description="both target bandwidth<SEP>both target bandwidth")
    layer = load_topic_layer(mem, "memory")
    opt = load_topic_layer(_optical_graph(), "optical")
    p = build_payload(layer, opt, shared_topics(layer, opt), [])
    memory_section = p.split("# OPTICAL DOMAIN")[0]
    assert "# MEMORY DOMAIN — 2 topic links (all: related_to topic / both target bandwidth)" in p
    assert "- HBM4 -> PIM\n" in p
    assert "- 2D Materials -> HBM4\n" in p
    assert "<SEP>" not in memory_section


def test_sep_joined_distinct_values_are_preserved():
    # Only IDENTICAL repeats collapse. A <SEP> value carrying genuinely
    # different segments is real multi-valued data and must survive intact.
    mem = _memory_graph()
    mem.add_edge("HBM4", "2D Materials", keywords="related_to topic",
                 description="stacked layers<SEP>thermal coupling")
    layer = load_topic_layer(mem, "memory")
    opt = load_topic_layer(_optical_graph(), "optical")
    p = build_payload(layer, opt, shared_topics(layer, opt), [])
    assert "- 2D Materials -> HBM4 [related_to topic]: stacked layers<SEP>thermal coupling" in p
    assert "(all:" not in p.split("# OPTICAL DOMAIN")[0]


def test_payload_renders_topics_and_links():
    p = _payload()
    assert "- HBM4: High bandwidth memory gen 4" in p
    # The memory layer's only link is trivially the sole distinct
    # (keywords, description) pair, so it renders bare with the shared value
    # folded into the section header instead of repeated per line.
    assert "- HBM4 -> PIM\n" in p
    assert "- HBM4 -> PIM [related_to topic]: both target bandwidth" not in p
    assert "- Kim (Author)" in p


def test_payload_is_deterministic():
    assert _payload() == _payload()


def test_payload_is_stable_under_node_insertion_order():
    # Same topics, same topic-to-topic links, but the two graphs insert the
    # nodes in opposite orders. dict/nx preserve insertion order, so this only
    # passes because build_payload explicitly does sorted(exclusive.items())
    # over the topics dict; it fails if that sort is replaced with .items().
    def graph(node_order: list[str]) -> nx.Graph:
        g = nx.Graph()
        for name in node_order:
            g.add_node(name, entity_type="Topic", description=f"{name} description")
        g.add_edge("Alpha", "Beta", keywords="related_to topic", description="a-b")
        g.add_edge("Beta", "Gamma", keywords="related_to topic", description="b-g")
        return g

    forward = load_topic_layer(graph(["Alpha", "Beta", "Gamma"]), "memory")
    reverse = load_topic_layer(graph(["Gamma", "Beta", "Alpha"]), "memory")
    other = load_topic_layer(_optical_graph(), "optical")

    p_forward = build_payload(forward, other, shared_topics(forward, other), [])
    p_reverse = build_payload(reverse, other, shared_topics(reverse, other), [])
    assert p_forward == p_reverse


def test_empty_section_renders_placeholder_not_blank():
    mem = load_topic_layer(_memory_graph(), "memory")
    opt = load_topic_layer(_optical_graph(), "optical")
    p = build_payload(mem, opt, [], [])
    assert "# TOPICS PRESENT IN BOTH DOMAINS (0)\n(none)" in p


def test_estimate_tokens_is_chars_over_four():
    assert estimate_tokens("a" * 400) == 100


def _stub_graphs(monkeypatch, mod):
    """Point the CLI at synthetic graphs instead of the live working dirs."""
    monkeypatch.setattr(mod, "resolve_target",
                        lambda name: SimpleNamespace(graphml=name))
    monkeypatch.setattr(mod, "load_graph",
                        lambda name: _memory_graph() if name == "memory"
                        else _optical_graph())


def test_dry_run_makes_no_llm_call(tmp_path, monkeypatch):
    import cross_domain

    _stub_graphs(monkeypatch, cross_domain)

    def boom():
        raise AssertionError("--dry-run must not create an LLM client")

    monkeypatch.setattr(cross_domain, "make_llm_client", boom)
    out = tmp_path / "out.md"
    monkeypatch.setattr(sys, "argv",
                        ["cross_domain.py", "--dry-run", "--out", str(out)])

    assert cross_domain.main() == 0
    payload_file = out.with_suffix(".payload.txt")
    assert payload_file.exists()
    assert "# MEMORY DOMAIN — 2 topics" in payload_file.read_text()
    assert not out.exists()


def test_real_run_writes_header_and_reply(tmp_path, monkeypatch):
    import cross_domain

    _stub_graphs(monkeypatch, cross_domain)
    monkeypatch.setattr(cross_domain, "make_llm_client", lambda: "client")

    captured = {}

    def fake_llm_call(client, model, system, user):
        captured.update(client=client, model=model, system=system, user=user)
        return "### A Candidate Topic\n- **Memory topics:** HBM4\n"

    monkeypatch.setattr(cross_domain, "llm_call", fake_llm_call)
    out = tmp_path / "out.md"
    monkeypatch.setattr(sys, "argv",
                        ["cross_domain.py", "--limit", "7", "--out", str(out)])

    assert cross_domain.main() == 0
    text = out.read_text()
    assert "# Cross-Domain Research Topics — Memory × Optical" in text
    assert "**Memory graph:** 3 topics, 1 topic links" in text
    assert "### A Candidate Topic" in text
    # The payload must reach the model, and --limit must be honoured.
    assert "Propose 7 candidate research topics." in captured["user"]
    assert "# OPTICAL DOMAIN — 2 topics" in captured["user"]
    assert captured["model"] == cross_domain.CROSS_DOMAIN_MODEL
    assert captured["system"] == cross_domain.SYSTEM

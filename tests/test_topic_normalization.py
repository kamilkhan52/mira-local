from mira.graph_ingest import _build_payload, normalize_topic


def test_strips_trailing_parenthetical():
    assert normalize_topic("CXL (Compute Express Link - memory pooling)") == "CXL"
    assert normalize_topic("Emerging Memory (ReRAM)") == "Emerging Memory"
    assert normalize_topic("Performance Optimization (bandwidth, latency)") == "Performance Optimization"


def test_leaves_paren_free_topics_unchanged():
    assert normalize_topic("HBM") == "HBM"
    assert normalize_topic("AI/ML Memory Systems") == "AI/ML Memory Systems"


def test_falls_back_when_only_parenthetical():
    assert normalize_topic("(ReRAM)") == "(ReRAM)"


def test_build_payload_merges_qualified_and_bare_topics():
    config = {"profile_id": "p", "current_date": "2026-03-01", "topic": {"focus": "memory"}}
    papers = [
        {"title": "Paper A", "raw_id": "u1", "primary_topic": "CXL (Compute Express Link - memory pooling)"},
        {"title": "Paper B", "raw_id": "u2", "primary_topic": "CXL"},
    ]
    payload = _build_payload(papers, [], config, "src")
    topic_nodes = [e["entity_name"] for e in payload["entities"] if e["entity_type"] == "Topic"]
    # Both papers collapse onto a single canonical CXL node
    assert topic_nodes.count("CXL") == 1
    assert not any("(" in t for t in topic_nodes)
    # Both papers' primary_topic edges point at the same canonical node
    cxl_edges = [r for r in payload["relationships"] if r["tgt_id"] == "CXL" and "primary_topic" in r["keywords"]]
    assert {r["src_id"] for r in cxl_edges} == {"Paper A", "Paper B"}

import mira.graph_ingest as gi
from mira.graph_ingest import _build_payload

PAPER = {
    "title": "Advances in HBM4 Memory Architecture",
    "id": "2605.19972",
    "raw_id": "https://arxiv.org/abs/2605.19972v1",
    "affiliations": ["Samsung Electronics", "KAIST"],
    "primary_topic": "HBM",
    "secondary_topics": ["DRAM", "memory bandwidth"],
    "relevance_score": 8,
    "credibility_tier": 9,
    "key_findings": "New bandwidth record achieved.",
    "short_summary": "A study on HBM4.",
}

ARTICLE = {
    "title": "EE Times: CXL 3.0 Gains Momentum",
    "source": "EE Times",
    "date": "2026-05-20",
    "url": "https://eetimes.com/cxl-3.0",
    "summary": "CXL 3.0 is gaining traction.",
}

CONFIG = {
    "profile_id": "memory-innovation",
    "current_date": "2026-05-27",
    "topic": {"focus": "memory technology"},
}

SOURCE_ID = "memory-innovation-2026-05-27T120000"


def test_paper_entity_name_is_title():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    names = [e["entity_name"] for e in payload["entities"]]
    assert "Advances in HBM4 Memory Architecture" in names


def test_no_noise_entities():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    names = [e["entity_name"] for e in payload["entities"]]
    for noise in ["relevance score", "credibility tier", "8/10", "2605.19972", "key findings"]:
        assert noise not in names, f"Noise entity found: {noise!r}"


def test_institution_entities_created():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    types = {e["entity_name"]: e["entity_type"] for e in payload["entities"]}
    assert types.get("Samsung Electronics") == "Institution"
    assert types.get("KAIST") == "Institution"


def test_topic_entities_created():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    types = {e["entity_name"]: e["entity_type"] for e in payload["entities"]}
    assert types.get("HBM") == "Topic"
    assert types.get("DRAM") == "Topic"
    assert types.get("memory bandwidth") == "Topic"


def test_institution_deduplication():
    paper2 = {**PAPER, "title": "Another HBM Paper", "raw_id": "https://arxiv.org/abs/2605.99999v1"}
    payload = _build_payload([PAPER, paper2], [], CONFIG, SOURCE_ID)
    samsung_count = sum(1 for e in payload["entities"] if e["entity_name"] == "Samsung Electronics")
    assert samsung_count == 1


def test_researches_edge_emitted():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    edges = [(r["src_id"], r["tgt_id"]) for r in payload["relationships"]]
    assert ("Samsung Electronics", "HBM") in edges
    assert ("KAIST", "HBM") in edges


def test_researches_edge_deduplicated():
    paper2 = {**PAPER, "title": "Another HBM Paper", "raw_id": "https://arxiv.org/abs/2605.99999v1"}
    payload = _build_payload([PAPER, paper2], [], CONFIG, SOURCE_ID)
    researches_edges = [
        r for r in payload["relationships"]
        if r["src_id"] == "Samsung Electronics" and r["tgt_id"] == "HBM"
    ]
    assert len(researches_edges) == 1


def test_published_by_edge_emitted():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    edges = [(r["src_id"], r["tgt_id"]) for r in payload["relationships"]]
    assert ("Advances in HBM4 Memory Architecture", "Samsung Electronics") in edges


def test_primary_topic_edge_emitted():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    edges = [(r["src_id"], r["tgt_id"]) for r in payload["relationships"]]
    assert ("Advances in HBM4 Memory Architecture", "HBM") in edges


def test_secondary_topic_edges_emitted():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    edges = [(r["src_id"], r["tgt_id"]) for r in payload["relationships"]]
    assert ("Advances in HBM4 Memory Architecture", "DRAM") in edges
    assert ("Advances in HBM4 Memory Architecture", "memory bandwidth") in edges


def test_related_to_edge_emitted():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    related = [
        r for r in payload["relationships"]
        if r["src_id"] == "HBM" and r["tgt_id"] == "DRAM"
    ]
    assert len(related) == 1, "Expected exactly one HBM→DRAM edge"
    assert "related_to" in related[0]["keywords"]

    related2 = [
        r for r in payload["relationships"]
        if r["src_id"] == "HBM" and r["tgt_id"] == "memory bandwidth"
    ]
    assert len(related2) == 1, "Expected exactly one HBM→memory bandwidth edge"
    assert "related_to" in related2[0]["keywords"]


def test_selected_in_edge_emitted():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    edges = [(r["src_id"], r["tgt_id"]) for r in payload["relationships"]]
    report_name = f"{CONFIG['profile_id']}-{CONFIG['current_date']}"
    assert ("Advances in HBM4 Memory Architecture", report_name) in edges


def test_article_entity_created():
    payload = _build_payload([], [ARTICLE], CONFIG, SOURCE_ID)
    names = [e["entity_name"] for e in payload["entities"]]
    assert "EE Times: CXL 3.0 Gains Momentum" in names


def test_article_publication_entity_created():
    payload = _build_payload([], [ARTICLE], CONFIG, SOURCE_ID)
    types = {e["entity_name"]: e["entity_type"] for e in payload["entities"]}
    assert types.get("EE Times") == "Publication"


def test_article_covers_topic_edge():
    payload = _build_payload([], [ARTICLE], CONFIG, SOURCE_ID)
    edges = [(r["src_id"], r["tgt_id"]) for r in payload["relationships"]]
    assert ("EE Times: CXL 3.0 Gains Momentum", "memory technology") in edges


def test_empty_input_returns_empty_payload():
    payload = _build_payload([], [], CONFIG, SOURCE_ID)
    assert payload["entities"] == []
    assert payload["relationships"] == []
    assert payload["chunks"] == []


def test_paper_chunk_includes_title_and_url():
    payload = _build_payload([PAPER], [], CONFIG, SOURCE_ID)
    assert len(payload["chunks"]) >= 1
    paper_chunk = next(c for c in payload["chunks"] if "Advances in HBM4" in c["content"])
    assert "https://arxiv.org/abs/2605.19972v1" in paper_chunk["content"]


def test_paper_without_title_is_skipped():
    bad_paper = {**PAPER, "title": ""}
    payload = _build_payload([bad_paper], [], CONFIG, SOURCE_ID)
    assert payload["entities"] == []


def test_article_without_title_is_skipped():
    bad_article = {**ARTICLE, "title": "   "}
    payload = _build_payload([], [bad_article], CONFIG, SOURCE_ID)
    assert payload["entities"] == []


PAPER_WITH_AUTHORS = {
    **PAPER,
    "author_affiliations": {
        "Alice Smith": ["Samsung Electronics"],
        "Bob Lee": ["KAIST"],
    },
}


def test_author_entities_created():
    payload = _build_payload([PAPER_WITH_AUTHORS], [], CONFIG, SOURCE_ID)
    types = {e["entity_name"]: e["entity_type"] for e in payload["entities"]}
    assert types.get("Alice Smith") == "Author"
    assert types.get("Bob Lee") == "Author"


def test_authored_by_edge_emitted():
    payload = _build_payload([PAPER_WITH_AUTHORS], [], CONFIG, SOURCE_ID)
    edges = [(r["src_id"], r["tgt_id"], r["keywords"]) for r in payload["relationships"]]
    assert ("Advances in HBM4 Memory Architecture", "Alice Smith", "authored_by author") in edges
    assert ("Advances in HBM4 Memory Architecture", "Bob Lee", "authored_by author") in edges


def test_affiliated_with_edge_emitted():
    payload = _build_payload([PAPER_WITH_AUTHORS], [], CONFIG, SOURCE_ID)
    edges = [(r["src_id"], r["tgt_id"], r["keywords"]) for r in payload["relationships"]]
    assert ("Alice Smith", "Samsung Electronics", "affiliated_with institution") in edges
    assert ("Bob Lee", "KAIST", "affiliated_with institution") in edges


def test_author_deduplication():
    paper2 = {**PAPER_WITH_AUTHORS, "title": "Another HBM Paper", "raw_id": "https://arxiv.org/abs/2605.99999v1"}
    payload = _build_payload([PAPER_WITH_AUTHORS, paper2], [], CONFIG, SOURCE_ID)
    alice_count = sum(1 for e in payload["entities"] if e["entity_name"] == "Alice Smith")
    assert alice_count == 1


def test_missing_author_affiliations_no_crash():
    paper = {**PAPER}  # no author_affiliations key
    payload = _build_payload([paper], [], CONFIG, SOURCE_ID)
    types = [e["entity_type"] for e in payload["entities"]]
    assert "Author" not in types


def test_resolve_base_url_prefers_explicit_value_over_environment(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_BASE_URL", "http://environment:9621")
    assert gi._resolve_base_url("http://explicit:9624") == "http://explicit:9624"


def test_resolve_base_url_uses_environment_then_memory_default(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_BASE_URL", "http://environment:9621")
    assert gi._resolve_base_url(None) == "http://environment:9621"

    monkeypatch.delenv("LIGHTRAG_BASE_URL")
    assert gi._resolve_base_url(None) == "http://localhost:9621"


def test_ingest_report_posts_entities_and_relations_to_custom_target(monkeypatch):
    posts = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def post(self, url, json, timeout):
            posts.append(url)
            return Response()

    monkeypatch.setattr(gi.requests, "Session", Session)

    assert gi.ingest_report([PAPER], [], CONFIG, base_url="http://storage:9624") is True
    assert posts
    assert all(url.startswith("http://storage:9624/") for url in posts)
    assert "http://storage:9624/graph/entity/create" in posts
    assert "http://storage:9624/graph/relation/create" in posts

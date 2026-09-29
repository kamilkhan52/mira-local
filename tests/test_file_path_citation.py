from mira.graph_ingest import _build_payload

CONFIG = {"profile_id": "memory-innovation", "current_date": "2026-03-01", "topic": {"focus": "memory"}}
PAPER = {"title": "Test Paper", "raw_id": "http://arxiv.org/abs/2603.12345v1", "primary_topic": "HBM"}
ARTICLE = {"title": "Some Article", "source": "EE Times", "url": "https://eetimes.com/x", "summary": "s"}


def test_paper_entity_file_path_is_arxiv_url():
    payload = _build_payload([PAPER], [], CONFIG, "src")
    paper = next(e for e in payload["entities"] if e["entity_type"] == "Paper")
    assert paper["file_path"] == "http://arxiv.org/abs/2603.12345v1"


def test_paper_chunk_file_path_is_arxiv_url():
    payload = _build_payload([PAPER], [], CONFIG, "src")
    chunk = payload["chunks"][0]
    assert chunk["file_path"] == "http://arxiv.org/abs/2603.12345v1"


def test_article_entity_and_chunk_file_path_is_url():
    payload = _build_payload([], [ARTICLE], CONFIG, "src")
    art = next(e for e in payload["entities"] if e["entity_type"] == "Article")
    assert art["file_path"] == "https://eetimes.com/x"
    assert payload["chunks"][0]["file_path"] == "https://eetimes.com/x"


def test_non_source_entities_keep_default_file_path():
    payload = _build_payload([PAPER], [], CONFIG, "src")
    topic = next(e for e in payload["entities"] if e["entity_type"] == "Topic")
    assert topic["file_path"] == "custom_kg"

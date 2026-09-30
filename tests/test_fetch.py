# tests/test_fetch.py
import sys
import types
import pytest
from unittest.mock import patch, MagicMock
from mira.fetch import _build_url, _parse_xml, _deduplicate

SAMPLE_CONFIG = {
    "arxiv": {"categories": ["cs.AR", "cs.LG"], "max_results": 100},
    "start_date": "20260511",
    "end_date": "20260518",
}

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2605.11277v1</id>
    <title>Test Paper One</title>
    <summary>An abstract about memory.</summary>
    <published>2026-05-11T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.AR"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2605.11277v2</id>
    <title>Test Paper One</title>
    <summary>An abstract about memory.</summary>
    <published>2026-05-11T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <category term="cs.AR"/>
  </entry>
</feed>"""


def test_build_url_contains_categories():
    url = _build_url(SAMPLE_CONFIG)
    assert "cat:cs.AR" in url
    assert "cat:cs.LG" in url


def test_build_url_contains_date_range():
    url = _build_url(SAMPLE_CONFIG)
    assert "202605110000" in url
    assert "202605182359" in url


def test_build_url_contains_max_results():
    url = _build_url(SAMPLE_CONFIG)
    assert "max_results=100" in url


def test_parse_xml_extracts_all_fields():
    papers = _parse_xml(SAMPLE_XML)
    assert len(papers) == 2
    assert papers[0]["id"] == "2605.11277"
    assert papers[0]["title"] == "Test Paper One"
    assert papers[0]["summary"] == "An abstract about memory."
    assert papers[0]["published"] == "2026-05-11"
    assert "Alice Smith" in papers[0]["authors"]
    assert "Bob Jones" in papers[0]["authors"]
    assert "cs.AR" in papers[0]["categories"]
    assert papers[0]["first_page_text"] == ""


def test_deduplicate_strips_version_duplicates():
    papers = _parse_xml(SAMPLE_XML)
    unique = _deduplicate(papers)
    assert len(unique) == 1
    assert unique[0]["id"] == "2605.11277"


from mira.fetch import extract_first_pages


def test_extract_first_pages_attaches_text(monkeypatch):
    mock_module = MagicMock()
    mock_results = [{"id": "2605.11277", "success": True, "first_page_text": "Authors: Alice"}]
    monkeypatch.setitem(sys.modules, "extract_arxiv_pdf", mock_module)
    # extract_first_pages now drives its own event loop (explicit executor);
    # patch run_until_complete instead of asyncio.run
    monkeypatch.setattr("mira.fetch.asyncio", types.SimpleNamespace(
        new_event_loop=lambda: MagicMock(run_until_complete=MagicMock(return_value=mock_results)),
        set_event_loop=lambda *_: None))
    papers = [{"id": "2605.11277", "title": "Test", "first_page_text": ""}]
    result = extract_first_pages(papers)
    assert result[0]["first_page_text"] == "Authors: Alice"


def test_extract_first_pages_uses_n8n_extractor_with_cache_and_32_workers(monkeypatch):
    """Uses n8n's extract_arxiv_first_page (PDF library + first-page cache) and
    gives the loop a 32-worker default pool: its run_in_executor(None, ...)
    would otherwise cap at min(32, cpu+4)."""
    import asyncio
    import mira.fetch as F
    captured = {}
    mock_module = MagicMock()

    def fake_batch(ids, concurrency, use_cache=True):
        loop = asyncio.get_event_loop()
        captured.update({"ids": ids, "concurrency": concurrency, "use_cache": use_cache,
                         "workers": loop._default_executor._max_workers})
        return []
    mock_module.process_batch = fake_batch
    monkeypatch.setitem(sys.modules, "extract_arxiv_first_page", mock_module)
    papers = [{"id": "2605.11277", "title": "Test", "first_page_text": ""}]
    extract_first_pages(papers)
    assert captured["concurrency"] == F.PDF_CONCURRENCY == 32
    assert captured["workers"] == 32
    assert captured["use_cache"] is True
    assert captured["ids"] == ["2605.11277"]


def test_extract_first_pages_empty_list_returns_empty():
    assert extract_first_pages([]) == []


def test_extract_first_pages_failed_pdf_leaves_empty(monkeypatch):
    mock_module = MagicMock()
    mock_results = [{"id": "9999.00000", "success": False, "first_page_text": ""}]
    monkeypatch.setitem(sys.modules, "extract_arxiv_first_page", mock_module)
    monkeypatch.setattr("mira.fetch.asyncio.run", lambda _: mock_results)
    papers = [{"id": "9999.00000", "title": "Test", "first_page_text": ""}]
    result = extract_first_pages(papers)
    assert result[0]["first_page_text"] == ""

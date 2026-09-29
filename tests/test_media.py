# tests/test_media.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from mira.media import _run_crawler, _normalize_articles


SAMPLE_CRAWLER_OUTPUT = json.dumps([
    {
        "url": "https://www.eetimes.com/hbm4-supply/",
        "title": "HBM4 Supply Tightens",
        "siteName": "EE Times",
        "content": "Memory supply is constrained as HBM4 ramp begins.",
        "listDate": "May 18, 2026",
        "crawledAt": "2026-05-18T10:00:00.000Z",
    }
])


def test_normalize_articles_maps_fields():
    raw = json.loads(SAMPLE_CRAWLER_OUTPUT)
    articles = _normalize_articles(raw, "EE Times")
    assert articles[0]["source"] == "EE Times"
    assert articles[0]["title"] == "HBM4 Supply Tightens"
    assert articles[0]["url"] == "https://www.eetimes.com/hbm4-supply/"
    assert "Memory supply" in articles[0]["content"]


def test_run_crawler_returns_empty_on_subprocess_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("mira.media.ROOT", tmp_path)
    with patch("mira.media.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        result = _run_crawler("ee-times-crawler.ts", {
            "start_date_iso": "2026-05-11",
            "end_date_iso": "2026-05-18",
        }, "eetimes")
    assert result == []


def test_run_crawler_reads_output_file(tmp_path, monkeypatch):
    monkeypatch.setattr("mira.media.ROOT", tmp_path)
    (tmp_path / "scripts" / "temp").mkdir(parents=True)
    output_file = tmp_path / "scripts" / "temp" / "eetimes-latest.json"
    output_file.write_text(SAMPLE_CRAWLER_OUTPUT)

    with patch("mira.media.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = _run_crawler("ee-times-crawler.ts", {
            "start_date_iso": "2026-05-11",
            "end_date_iso": "2026-05-18",
        }, "eetimes")
    assert len(result) == 1
    assert result[0]["title"] == "HBM4 Supply Tightens"


def test_normalize_articles_listtitle_fallback_and_date():
    raw = [
        {
            "url": "https://example.com/article",
            "listTitle": "Fallback Title",
            "content": "Content here.",
            "crawledAt": "2026-05-18T12:00:00.000Z",
            # no "title", no "listDate"
        }
    ]
    articles = _normalize_articles(raw, "SemiAnalysis")
    assert len(articles) == 1
    assert articles[0]["title"] == "Fallback Title"
    assert articles[0]["date"] == "2026-05-18"

def test_normalize_articles_filters_no_title():
    raw = [{"url": "https://example.com", "content": "no title here"}]
    articles = _normalize_articles(raw, "TrendForce")
    assert articles == []

# tests/test_port_front_fetch.py — Query arXiv port (URL + retries), no network
import pytest
import requests

import mira.fetch as fetch
from mira.fetch import _build_url, fetch_papers

BASE = {"arxiv": {"categories": ["cs.AR", "cs.ET"], "keywords": [], "max_results": 2000},
        "start_date": "20260901", "end_date": "20260928"}


def test_url_without_keywords_matches_n8n():
    assert _build_url(BASE) == (
        "https://export.arxiv.org/api/query?search_query=(cat:cs.AR+OR+cat:cs.ET)"
        "+AND+submittedDate:[202609010000+TO+202609282359]&start=0&max_results=2000")


def test_keyword_clause_quotes_and_encodes_like_encodeURIComponent():
    cfg = {**BASE, "arxiv": {**BASE["arxiv"], "keywords": ["HBM", "processing in memory", "3D-DRAM"]}}
    url = _build_url(cfg)
    assert ("(cat:cs.AR+OR+cat:cs.ET)+AND+(ti:HBM+OR+ti:%22processing%20in%20memory%22+OR+ti:3D-DRAM"
            "+OR+abs:HBM+OR+abs:%22processing%20in%20memory%22+OR+abs:3D-DRAM)"
            "+AND+submittedDate:") in url


def test_max_limit_override_wins_over_profile_max_results():
    assert _build_url({**BASE, "max_limit": 25}).endswith("&max_results=25")
    assert _build_url({**BASE, "max_limit": None}).endswith("&max_results=2000")
    no_max = {**BASE, "arxiv": {"categories": ["cs.AR"]}}
    assert _build_url(no_max).endswith("&max_results=2000")


class _Resp:
    def __init__(self, status, text=""):
        self.status_code, self.text = status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


FEED = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/2609.00001v1</id><title>A</title><summary>S</summary>
<published>2026-09-02T00:00:00Z</published><author><name>X</name></author></entry>
<entry><id>http://arxiv.org/abs/2609.00001v1</id><title>A</title><summary>S</summary>
<published>2026-09-02T00:00:00Z</published></entry></feed>"""


@pytest.fixture(autouse=True)
def _no_query_cache(monkeypatch):
    monkeypatch.setattr(fetch, "ARXIV_CACHE_HOURS", 0)


def test_fetch_retries_transient_errors_then_succeeds(monkeypatch):
    calls, sleeps = [], []
    responses = [requests.Timeout("slow"), _Resp(503), requests.ConnectionError("reset"), _Resp(200, FEED)]

    def fake_get(url, headers, timeout):
        calls.append(url)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    monkeypatch.setattr(fetch.time, "sleep", sleeps.append)
    papers = fetch_papers(BASE)
    assert len(calls) == 4 and sleeps == [30.0, 60.0, 120.0]
    assert [p["raw_id"] for p in papers] == ["http://arxiv.org/abs/2609.00001v1"]  # deduplicated


def test_fetch_gives_up_after_attempts(monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Resp(503))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="after 6 attempts"):
        fetch_papers(BASE)


def test_fetch_does_not_retry_a_bad_request(monkeypatch):
    calls = []
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: calls.append(1) or _Resp(400))
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    with pytest.raises(requests.HTTPError):
        fetch_papers(BASE)
    assert len(calls) == 1


def test_identical_query_reuses_saved_response(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch, "ARXIV_CACHE_HOURS", 12)
    monkeypatch.setattr(fetch, "LOCAL_CACHE", tmp_path)
    calls = []
    monkeypatch.setattr(fetch.requests, "get", lambda url, headers, timeout: calls.append(url) or _Resp(200, FEED))
    first, second = fetch_papers(BASE), fetch_papers(BASE)
    assert len(calls) == 1 and first == second

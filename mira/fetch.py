from __future__ import annotations
from mira.paths import TEMP_DIR, SCRIPTS_DIR
import asyncio
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).parent.parent
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
# arXiv asks API clients to identify themselves; set MIRA_CONTACT_EMAIL.
_HEADERS = {"User-Agent": "mira-local/1.0" + (f" (mailto:{os.environ['MIRA_CONTACT_EMAIL']})" if os.environ.get("MIRA_CONTACT_EMAIL") else "")}
# Match the n8n "Extract PDFs (Batch)" node's --concurrency 32 flag. NOTE: n8n runs
# scripts/extract_arxiv_first_page.py while the CLI imports extract_arxiv_pdf.py —
# only the concurrency value is being matched, not the script. Also note the CLI's
# arXiv ids are version-stripped (2608.11840) while n8n's keep the version
# (2608.11840v1) — pipeline.py's cache lookups account for both forms.
PDF_CONCURRENCY = 32

# n8n's Query arXiv node has no retry; a transient arXiv 503 used to kill the
# run. Four attempts, 5 s doubling backoff (5, 10, 20 s).
FETCH_ATTEMPTS = 4
FETCH_BACKOFF_SECONDS = 5.0
_RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
# Characters encodeURIComponent leaves unescaped.
_URI_COMPONENT_SAFE = "-_.!~*'()"


def _keyword_query(keywords: list[str]) -> str:
    """Query arXiv's keyword clause: ti:/abs: terms, multi-word keywords in
    double quotes, each term encodeURIComponent-escaped, all OR-joined."""
    def wrap(k: str) -> str:
        return f'"{k}"' if " " in k else k
    ti = [f"ti:{quote(wrap(k), safe=_URI_COMPONENT_SAFE)}" for k in keywords]
    ab = [f"abs:{quote(wrap(k), safe=_URI_COMPONENT_SAFE)}" for k in keywords]
    return "+OR+".join(ti + ab)


def _build_url(config: dict) -> str:
    """Port of the n8n "Query arXiv" URL expression. Category clause, optional
    keyword clause (arxiv.keywords), submittedDate window, and max_results =
    max_limit override ?? arxiv.max_results ?? 2000."""
    arxiv = config.get("arxiv") or {}
    cats = arxiv.get("categories") or []
    keywords = arxiv.get("keywords") or []
    cat_query = "+OR+".join(f"cat:{c}" for c in cats)
    # n8n quirk (reproduced): with no categories the query starts with
    # "+AND+..."; every shipped profile has categories.
    topic_query = f"({cat_query})" if cat_query else ""
    keyword_part = f"+AND+({_keyword_query(keywords)})" if keywords else ""
    start = config["start_date"]
    end = config["end_date"]
    max_limit = config.get("max_limit")
    max_results = max_limit if max_limit is not None else arxiv.get("max_results", 2000)
    if max_results is None:
        max_results = 2000
    return (
        f"https://export.arxiv.org/api/query"
        f"?search_query={topic_query}{keyword_part}"
        f"+AND+submittedDate:[{start}0000+TO+{end}2359]"
        f"&start=0&max_results={max_results}"
    )


def _parse_xml(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise RuntimeError(f"Failed to parse arXiv XML response: {e}") from e
    papers = []
    for entry in root.findall("atom:entry", _NS):
        raw_id = entry.findtext("atom:id", "", _NS)
        arxiv_id = raw_id.split("/abs/")[-1].split("v")[0]
        # Verbatim like n8n's "Convert XML to JSON" (default options): ends
        # trimmed, internal newlines kept. Flattening them changed the prompts
        # and the per-paper cache fingerprints for multi-paragraph abstracts.
        title = (entry.findtext("atom:title", "", _NS) or "").strip()
        summary = (entry.findtext("atom:summary", "", _NS) or "").strip()
        published = (entry.findtext("atom:published", "", _NS) or "")[:10]
        authors = [
            a.findtext("atom:name", "", _NS)
            for a in entry.findall("atom:author", _NS)
        ]
        categories = [c.get("term", "") for c in entry.findall("atom:category", _NS)]
        papers.append({
            "id": arxiv_id,
            # Verbatim Atom <id> (e.g. https://arxiv.org/abs/2608.11840v1). n8n's
            # cache input-hash uses the FULL url id — keep it so CLI-written cache
            # fingerprints match n8n's byte-for-byte (PR #30 review round 1).
            "raw_id": raw_id,
            "title": title,
            "summary": summary,
            "published": published,
            "authors": authors,
            "categories": categories,
            "first_page_text": "",
        })
    return papers


def _deduplicate(papers: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for p in papers:
        if p["id"] not in seen:
            seen.add(p["id"])
            result.append(p)
    return result


def _get_with_retries(url: str, *, attempts: int = FETCH_ATTEMPTS,
                      backoff: float = FETCH_BACKOFF_SECONDS, timeout: float = 300) -> requests.Response:
    """GET with exponential backoff on timeouts, connection errors, 5xx, 408
    and 429 (arXiv throttles with 503/429). Other 4xx fail immediately."""
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            if resp.status_code in _RETRY_STATUSES:
                raise requests.HTTPError(f"HTTP {resp.status_code} from arXiv", response=resp)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and status not in _RETRY_STATUSES:
                raise
            last_err = e
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
        if attempt < attempts - 1:
            wait = backoff * (2 ** attempt)
            print(f"  WARNING: arXiv request failed ({last_err}); retrying in {wait:.0f}s "
                  f"[{attempt + 1}/{attempts}]")
            time.sleep(wait)
    raise RuntimeError(f"arXiv query failed after {attempts} attempts: {last_err}")


def fetch_papers(config: dict) -> list[dict]:
    url = _build_url(config)
    resp = _get_with_retries(url)
    papers = _parse_xml(resp.text)
    return _deduplicate(papers)


def extract_first_pages(papers: list[dict]) -> list[dict]:
    if not papers:
        return papers

    (TEMP_DIR).mkdir(parents=True, exist_ok=True)
    scripts_dir = str(SCRIPTS_DIR)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import extract_arxiv_pdf as m  # type: ignore
    arxiv_ids = [p["id"] for p in papers]
    try:
        # process_batch dispatches sync downloads via run_in_executor(None, ...),
        # whose default pool is min(32, cpu+4) — 12 on an 8-core Mac — so the
        # semaphore's 32 would never be reached without an explicit pool.
        executor = ThreadPoolExecutor(max_workers=PDF_CONCURRENCY)
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                m.process_batch(arxiv_ids, PDF_CONCURRENCY, executor=executor)
            )
        finally:
            executor.shutdown(wait=False)
            loop.close()
            asyncio.set_event_loop(None)
    except Exception as e:
        print(f"  WARNING: PDF extraction failed — {e}. Continuing with empty first_page_text.")
        return papers

    by_id = {r["id"]: r for r in results}
    for paper in papers:
        paper["first_page_text"] = by_id.get(paper["id"], {}).get("first_page_text", "")

    return papers

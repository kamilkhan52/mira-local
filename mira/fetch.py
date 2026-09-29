from __future__ import annotations
from mira.paths import TEMP_DIR, SCRIPTS_DIR
import asyncio
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_HEADERS = {"User-Agent": "mira-local-script/1.0 (mailto:oed8205@gmail.com)"}
# Match the n8n "Extract PDFs (Batch)" node's --concurrency 32 flag. NOTE: n8n runs
# scripts/extract_arxiv_first_page.py while the CLI imports extract_arxiv_pdf.py —
# only the concurrency value is being matched, not the script. Also note the CLI's
# arXiv ids are version-stripped (2608.11840) while n8n's keep the version
# (2608.11840v1) — pipeline.py's cache lookups account for both forms.
PDF_CONCURRENCY = 32


def _build_url(config: dict) -> str:
    cats = config["arxiv"]["categories"]
    cat_query = "+OR+".join(f"cat:{c}" for c in cats)
    start = config["start_date"]
    end = config["end_date"]
    max_results = config["arxiv"].get("max_results", 2000)
    return (
        f"https://export.arxiv.org/api/query"
        f"?search_query=({cat_query})"
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
        title = (entry.findtext("atom:title", "", _NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", "", _NS) or "").strip().replace("\n", " ")
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


def fetch_papers(config: dict) -> list[dict]:
    url = _build_url(config)
    resp = requests.get(url, headers=_HEADERS, timeout=300)
    resp.raise_for_status()
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

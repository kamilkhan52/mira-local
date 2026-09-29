#!/usr/bin/env python3
"""Build the local venue corpus for novelty checking (DBLP + Crossref + OpenAlex).

    python3 scripts/fetch_venue_corpus.py --resolve "memory systems"   # find DBLP stream keys
    python3 scripts/fetch_venue_corpus.py            # build from configs/venue-corpus.json

Paper lists (title/year/DOI) come from DBLP venue streams for venues DBLP
indexes, and from Crossref (by exact conference container-title) for venues it
does not (IEDM/ISSCC/VLSI, ECTC). OpenAlex's own conference sources went stale
around 2021, so it is used only to hydrate abstracts by DOI.
Set OPENALEX_MAILTO (and optionally CROSSREF_MAILTO) to use the polite pools.
Specs: docs/superpowers/specs/2026-07-14-venue-scoped-novelty-design.md,
docs/superpowers/specs/2026-07-21-crossref-venue-fetch-design.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mira.venue_corpus import build_db, reconstruct_abstract  # noqa: E402

DBLP = "https://dblp.org"
OPENALEX = "https://api.openalex.org"
CROSSREF = "https://api.crossref.org"
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "venue-corpus.json"
DB = ROOT / "cache" / "venue_corpus.sqlite"
DBLP_PACING = 5.0     # seconds between DBLP requests — DBLP throttles bursts hard
OPENALEX_PACING = 0.25
CROSSREF_PACING = 1.0
DBLP_PAGE = 1000      # DBLP search API maximum page size
CROSSREF_PAGE = 1000  # Crossref /works maximum rows per page
OPENALEX_BATCH = 50   # max values per OpenAlex |-joined doi filter
HEADERS = {"User-Agent": "mira-venue-corpus/1.0"}
PAPER_TYPES = {"Conference and Workshop Papers", "Journal Articles"}


def _get(url: str, params: dict, *, pacing: float, tries: int = 6) -> dict:
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
        except (requests.ConnectionError, requests.Timeout):
            # DBLP drops or stalls connections when throttling; back off and retry
            time.sleep(30 * (attempt + 1))
            continue
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", 30 * (attempt + 1))))
            continue
        resp.raise_for_status()
        time.sleep(pacing)
        return resp.json()
    raise RuntimeError(f"still rate-limited after {tries} attempts: {url}")


def resolve(name: str) -> None:
    data = _get(f"{DBLP}/search/venue/api",
                {"q": name, "format": "json", "h": 20}, pacing=DBLP_PACING)
    for hit in (data["result"]["hits"].get("hit") or []):
        info = hit["info"]
        url = info.get("url", "")
        # stream key is the path under dblp.org/db/, e.g. conf/isca
        stream = url.split("/db/", 1)[-1].rstrip("/").removesuffix("/index.html")
        print(f'{stream:<24} {info.get("venue", "")}')


def fetch_stream(stream: str, year: int) -> list[dict]:
    hits, first = [], 0
    while True:
        data = _get(f"{DBLP}/search/publ/api", {
            "q": f"streamid:{stream}: year:{year}",
            "format": "json", "h": DBLP_PAGE, "f": first}, pacing=DBLP_PACING)
        result = data["result"]["hits"]
        page = result.get("hit") or []
        hits.extend(h["info"] for h in page)
        first += len(page)
        if not page or first >= int(result.get("@total", 0)):
            return hits


def dblp_rows(config: dict, since_year: int) -> list[dict]:
    rows, seen = [], set()
    for venue in config["venues"]:
        if not venue.get("dblp_stream"):
            continue  # Crossref-only venue; handled by crossref_rows
        count = 0
        for year in range(since_year, date.today().year + 1):
            for info in fetch_stream(venue["dblp_stream"], year):
                if info.get("type") not in PAPER_TYPES:
                    continue  # skip Editorship (proceedings frontmatter) etc.
                key = info.get("key") or ""
                title = (info.get("title") or "").rstrip(".").strip()
                if not key or not title or key in seen:
                    continue
                seen.add(key)
                doi = (info.get("doi") or "").lower()
                rows.append({
                    "id": f"dblp:{key}", "title": title, "abstract": "",
                    "year": int(info["year"]), "venue": venue["name"],
                    "url": f"https://doi.org/{doi}" if doi else info.get("ee") or "",
                    "doi": doi,
                })
                count += 1
        print(f"  {venue['name']}: {count} papers")
    return rows


def ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 73 -> '73rd'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _format_container(container: str, year: int, ordinal_base: int | None) -> str:
    """Fill a container-title template. `{ordinal}` without an ordinal_base
    raises KeyError (str.format lacks the key) — a config error, surfaced loud."""
    values = {"year": year}
    if ordinal_base is not None:
        values["ordinal"] = ordinal(year - ordinal_base)
    return container.format(**values)


def fetch_crossref_year(containers: list[str], year: int,
                        ordinal_base: int | None = None) -> list[dict]:
    """All Crossref proceedings works whose container-title exactly matches any
    of `containers` (formatted for `year`), cursor-paginated. Returns raw works."""
    titles = [_format_container(c, year, ordinal_base) for c in containers]
    filt = ",".join(f"container-title:{t}" for t in titles) + ",type:proceedings-article"
    items, cursor = [], "*"
    while cursor:
        params = {"filter": filt, "rows": CROSSREF_PAGE, "cursor": cursor,
                  "select": "DOI,title,container-title"}
        mailto = os.environ.get("CROSSREF_MAILTO") or os.environ.get("OPENALEX_MAILTO")
        if mailto:
            params["mailto"] = mailto
        data = _get(f"{CROSSREF}/works", params, pacing=CROSSREF_PACING)
        page = data["message"]["items"]
        items.extend(page)
        cursor = data["message"].get("next-cursor") if page else None
    return items


def crossref_rows(config: dict, since_year: int) -> list[dict]:
    rows = []
    for venue in config["venues"]:
        containers = venue.get("crossref_containers")
        if not containers:
            continue  # DBLP venue; handled by dblp_rows
        count = 0
        for year in range(since_year, date.today().year + 1):
            for work in fetch_crossref_year(containers, year,
                                            venue.get("ordinal_base")):
                doi = (work.get("DOI") or "").lower()
                titles = work.get("title") or []
                if not doi or not titles:
                    continue  # a proceedings entry without a DOI/title can't index
                rows.append({
                    "id": f"crossref:{doi}",
                    "title": " ".join(titles[0].split()).rstrip("."),
                    "abstract": "",
                    "year": year,
                    "venue": venue["name"],
                    "url": f"https://doi.org/{doi}",
                    "doi": doi,
                })
                count += 1
        print(f"  {venue['name']}: {count} papers")
    return rows


def dedup_by_doi(rows: list[dict]) -> list[dict]:
    """Drop later rows sharing a DOI (keep first — DBLP before Crossref).
    DOI-less rows are never merged: an empty DOI is not an identity."""
    out, seen = [], set()
    for row in rows:
        doi = row.get("doi") or ""
        if doi and doi in seen:
            continue
        if doi:
            seen.add(doi)
        out.append(row)
    return out


def unresolved_venues(config: dict) -> list[str]:
    """Names of venues with neither a dblp_stream nor crossref_containers."""
    return [v["name"] for v in config["venues"]
            if not v.get("dblp_stream") and not v.get("crossref_containers")]


def novelty_cache_for(venue_db: Path) -> Path:
    """The per-corpus novelty cache path beside a venue DB, matching
    mira.graph_target naming (venue_corpus[_x].sqlite -> venue_novelty[_x].json)."""
    return venue_db.with_name(
        venue_db.name.replace("venue_corpus", "venue_novelty").replace(".sqlite", ".json"))


def hydrate_abstracts(rows: list[dict]) -> int:
    """Fill row['abstract'] from OpenAlex by DOI, in batches. Best-effort."""
    by_doi = {r["doi"]: r for r in rows if r["doi"]}
    dois = list(by_doi)
    hydrated = 0
    for i in range(0, len(dois), OPENALEX_BATCH):
        batch = dois[i:i + OPENALEX_BATCH]
        params = {"filter": "doi:" + "|".join(batch),
                  "select": "doi,abstract_inverted_index",
                  "per-page": len(batch)}
        if os.environ.get("OPENALEX_MAILTO"):
            params["mailto"] = os.environ["OPENALEX_MAILTO"]
        try:
            data = _get(f"{OPENALEX}/works", params, pacing=OPENALEX_PACING)
        except (requests.RequestException, RuntimeError):
            continue  # abstracts are an enrichment; titles alone still index
        for work in data.get("results", []):
            doi = (work.get("doi") or "").removeprefix("https://doi.org/").lower()
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if doi in by_doi and abstract:
                by_doi[doi]["abstract"] = abstract
                hydrated += 1
    return hydrated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the local venue corpus from DBLP + OpenAlex")
    parser.add_argument("--resolve", metavar="NAME",
                        help="look up DBLP stream keys for a venue name and exit")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--db", type=Path, default=DB)
    args = parser.parse_args()

    if args.resolve:
        resolve(args.resolve)
        return 0

    config = json.loads(args.config.read_text())
    unresolved = unresolved_venues(config)
    if unresolved:
        print(f"ERROR: venues in {args.config} have neither dblp_stream nor "
              f"crossref_containers: {', '.join(unresolved)}\n"
              f"Run: python3 scripts/fetch_venue_corpus.py --resolve '<venue name>' "
              f"and paste a dblp_stream, or add crossref_containers.", file=sys.stderr)
        return 1

    since_year = date.today().year - config["years_back"]
    rows = dedup_by_doi(dblp_rows(config, since_year)
                        + crossref_rows(config, since_year))
    print(f"Hydrating abstracts from OpenAlex for {len(rows)} papers ...")
    hydrated = hydrate_abstracts(rows)
    print(f"  {hydrated} abstracts found")
    for row in rows:
        row.pop("doi")
    build_db(rows, args.db, years_back=config["years_back"],
             venues=[v["name"] for v in config["venues"]])
    # Per-pair novelty counts are only valid for the corpus they were computed
    # against; a rebuild must invalidate the cache.
    novelty_cache_for(args.db).unlink(missing_ok=True)
    print(f"Built {args.db} with {len(rows)} papers (years >= {since_year})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

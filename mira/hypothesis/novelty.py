"""External novelty check via an injectable fetch (venue corpus in production).

A local "gap" may simply be a paper MIRA never fetched (spec §10), so each
candidate pair is checked against recent external literature. Both CLIs inject
the local venue-corpus fetch (mira.venue_corpus.make_venue_fetch); the live
Semantic Scholar `_default_fetch` remains only as a standalone fallback.
Failures degrade to "unverified" — they never abort the run (spec §6).
"""
from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Callable

import requests

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
RECENT_YEARS = 3
DROP_THRESHOLD = 20
DEMOTE_THRESHOLD = 3
_REQUEST_INTERVAL = 1.1  # seconds — polite pacing for the shared public pool


def classify(hits: int | None) -> str:
    if hits is None:
        return "unverified"
    if hits >= DROP_THRESHOLD:
        return "dropped"
    if hits >= DEMOTE_THRESHOLD:
        return "sparsely explored"
    return "open gap"


def unverified_note(candidates) -> str | None:
    """Degraded-header line naming every pair whose novelty check failed
    (novelty_hits is None), so the reader knows which agenda rows' novelty
    column to distrust. None when all pairs verified."""
    failed = [c for c in candidates if c.novelty_hits is None]
    if not failed:
        return None
    pairs = "; ".join(f"{c.topic_a} × {c.topic_c}" for c in failed)
    plural = "pair" if len(failed) == 1 else "pairs"
    return (f"venue corpus could not verify {len(failed)} {plural} "
            f"(missing corpus or topic vocabulary) — novelty unverified: {pairs}")


def _default_fetch(query: str) -> list[dict]:
    headers = {}
    if os.environ.get("S2_API_KEY"):
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    try:
        resp = requests.get(
            S2_SEARCH_URL,
            params={"query": query, "fields": "title,year", "limit": 100},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("data") or []
    finally:
        time.sleep(_REQUEST_INTERVAL)


class NoveltyChecker:
    """Check external novelty through the injected `fetch` seam.

    The recency window is inclusive of the cutoff calendar year
    (today.year - RECENT_YEARS), deliberately erring toward counting more
    external work — over-counting demotes a gap; under-counting would
    over-claim novelty.
    """
    def __init__(
        self,
        cache_path: Path,
        fetch: Callable[[str], list[dict]] = _default_fetch,
        today: date | None = None,
    ):
        self.cache_path = Path(cache_path)
        self.fetch = fetch
        self.cutoff_year = (today or date.today()).year - RECENT_YEARS
        try:
            self.cache = json.loads(self.cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            self.cache = {}

    def check(self, topic_a: str, topic_c: str) -> tuple[int | None, list[str]]:
        key = f"{topic_a} || {topic_c}"
        if key not in self.cache:
            titles: dict[str, None] = {}  # insertion-ordered de-dup
            try:
                for query in (f'"{topic_a}" "{topic_c}"', f"{topic_a} {topic_c}"):
                    for rec in self.fetch(query):
                        year = rec.get("year")
                        if year and year >= self.cutoff_year and rec.get("title"):
                            titles.setdefault(rec["title"])
            except requests.RequestException:
                return None, []
            self.cache[key] = {"hits": len(titles), "titles": list(titles)[:3]}
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2))
        entry = self.cache[key]
        return entry["hits"], entry["titles"]

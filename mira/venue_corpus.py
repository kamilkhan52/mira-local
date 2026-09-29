"""Local venue corpus for novelty checking.

Papers from tracked top venues (configs/venue-corpus.json, fetched from
OpenAlex by scripts/fetch_venue_corpus.py) are stored in SQLite+FTS5.
Novelty for a topic pair is answered by a local full-text query instead of a
live Semantic Scholar call: `make_venue_fetch` returns a drop-in for
`NoveltyChecker`'s injectable `fetch` seam.

Spec: docs/superpowers/specs/2026-07-14-venue-scoped-novelty-design.md
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Callable

import requests

from mira.graph_ingest import normalize_topic

# Single words too generic to identify a topic side in a memory-systems venue
# corpus — a bare "Memory" phrase would match most papers and fabricate
# "dropped" labels. Kept only when a label yields nothing more specific.
GENERIC_PHRASES = {"memory", "storage", "systems", "computing", "hardware", "ai", "ml"}
STALE_AFTER_DAYS = 183

# Topic labels and paper abstracts rarely share exact vocabulary: a
# "compression" topic covers quantization/pruning papers (Oaken, ISCA'25,
# was missed this way and the pair over-claimed novelty). Each phrase also
# searches with these single-token substitutions. Keep the table tight —
# every entry widens recall for all pairs mentioning the term.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "compression": ("quantization", "pruning"),
    "eviction": ("offloading", "swapping"),
    "pooling": ("disaggregation",),
    "interconnect": ("fabric", "link"),
}


class VenueCorpusError(requests.RequestException):
    """Missing/broken corpus. Subclasses RequestException deliberately so
    NoveltyChecker's existing degrade path catches it unchanged."""


# Taxonomy labels pair a spelled-out name with its acronym via a spaced dash
# ("Microring Modulators - MRM"), which the corpus never writes inline — papers
# say "microring modulator". The tail must be split off so the base term is
# searched. Guard tightly: a single 2-6 char run with >=2 uppercase letters is
# an acronym; a multi-word tail (affiliation strings like "Technion - Israel
# Institute of Technology") is not, and the internal hyphen of "Mach-Zehnder"
# is untouched because the separator requires surrounding spaces.
_ACRONYM_TAIL = re.compile(r"^(?P<base>.+?)\s-\s(?P<acr>[A-Za-z0-9]{2,6})$")


def _split_acronym_tail(text: str) -> tuple[str, str | None]:
    m = _ACRONYM_TAIL.match(text)
    if m and sum(c.isupper() for c in m.group("acr")) >= 2:
        return m.group("base").strip(), m.group("acr")
    return text, None


def extract_phrases(label: str) -> list[str]:
    """Deterministic search phrases for one topic label (spec §4.3)."""
    text = label.strip()
    head, sep, tail = text.partition(":")
    if sep and tail.strip():
        text = tail.strip()
    text, acronym = _split_acronym_tail(text)
    text = normalize_topic(text)
    parts = [p.strip() for p in re.split(r"[&/,]", text)]
    phrases = [p for p in parts if len(p) >= 3]
    specific = [p for p in phrases if " " in p or p.lower() not in GENERIC_PHRASES]
    if specific:
        phrases = specific
    if not phrases:
        phrases = [normalize_topic(label.strip()) or label.strip()]
    if acronym and acronym not in phrases:
        phrases.append(acronym)
    return phrases


def expand_phrases(phrases: list[str]) -> list[str]:
    """Originals plus one-token synonym substitutions, order-preserving."""
    out = list(phrases)
    seen = {p.casefold() for p in out}
    for p in phrases:
        words = p.split()
        for i, w in enumerate(words):
            for syn in SYNONYMS.get(w.casefold(), ()):
                variant = " ".join(words[:i] + [syn] + words[i + 1:])
                if variant.casefold() not in seen:
                    seen.add(variant.casefold())
                    out.append(variant)
    return out


def parse_pair_query(query: str) -> tuple[str, str] | None:
    """Recover (topic_a, topic_c) from NoveltyChecker's quoted query form."""
    quoted = re.findall(r'"([^"]+)"', query)
    return (quoted[0], quoted[1]) if len(quoted) == 2 else None


def _side_expr(phrases: list[str]) -> str:
    return "(" + " OR ".join(
        '"' + p.replace('"', '""') + '"' for p in phrases) + ")"


def _fts_expr(side_a: list[str], side_c: list[str]) -> str:
    return f"{_side_expr(side_a)} AND {_side_expr(side_c)}"


def venue_corpus_is_queryable(db_path: Path) -> bool:
    """Whether the corpus supports the FTS/join query used by venue fetches."""
    db_path = Path(db_path)
    if not db_path.exists():
        return False
    con: sqlite3.Connection | None = None
    try:
        con = sqlite3.connect(db_path)
        con.execute(
            "SELECT p.title, p.year FROM papers_fts "
            "JOIN papers p ON p.rowid = papers_fts.rowid "
            "WHERE papers_fts MATCH ? LIMIT 1",
            (_fts_expr(
                ["mira schema probe side a"],
                ["mira schema probe side c"],
            ),),
        ).fetchone()
        return True
    except sqlite3.Error:
        return False
    finally:
        if con is not None:
            con.close()


def make_venue_fetch(db_path: Path) -> Callable[[str], list[dict]]:
    """Drop-in for NoveltyChecker's `fetch`: local FTS instead of live S2."""
    db_path = Path(db_path)

    def fetch(query: str) -> list[dict]:
        pair = parse_pair_query(query)
        if pair is None:
            # NoveltyChecker's second, unquoted query: topic boundaries are
            # unrecoverable and loose matching only adds noise; titles are
            # de-duplicated across both calls, so empty is harmless.
            return []
        if not db_path.exists():
            raise VenueCorpusError(
                f"venue corpus missing: {db_path} — run scripts/fetch_venue_corpus.py")
        sides = [expand_phrases(extract_phrases(pair[0])),
                 expand_phrases(extract_phrases(pair[1]))]
        con = sqlite3.connect(db_path)
        try:
            for topic, phrases in zip(pair, sides):
                present = con.execute(
                    "SELECT 1 FROM papers_fts WHERE papers_fts MATCH ? LIMIT 1",
                    (_side_expr(phrases),)).fetchone()
                if present is None:
                    # A side with zero standalone matches makes the pair
                    # unverifiable: 0 joint hits would say "open gap" about
                    # vocabulary the corpus simply never uses.
                    raise VenueCorpusError(
                        f"topic {topic!r} has no vocabulary match in venue corpus"
                        " — novelty unverifiable for this pair")
            rows = con.execute(
                "SELECT p.title, p.year FROM papers_fts "
                "JOIN papers p ON p.rowid = papers_fts.rowid "
                "WHERE papers_fts MATCH ?", (_fts_expr(*sides),)).fetchall()
        except sqlite3.Error as exc:
            raise VenueCorpusError(f"venue corpus query failed: {exc}") from exc
        finally:
            con.close()
        return [{"title": title, "year": year} for title, year in rows]

    # The vocabulary back-off ladder (side_ladder/NEAR) was lost in the
    # 2026-07-21 workspace wipe. This stub keeps callers honest by reporting
    # zero back-offs; restoring the ladder is tracked separately.
    backoffs: dict[str, str] = {}
    fetch.backoffs = backoffs
    return fetch


def make_multi_venue_fetch(db_paths: tuple[Path, ...]) -> Callable[[str], list[dict]]:
    """Search a union of venue corpora with union-level side validation."""
    paths = tuple(Path(path) for path in db_paths)
    notes: list[str] = []

    def record_unavailable(path: Path, reason: str) -> None:
        note = f"venue corpus unavailable ({path}): {reason}"
        if note not in notes:
            notes.append(note)

    def fetch(query: str) -> list[dict]:
        pair = parse_pair_query(query)
        if pair is None:
            return []
        sides = [expand_phrases(extract_phrases(pair[0])),
                 expand_phrases(extract_phrases(pair[1]))]
        side_present = [False, False]
        rows: list[tuple[str, int | None]] = []
        readable = 0
        for path in paths:
            if not path.exists():
                record_unavailable(path, "missing")
                continue
            con: sqlite3.Connection | None = None
            try:
                con = sqlite3.connect(path)
                found = [
                    con.execute(
                        "SELECT 1 FROM papers_fts "
                        "WHERE papers_fts MATCH ? LIMIT 1",
                        (_side_expr(phrases),),
                    ).fetchone() is not None
                    for phrases in sides
                ]
                corpus_rows = con.execute(
                    "SELECT p.title, p.year FROM papers_fts "
                    "JOIN papers p ON p.rowid = papers_fts.rowid "
                    "WHERE papers_fts MATCH ?",
                    (_fts_expr(*sides),),
                ).fetchall()
                for index, present in enumerate(found):
                    if present:
                        side_present[index] = True
                rows.extend(corpus_rows)
                readable += 1
            except sqlite3.Error as exc:
                record_unavailable(path, str(exc))
            finally:
                if con is not None:
                    con.close()
        if readable == 0:
            raise VenueCorpusError(
                "no readable venue corpus: "
                + ("; ".join(notes) or "(no database paths configured)")
                + " — run scripts/fetch_venue_corpus.py"
            )
        for topic, present in zip(pair, side_present):
            if not present:
                raise VenueCorpusError(
                    f"topic {topic!r} has no vocabulary match in venue corpora"
                    " — novelty unverifiable for this pair"
                )
        hits: dict[str, dict] = {}
        for title, year in rows:
            key = title.strip().casefold()
            current = hits.get(key)
            if current is None or (
                isinstance(year, int)
                and (not isinstance(current["year"], int)
                     or year > current["year"])
            ):
                hits[key] = {"title": title, "year": year}
        return list(hits.values())

    backoffs: dict[str, str] = {}
    fetch.backoffs = backoffs
    fetch.notes = notes
    return fetch


def reconstruct_abstract(inverted: dict | None) -> str:
    """Rebuild abstract text from OpenAlex's abstract_inverted_index."""
    if not inverted:
        return ""
    positions = [(pos, word) for word, poss in inverted.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))


def work_to_row(work: dict, venue: str) -> dict:
    """Shape one OpenAlex work into a papers-table row."""
    loc = work.get("primary_location") or {}
    return {
        "id": work.get("id") or "",
        "title": work.get("title") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "year": work.get("publication_year"),
        "venue": venue,
        "url": work.get("doi") or loc.get("landing_page_url") or work.get("id") or "",
    }


def build_db(rows: list[dict], db_path: Path, *, years_back: int,
             venues: list[str], today: date | None = None) -> None:
    """Full rebuild, atomically installed (write .tmp then os.replace)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.executescript(
            "CREATE TABLE papers(id TEXT PRIMARY KEY, title TEXT NOT NULL, "
            "abstract TEXT NOT NULL DEFAULT '', year INTEGER, venue TEXT, url TEXT);"
            "CREATE VIRTUAL TABLE papers_fts USING fts5(title, abstract, "
            "content='papers', content_rowid='rowid', "
            "tokenize='porter unicode61');"
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);")
        con.executemany(
            "INSERT OR REPLACE INTO papers(id, title, abstract, year, venue, url) "
            "VALUES(:id, :title, :abstract, :year, :venue, :url)", rows)
        con.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
        con.executemany("INSERT INTO meta(key, value) VALUES(?, ?)", [
            ("built_at", (today or date.today()).isoformat()),
            ("years_back", str(years_back)),
            ("venues", json.dumps(venues)),
        ])
        con.commit()
    finally:
        con.close()
    os.replace(tmp, db_path)


def read_built_at(db_path: Path) -> date | None:
    """built_at from the meta table; None when missing/unreadable."""
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(db_path)
        try:
            row = con.execute(
                "SELECT value FROM meta WHERE key = 'built_at'").fetchone()
        finally:
            con.close()
        return date.fromisoformat(row[0]) if row else None
    except (sqlite3.Error, ValueError):
        return None

"""Read-only, profile-scoped view over the LightRAG GraphML.

The graph is undirected; edge type lives in the `keywords` edge attribute
(written by mira/graph_ingest.py). Node `source_id` is 'UNKNOWN' in the store,
so profile scoping goes through Report node names ('{profile_id}-{YYYY-MM-DD}').
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import networkx as nx

from mira.graph_ingest import normalize_topic

KW_PRIMARY_TOPIC = "primary_topic topic"
KW_ALSO_COVERS = "also_covers topic"
KW_RELATED_TO = "related_to topic"
KW_RESEARCHES = "researches topic"
KW_AUTHORED_BY = "authored_by author"
KW_SELECTED_IN = "selected_in report"

PAPER_TOPIC_KEYWORDS = {KW_PRIMARY_TOPIC, KW_ALSO_COVERS}


def load_graph(path: Path) -> nx.Graph:
    return nx.read_graphml(path)


def report_date(report_name: str) -> date | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})$", report_name)
    return date.fromisoformat(m.group(1)) if m else None


def _edge_kw(g: nx.Graph, u: str, v: str) -> str:
    return g[u][v].get("keywords", "")


def _node_type(g: nx.Graph, n: str) -> str:
    return g.nodes[n].get("entity_type", "")


@dataclass
class Corpus:
    g: nx.Graph
    profile_ids: tuple[str, ...]
    papers: set[str]
    paper_dates: dict[str, date]
    paper_profiles: dict[str, set[str]]
    report_dates: list[date]
    topic_papers: dict[str, set[str]]
    topic_institutions: dict[str, set[str]]
    topic_authors: dict[str, set[str]]

    @classmethod
    def build(
        cls, g: nx.Graph, profile_ids: Iterable[str], since: date | None = None
    ) -> "Corpus":
        if isinstance(profile_ids, str):
            profile_ids = (profile_ids,)
        else:
            profile_ids = tuple(profile_ids)
        selected_profile_ids = set(profile_ids)
        reports = [
            n for n in g.nodes
            if (_node_type(g, n) == "Report" and report_date(n)
                and n[:-11] in selected_profile_ids)
        ]
        report_dates = sorted(report_date(r) for r in reports)

        papers: set[str] = set()
        paper_dates: dict[str, date] = {}
        paper_profiles: dict[str, set[str]] = {}
        for r in reports:
            d = report_date(r)
            if since and d < since:
                continue
            profile_id = r[:-11]
            for n in g.neighbors(r):
                if _node_type(g, n) == "Paper" and _edge_kw(g, r, n) == KW_SELECTED_IN:
                    papers.add(n)
                    paper_profiles.setdefault(n, set()).add(profile_id)
                    if n not in paper_dates or d < paper_dates[n]:
                        paper_dates[n] = d

        topic_papers: dict[str, set[str]] = {}
        topic_authors: dict[str, set[str]] = {}
        for p in papers:
            p_topics = [
                n for n in g.neighbors(p)
                if _node_type(g, n) == "Topic" and _edge_kw(g, p, n) in PAPER_TOPIC_KEYWORDS
            ]
            p_authors = {
                n for n in g.neighbors(p)
                if _node_type(g, n) == "Author" and _edge_kw(g, p, n) == KW_AUTHORED_BY
            }
            for t in p_topics:
                topic_papers.setdefault(t, set()).add(p)
                topic_authors.setdefault(t, set()).update(p_authors)

        topic_institutions = {
            t: {
                n for n in g.neighbors(t)
                if _node_type(g, n) == "Institution" and _edge_kw(g, t, n) == KW_RESEARCHES
            }
            for t in topic_papers
        }

        return cls(g, profile_ids, papers, paper_dates, paper_profiles, report_dates,
                   topic_papers, topic_institutions, topic_authors)

    def related_topics(self, topic: str) -> set[str]:
        if topic not in self.g:
            return set()
        return {
            n for n in self.g.neighbors(topic)
            if _node_type(self.g, n) == "Topic" and _edge_kw(self.g, topic, n) == KW_RELATED_TO
        }


def match_anchor_topics(seed: str, topics: Iterable[str]) -> list[str]:
    """Exact normalized match first; otherwise substring either way."""
    topics = list(topics)
    norm_seed = normalize_topic(seed).casefold()
    exact = [t for t in topics if normalize_topic(t).casefold() == norm_seed]
    if exact:
        return sorted(exact)
    return sorted(
        t for t in topics
        if norm_seed in t.casefold() or t.casefold() in norm_seed
    )

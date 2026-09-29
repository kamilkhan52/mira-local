# mira/graph_ingest.py
from __future__ import annotations

import os
import re
import time
from datetime import datetime

import requests

_DEFAULT_BASE_URL = "http://localhost:9621"


def _resolve_base_url(explicit: str | None) -> str:
    """Resolve an ingest target without changing memory's default endpoint."""
    return explicit or os.environ.get("LIGHTRAG_BASE_URL") or _DEFAULT_BASE_URL


def normalize_topic(topic: str) -> str:
    """Collapse a parenthetical qualifier into its canonical label.

    The classifier emits both bare and qualified forms of the same concept
    ("CXL" vs "CXL (Compute Express Link - memory pooling)"), which would create
    separate Topic nodes and fragment Leiden clustering. Stripping the trailing
    parenthetical merges them onto one node. Falls back to the original string if
    stripping would leave nothing (e.g. a topic that is only "(ReRAM)").
    """
    stripped = re.sub(r"\s*\([^)]*\)", "", topic).strip()
    return stripped or topic.strip()


def _build_payload(
    selected: list[dict],
    media: list[dict],
    config: dict,
    source_id: str,
) -> dict:
    profile_id = config["profile_id"]
    run_date = config["current_date"]
    profile_topic = config.get("topic", {}).get("focus", "")

    entities: dict[str, dict] = {}
    relationships: list[dict] = []
    chunks: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_entity(name: str, entity_type: str, description: str = "", file_path: str = "custom_kg") -> None:
        if name and name not in entities:
            entities[name] = {
                "entity_name": name,
                "entity_type": entity_type,
                "description": description or name,
                "source_id": source_id,
                # file_path is what LightRAG shows in its References section; point
                # Paper/Article nodes at their source URL so citations are clickable.
                "file_path": file_path,
            }

    def add_edge(src: str, tgt: str, description: str, keywords: str) -> None:
        key = (src, tgt, keywords)
        if src and tgt and key not in seen_edges:
            seen_edges.add(key)
            relationships.append({
                "src_id": src,
                "tgt_id": tgt,
                "description": description,
                "keywords": keywords,
                "weight": 1.0,
                "source_id": source_id,
            })

    report_name = f"{profile_id}-{run_date}"
    add_entity(report_name, "Report", f"profile_id: {profile_id} · run_date: {run_date}")

    for paper in selected:
        title = paper.get("title", "").strip()
        if not title:
            continue

        arxiv_url = paper.get("raw_id") or f"https://arxiv.org/abs/{paper.get('id', '')}"
        affiliations = [a.strip() for a in (paper.get("affiliations") or []) if a.strip()]
        primary_topic = normalize_topic((paper.get("primary_topic") or "").strip())
        secondary_topics = [
            normalize_topic(t.strip()) for t in (paper.get("secondary_topics") or []) if t.strip()
        ]
        relevance_score = paper.get("relevance_score", "")
        credibility_tier = paper.get("credibility_tier", "")
        key_findings = paper.get("key_findings", "")
        short_summary = paper.get("short_summary", "")

        add_entity(
            title,
            "Paper",
            f"{arxiv_url} · relevance {relevance_score}/10 · credibility {credibility_tier}/10 · {key_findings}",
            file_path=arxiv_url,
        )

        for inst in affiliations:
            add_entity(inst, "Institution", inst)
            add_edge(title, inst, "Paper published by institution", "published_by institution")
            if primary_topic:
                add_edge(inst, primary_topic, "Institution researches topic", "researches topic")

        if primary_topic:
            add_entity(primary_topic, "Topic", primary_topic)
            add_edge(title, primary_topic, "Paper's primary research topic", "primary_topic topic")

        for sec in secondary_topics:
            add_entity(sec, "Topic", sec)
            add_edge(title, sec, "Paper also covers topic", "also_covers topic")
            if primary_topic:
                add_edge(primary_topic, sec, "Related research topics", "related_to topic")

        add_edge(title, report_name, "Paper selected in report", "selected_in report")

        author_affiliations = paper.get("author_affiliations") or {}
        for author_name, inst_list in author_affiliations.items():
            author_name = author_name.strip()
            if not author_name:
                continue
            add_entity(author_name, "Author", author_name)
            add_edge(title, author_name, "Paper authored by researcher", "authored_by author")
            for inst in (inst_list or []):
                inst = inst.strip()
                if inst:
                    add_entity(inst, "Institution", inst)
                    add_edge(author_name, inst, "Researcher affiliated with institution", "affiliated_with institution")

        chunks.append({
            "content": (
                f"Paper: {title}\n"
                f"ArXiv: {arxiv_url}\n"
                f"Profile: {profile_id}\n"
                f"Report date: {run_date}\n"
                f"Institutions: {', '.join(affiliations)}\n"
                f"Primary topic: {primary_topic}\n"
                f"Secondary topics: {', '.join(secondary_topics)}\n"
                f"Relevance score: {relevance_score}/10\n"
                f"Credibility tier: {credibility_tier}/10\n"
                f"Key findings: {key_findings}\n"
                f"Summary: {short_summary}"
            ),
            "source_id": source_id,
            "file_path": arxiv_url,
        })

    for article in media:
        title = article.get("title", "").strip()
        if not title:
            continue

        source = (article.get("source") or "").strip()
        date = article.get("date", "")
        url = article.get("url", "")
        summary = article.get("summary", "")

        add_entity(title, "Article", f"{url} · {date} · {summary[:120]}", file_path=url or "custom_kg")

        if source:
            add_entity(source, "Publication", source)
            add_edge(title, source, "Article from publication", "from_source publication")

        if profile_topic:
            add_entity(profile_topic, "Topic", profile_topic)
            add_edge(title, profile_topic, "Article covers research topic", "covers_topic topic")

        add_edge(title, report_name, "Article selected in report", "selected_in report")

        chunks.append({
            "content": (
                f"Article: {title}\n"
                f"Source: {source}\n"
                f"Date: {date}\n"
                f"URL: {url}\n"
                f"Profile: {profile_id}\n"
                f"Report date: {run_date}\n"
                f"Summary: {summary}"
            ),
            "source_id": source_id,
            "file_path": url or "custom_kg",
        })

    # Don't include the Report entity if nothing else was ingested
    if len(entities) == 1 and report_name in entities:
        return {"entities": [], "relationships": [], "chunks": []}

    return {
        "entities": list(entities.values()),
        "relationships": relationships,
        "chunks": chunks,
    }


# Creating an entity/relation triggers an embedding call + a rewrite of the
# (large) vector DB, so a single call runs a few seconds and can occasionally
# stall on the embedding provider. Give each call room and retry transient
# timeouts/connection drops so one slow call doesn't fail a whole week.
_POST_TIMEOUT = 180
_POST_RETRIES = 3


def _post_with_retry(session: requests.Session, url: str, payload: dict) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(_POST_RETRIES):
        try:
            return session.post(url, json=payload, timeout=_POST_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            # A timeout may mean the server still applied the write; a retry then
            # returns 400 ("already exists"), which callers treat as success.
            last_err = exc
            if attempt < _POST_RETRIES - 1:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"POST {url} failed after {_POST_RETRIES} attempts: {last_err}")


def _post_entity(session: requests.Session, entity: dict, entity_url: str) -> bool:
    resp = _post_with_retry(session, entity_url, {
        "entity_name": entity["entity_name"],
        "entity_data": {
            "description": entity["description"],
            "entity_type": entity["entity_type"],
            "source_id": entity["source_id"],
            "file_path": entity.get("file_path", "custom_kg"),
        },
    })
    if resp.status_code == 400:
        return False  # already exists — not an error
    resp.raise_for_status()
    return True


def _post_relation(session: requests.Session, rel: dict, relation_url: str) -> bool:
    resp = _post_with_retry(session, relation_url, {
        "source_entity": rel["src_id"],
        "target_entity": rel["tgt_id"],
        "relation_data": {
            "description": rel["description"],
            "keywords": rel["keywords"],
            "weight": rel.get("weight", 1.0),
            "source_id": rel["source_id"],
        },
    })
    if resp.status_code == 400:
        return False  # already exists or missing entity — skip
    resp.raise_for_status()
    return True


def ingest_report(
    selected: list[dict],
    media: list[dict],
    config: dict,
    base_url: str | None = None,
) -> bool:
    try:
        resolved_base_url = _resolve_base_url(base_url)
        entity_url = f"{resolved_base_url}/graph/entity/create"
        relation_url = f"{resolved_base_url}/graph/relation/create"
        run_date = config["current_date"]
        profile_id = config["profile_id"]
        source_id = f"{profile_id}-{run_date}T{datetime.now().strftime('%H%M%S')}"

        payload = _build_payload(selected, media, config, source_id)
        if not payload["entities"]:
            return True

        with requests.Session() as session:
            # Entities must be created before relations
            n_entities = sum(_post_entity(session, e, entity_url) for e in payload["entities"])
            n_rels = sum(_post_relation(session, r, relation_url) for r in payload["relationships"])

        print(
            f"  Ingested {n_entities}/{len(payload['entities'])} entities, "
            f"{n_rels}/{len(payload['relationships'])} relationships"
        )
        return True

    except Exception as exc:
        print(f"[graph_ingest] WARNING — failed to ingest report into LightRAG: {exc}")
        return False

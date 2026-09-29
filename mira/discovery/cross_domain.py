"""Topic-layer view across BOTH graphs, rendered for a single LLM call.

Unlike mira/hypothesis/corpus.py, which builds a profile-scoped view of one
graph for retrieval, this module reads the complete `Topic` layer of two graphs
and flattens it to text. There is no top-k and no vector search: the whole layer
is small enough (~41k tokens as of 2026-07-21) to pass in one prompt, which is
the point — the two graphs share only 38 of 933 topics, so retrieval-based
approaches have almost no existing structure to traverse between them.

Spec: docs/superpowers/specs/2026-07-21-cross-domain-topic-discovery-design.md
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

TOPIC_TYPE = "Topic"

# LightRAG's separator for multi-valued node/edge attributes.
SEP = "<SEP>"

# Entity names that survived ingest carrying no information. "Unknown" appears
# in BOTH graphs (fallout from the corrupted-author-name family), so without
# this filter it would be handed to the model as evidence of a researcher
# publishing across domains.
PLACEHOLDER_NAMES = {"", "unknown", "n/a", "none", "null"}


@dataclass(frozen=True)
class TopicLayer:
    domain: str
    topics: dict[str, str]                      # name -> description
    links: list[tuple[str, str, str, str]]      # (src, dst, keywords, description)


def load_topic_layer(graph: nx.Graph, domain: str) -> TopicLayer:
    """Extract the `Topic` layer of `graph`: every Topic node's description,
    plus every edge whose BOTH endpoints are Topics (links to/from non-Topic
    entities, e.g. Paper or Author nodes, are excluded from `links`)."""
    topics = {
        n: (graph.nodes[n].get("description") or "")
        for n in graph.nodes
        if graph.nodes[n].get("entity_type") == TOPIC_TYPE
    }
    # nx yields undirected edges in arbitrary endpoint order; sort each pair and
    # then the whole list so the rendered payload is byte-stable across runs.
    links = sorted(
        (min(u, v), max(u, v),
         graph[u][v].get("keywords") or "",
         graph[u][v].get("description") or "")
        for u, v in graph.edges
        if u in topics and v in topics
    )
    return TopicLayer(domain=domain, topics=topics, links=links)


def shared_topics(a: TopicLayer, b: TopicLayer) -> list[str]:
    return sorted(set(a.topics) & set(b.topics))


def shared_entities(a: nx.Graph, b: nx.Graph) -> list[tuple[str, str]]:
    """Non-Topic entities present in both graphs, as (name, entity_type).

    An author or lab publishing on both sides is evidence a cross-domain topic
    is practically viable, which the topic nodes alone do not carry.

    Where the two graphs disagree on entity_type (exactly one node as of
    2026-07-21) the first graph's type wins — cosmetic labelling, not
    correctness.
    """
    out: list[tuple[str, str]] = []
    for name in sorted(set(a.nodes) & set(b.nodes)):
        if name.strip().lower() in PLACEHOLDER_NAMES:
            continue
        etype = a.nodes[name].get("entity_type") or "Unknown"
        if etype == TOPIC_TYPE:
            continue
        out.append((name, etype))
    return out


def _section(title: str, lines: list[str]) -> str:
    body = "\n".join(lines) if lines else "(none)"
    return f"# {title}\n{body}\n"


def _collapse_repeats(value: str) -> str:
    """Reduce a <SEP>-joined value whose segments are all identical to one segment.

    LightRAG joins multi-valued attributes with <SEP>. "X<SEP>X" therefore
    records the same fact twice, not two facts. Left alone, 10 such links out of
    608 in the memory graph were enough to make the whole section look
    heterogeneous and force the verbose per-link format on the other 598.
    """
    if SEP not in value:
        return value
    segments = value.split(SEP)
    return segments[0] if all(s == segments[0] for s in segments) else value


def _links_section(label: str, links: list[tuple[str, str, str, str]]) -> str:
    """Render a domain's topic-topic links.

    Real graphs turn out to have every link share ONE (keywords, description)
    pair — spelling it out per link burns tokens without adding information.
    When links are heterogeneous (or empty), fall back to the explicit
    per-link format so no information is lost.
    """
    normalized = [(u, v, _collapse_repeats(kw), _collapse_repeats(desc))
                  for u, v, kw, desc in links]
    distinct = {(kw, desc) for _, _, kw, desc in normalized}
    title = f"{label} DOMAIN — {len(links)} topic links"
    if len(distinct) == 1:
        (kw, desc), = distinct
        title += f" (all: {kw} / {desc})"
        lines = [f"- {u} -> {v}" for u, v, _, _ in normalized]
    else:
        lines = [f"- {u} -> {v} [{kw}]: {desc}" for u, v, kw, desc in normalized]
    return _section(title, lines)


def build_payload(mem: TopicLayer, opt: TopicLayer, shared_t: list[str],
                  shared_e: list[tuple[str, str]]) -> str:
    """Flatten both topic layers to plain text.

    A topic present in both graphs is listed ONCE, in its own section — never
    repeated in each domain's list — so "what is unique to this domain" reads
    directly off the payload. Links may still name a shared topic; it resolves
    against the shared section.

    Plain text rather than JSON: JSON would spend a meaningful slice of the
    token budget on punctuation for no comprehension gain.
    """
    shared_set = set(shared_t)
    parts: list[str] = []
    for layer in (mem, opt):
        label = layer.domain.upper()
        exclusive = {n: d for n, d in layer.topics.items() if n not in shared_set}
        parts.append(_section(
            f"{label} DOMAIN — {len(exclusive)} topics",
            # Collapse identical <SEP> repeats first, so "X<SEP>X" is recognised
            # as redundant against a topic named "X" and dropped like a plain
            # duplicate. Descriptions holding genuinely different <SEP> segments
            # (e.g. "MRAM (Magnetic RAM)<SEP>MRAM") survive intact.
            [f"- {name}" if _collapse_repeats(desc).strip() == name.strip()
             else f"- {name}: {_collapse_repeats(desc)}"
             for name, desc in sorted(exclusive.items())],
        ))
        parts.append(_links_section(label, layer.links))
    parts.append(_section(
        f"TOPICS PRESENT IN BOTH DOMAINS ({len(shared_t)})",
        [f"- {name}" for name in shared_t],
    ))
    parts.append(_section(
        f"ENTITIES PRESENT IN BOTH DOMAINS ({len(shared_e)})",
        [f"- {name} ({etype})" for name, etype in shared_e],
    ))
    return "\n".join(parts)


def estimate_tokens(payload: str) -> int:
    """Rough budget check, not billing. ~4 chars/token holds for this content."""
    return len(payload) // 4

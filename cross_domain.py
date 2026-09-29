#!/usr/bin/env python3
"""Cross-domain research topics from the memory and optical graphs.

    python3 cross_domain.py --dry-run      # build payload, print size, no LLM call
    python3 cross_domain.py --limit 10

Unlike discover.py and hypothesize.py, this reads BOTH graphs and passes their
complete topic layer to the model in one call. No vector retrieval, no top-k:
the two graphs share only 38 of 933 topics, so there is almost no existing
structure for a retrieval-based approach to traverse between them. Passing the
whole layer sidesteps that entirely.

Reads .graphml files directly. The LightRAG servers are never contacted.

Spec: docs/superpowers/specs/2026-07-21-cross-domain-topic-discovery-design.md
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from mira.config import ROOT, llm_call, make_llm_client
from mira.discovery.cross_domain import (
    build_payload, estimate_tokens, load_topic_layer, shared_entities, shared_topics,
)
from mira.graph_target import resolve_target
from mira.hypothesis.corpus import load_graph

CROSS_DOMAIN_MODEL = "anthropic/claude-sonnet-4.6"

SYSTEM = """You are a research strategist identifying topics that require BOTH \
semiconductor memory expertise and optical/photonic interconnect expertise.

You will receive the COMPLETE topic layer of two separate knowledge graphs built \
from conference and arXiv literature — one on memory, one on optical \
interconnects. This is everything both graphs know at the topic level: topic \
names, short descriptions, and how topics link within each domain. It is a \
skeleton. There are no paper abstracts. Most topic "descriptions" are just the \
topic name repeated back and carry no extra information beyond the name itself. \
Links record only that two topics co-occur in the literature — they do NOT \
describe HOW or WHY the topics relate; that connection is never in the payload.

Rules:
1. Every candidate MUST require both domains. A topic answerable from one domain \
alone is a failure, not a safe answer.
2. Cite only topics that appear verbatim in the payload. Never invent one.
3. Topics under "TOPICS PRESENT IN BOTH DOMAINS" are where the two literatures \
ALREADY meet, so they are the least likely to be novel. Treat them as low-value \
and put any candidate built on them in a final section headed \
"## Already-bridged (low novelty)".
4. When an entry from "ENTITIES PRESENT IN BOTH DOMAINS" supports a candidate — \
an author or lab already publishing on both sides — name it. Some of these \
entries are Paper entities that carry a title only, with no abstract and no \
findings; you may name such a paper as a bridge, but never state what it found, \
measured, or concluded — you have no basis for that claim.
5. You are working from a skeleton. Every candidate is a direction to \
investigate, not an established result. Do not cite papers, measurements, or \
findings that are not in the payload. If a connection is speculative, say so in \
the Confidence field rather than dressing it up.
6. The payload cannot supply a mechanism — it has no descriptions of how or why \
topics relate. The Mechanism field is your own reasoning, not a fact read out of \
the corpus; never write it in a way that implies the payload stated it.

For each candidate output exactly this shape:

### <title>
- **Memory topics:** <verbatim topic names from the payload>
- **Optical topics:** <verbatim topic names from the payload>
- **Mechanism (your own reasoning, not from the payload):** <why these connect, 2-3 sentences>
- **Must be true:** <the load-bearing assumption>
- **Cheapest falsification:** <the check that would kill this fastest>
- **Bridging entities:** <names from the shared-entity list, or "none">
- **Confidence:** <high | medium | low, and why>
"""


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Cross-domain research topics from the memory and optical graphs.")
    ap.add_argument("--limit", type=int, default=10,
                    help="Candidate topics to request (default 10)")
    ap.add_argument("--model", default=CROSS_DOMAIN_MODEL,
                    help=f"Override the model (default {CROSS_DOMAIN_MODEL})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the payload and report its size; make no LLM call")
    ap.add_argument("--out", default=None,
                    help="Output path (default report-files/cross-domain/)")
    return ap.parse_args()


def _default_out() -> Path:
    return (ROOT / "report-files" / "cross-domain"
            / f"{date.today().isoformat()}-cross-domain-topics.md")


def _header(mem, opt, shared_t, shared_e, payload: str, model: str) -> str:
    # The graphs change weekly; an output file with no provenance cannot be
    # interpreted later.
    return "\n".join([
        "# Cross-Domain Research Topics — Memory × Optical",
        "",
        f"**Run date:** {date.today().isoformat()}",
        f"**Model:** {model}",
        f"**Memory graph:** {len(mem.topics)} topics, {len(mem.links)} topic links",
        f"**Optical graph:** {len(opt.topics)} topics, {len(opt.links)} topic links",
        f"**Shared topics:** {len(shared_t)}  **Shared entities:** {len(shared_e)}",
        f"**Payload:** ~{estimate_tokens(payload):,} tokens",
        "",
        "> Generated from the topic layer only — no paper abstracts. Every",
        "> candidate is a direction to investigate, not an established result.",
        "",
        "---",
        "",
        "",
    ])


def main() -> int:
    args = _parse_args()

    mem_graph = load_graph(resolve_target("memory").graphml)
    opt_graph = load_graph(resolve_target("optical").graphml)
    mem = load_topic_layer(mem_graph, "memory")
    opt = load_topic_layer(opt_graph, "optical")
    shared_t = shared_topics(mem, opt)
    shared_e = shared_entities(mem_graph, opt_graph)
    payload = build_payload(mem, opt, shared_t, shared_e)

    print(f"memory:  {len(mem.topics)} topics, {len(mem.links)} links")
    print(f"optical: {len(opt.topics)} topics, {len(opt.links)} links")
    print(f"shared:  {len(shared_t)} topics, {len(shared_e)} entities")
    print(f"payload: {len(payload):,} chars  ~{estimate_tokens(payload):,} tokens")

    out = Path(args.out) if args.out else _default_out()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        payload_path = out.with_suffix(".payload.txt")
        payload_path.write_text(payload)
        print(f"dry run — no LLM call. Payload written to {payload_path}")
        return 0

    reply = llm_call(make_llm_client(), args.model, SYSTEM,
                     f"Propose {args.limit} candidate research topics.\n\n{payload}")
    out.write_text(_header(mem, opt, shared_t, shared_e, payload, args.model) + reply)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

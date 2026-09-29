#!/usr/bin/env python3
"""Hypothesis dossier generator — literature-based discovery over the MIRA
LightRAG graph.

    python3 hypothesize.py --topic "CXL memory pooling" --profile memory-innovation

Spec: docs/superpowers/specs/2026-07-09-hypothesis-generation-design.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

from mira.config import ROOT, iso_date_arg, make_llm_client
from chatbot.config import Settings
from mira.exhaustive.providers import run_research_job
from mira.graph_target import DEFAULT_GRAPH, GRAPH_NAMES, resolve_target
from mira.hypothesis.corpus import Corpus, load_graph, match_anchor_topics
from mira.hypothesis.dossier import render_dossier
from mira.hypothesis.gaps import mine_gaps, score_candidates
from mira.hypothesis.novelty import NoveltyChecker, classify, unverified_note
from mira.hypothesis.synthesis import apply_critic, retrieve_context, synthesize
from mira.hypothesis.vectors import EntityVectors
from mira.venue_corpus import (
    STALE_AFTER_DAYS,
    make_multi_venue_fetch,
    make_venue_fetch,
    read_built_at,
    venue_corpus_is_queryable,
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def out_name(topic: str, stamp: str) -> str:
    """Dossier filename with a minute stamp so same-day reruns never collide."""
    return f"{_slug(topic)}-{stamp}.md"


def _topic_is_storage_only(corpus: Corpus, topic: str) -> bool:
    papers = corpus.topic_papers.get(topic, set())
    profile_sets = [
        corpus.paper_profiles.get(paper, set()) for paper in papers
    ]
    return bool(profile_sets) and all(
        profiles
        and all(profile.casefold().startswith("storage")
                for profile in profiles)
        for profiles in profile_sets
    )


def _storage_coverage_note(corpus: Corpus, candidates) -> str | None:
    """Explain topics whose graph evidence has no authoritative venue corpus."""
    topics = {
        topic
        for candidate in candidates
        for topic in (candidate.topic_a, candidate.topic_c)
        if _topic_is_storage_only(corpus, topic)
    }
    if not topics:
        return None
    return (
        "storage profiles have no venue corpus — novelty unverified for "
        f"storage-only topics: {', '.join(sorted(topics))}"
    )


def _enforce_storage_unverifiability(corpus: Corpus, candidate) -> bool:
    """Storage-only graph evidence cannot be verified by memory/optical DBs."""
    if not any(
        _topic_is_storage_only(corpus, topic)
        for topic in (candidate.topic_a, candidate.topic_c)
    ):
        return False
    candidate.novelty_hits = None
    candidate.external_titles = []
    candidate.novelty_label = classify(None)
    return True


def _apply_storage_coverage(corpus: Corpus, candidates) -> str | None:
    """Enforce storage-only unverifiability and return its degraded note."""
    for candidate in candidates:
        _enforce_storage_unverifiability(corpus, candidate)
    return _storage_coverage_note(corpus, candidates)


def _target_venue_fetch(target):
    """Preserve singular fetch semantics; union only multi-corpus targets."""
    if len(target.venue_dbs) > 1:
        return make_multi_venue_fetch(target.venue_dbs)
    return make_venue_fetch(target.venue_db)


def _invalidate_union_novelty_cache(target) -> bool:
    """Discard a combined cache when any member is unavailable or newer."""
    cache_path = target.novelty_cache
    if len(target.venue_dbs) <= 1 or not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime_ns
    unsafe = False
    for venue_db in target.venue_dbs:
        try:
            if (
                not venue_db.exists()
                or not venue_corpus_is_queryable(venue_db)
                or read_built_at(venue_db) is None
                or venue_db.stat().st_mtime_ns > cache_mtime
            ):
                unsafe = True
                break
        except OSError:
            unsafe = True
            break
    if not unsafe:
        return False
    cache_path.unlink(missing_ok=True)
    return True


def _check_candidate_novelty(checker: NoveltyChecker, candidate) -> None:
    """Apply a check, evicting results produced with incomplete union data."""
    candidate.novelty_hits, candidate.external_titles = checker.check(
        candidate.topic_a, candidate.topic_c)
    candidate.novelty_label = classify(candidate.novelty_hits)
    if not getattr(checker.fetch, "notes", ()):
        return
    key = f"{candidate.topic_a} || {candidate.topic_c}"
    if checker.cache.pop(key, None) is not None:
        checker.cache_path.parent.mkdir(parents=True, exist_ok=True)
        checker.cache_path.write_text(json.dumps(checker.cache, indent=2))


def _run_exhaustive(args) -> int:
    # The exhaustive path reads the three graph directories directly and never
    # calls the LightRAG server, so it does not need LIGHTRAG_API_KEY.
    settings = Settings.from_env(require_lightrag=False)
    printed = set()

    def emit(event) -> None:
        name = (
            event.name if hasattr(event, "name")
            else event.get("name", "")
        )
        marker = None
        text = None
        if name == "evidence_collected":
            marker, text = 2, "Compiling all selected evidence"
        elif name == "hypothesis_candidates_started":
            marker, text = 3, "Enumerating every structural gap candidate"
        elif name == "hypothesis_synthesis_started":
            marker, text = 4, "Synthesizing grounded hypotheses"
        if marker and marker not in printed:
            printed.add(marker)
            print(f"[{marker}/5] {text} ...", flush=True)

    print("[1/5] Scanning memory, optical, and storage graphs ...", flush=True)
    result = run_research_job(
        settings,
        "hypotheses",
        {
            "topics": args.topic,
            "profiles": args.profile,
            "max_hypotheses": args.max_hypotheses,
            # Deliberately omit max_candidates: exhaustive enumeration has no
            # candidate-count truncation before deterministic scoring.
            "critic": args.critic,
            "no_external": args.no_external,
        },
        emit,
    )
    print("[5/5] Writing dossier ...", flush=True)
    output_profile = (
        args.profile[0] if len(args.profile) == 1 else "combined"
    )
    out_root = (
        Path(args.out_dir)
        if args.out_dir else ROOT / "report-files" / "hypotheses"
    )
    out_dir = out_root / output_profile
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        args.topic[0]
        if len(args.topic) == 1
        else f"{args.topic[0]}-plus{len(args.topic) - 1}"
    )
    out = out_dir / out_name(
        stem, datetime.now().strftime("%Y-%m-%d-%H%M")
    )
    cost = result.cost
    markdown = (
        result.result.markdown
        + "\n\n## Exhaustive execution\n\n"
        + f"- **Actual cost:** ${cost.actual_cost_usd:.6f} "
        + f"({cost.cost_status})\n"
        + "- **Coverage:** "
        + " · ".join(
            f"{domain} "
            f"{result.coverage.nodes_scanned[domain]} nodes / "
            f"{result.coverage.edges_scanned[domain]} edges"
            for domain in ("memory", "optical", "storage")
        )
        + "\n"
    )
    out.write_text(markdown)
    print(f"Dossier written: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a hypothesis dossier for a seed topic.")
    ap.add_argument("--topic", action="append", required=True,
                    help="Seed topic; repeat to anchor on several at once "
                         "(the gateway's picker sends one per selection).")
    ap.add_argument("--profile", action="append", required=True)
    ap.add_argument("--graph", choices=GRAPH_NAMES, default=DEFAULT_GRAPH,
                    help=f"Which LightRAG graph to read (default {DEFAULT_GRAPH})")
    ap.add_argument("--max-hypotheses", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=12)
    ap.add_argument("--since", type=iso_date_arg, default=None, help="YYYY-MM-DD")
    ap.add_argument("--no-external", action="store_true",
                    help="Skip the venue-corpus novelty check; novelty marked 'unverified'")
    ap.add_argument("--critic", action="store_true",
                    help="Run a critic LLM pass on each hypothesis")
    ap.add_argument(
        "--exhaustive",
        action="store_true",
        help="scan all three live graphs and use the shared evidence compiler",
    )
    ap.add_argument("--out-dir", default=None,
                    help="Root for dossier output (default report-files/hypotheses). "
                         "The gateway passes its configured directory so the run "
                         "writes where the caller looks for the result.")
    args = ap.parse_args()

    if args.exhaustive:
        return _run_exhaustive(args)

    target = resolve_target(args.graph)
    degraded: list[str] = []

    print(f"[1/5] Loading graph from {target.graphml} ...")
    g = load_graph(target.graphml)
    corpus = Corpus.build(g, args.profile, since=args.since)
    if not corpus.papers:
        profiles = sorted({
            re.sub(r"-\d{4}-\d{2}-\d{2}$", "", n)
            for n, d in g.nodes(data=True) if d.get("entity_type") == "Report"
        })
        requested = "', '".join(args.profile)
        noun = "profile" if len(args.profile) == 1 else "profiles"
        print(f"ERROR: no papers found for {noun} '{requested}'. "
              f"Profiles in graph: {profiles}", file=sys.stderr)
        return 1

    # Union across every --topic, preserving order and dropping duplicates, so
    # several selected topics anchor one run.
    anchors = list(dict.fromkeys(
        a for seed in args.topic
        for a in match_anchor_topics(seed, corpus.topic_papers)
    ))
    if not anchors:
        requested_topics = "', '".join(args.topic)
        words = [w for seed in args.topic
                 for w in re.split(r"\W+", seed.lower()) if len(w) > 2]
        near = sorted(t for t in corpus.topic_papers
                      if any(w in t.lower() for w in words))[:20]
        noun = "Topic node matches" if len(args.topic) == 1 else "Topic nodes match"
        print(f"ERROR: no {noun} '{requested_topics}'.", file=sys.stderr)
        print("Nearest topic names: " + (", ".join(near) or "(none)"), file=sys.stderr)
        return 1
    expanded = sorted(set(anchors) | {
        r for a in anchors for r in corpus.related_topics(a)
        if r in corpus.topic_papers
    })
    print(f"      Anchors: {anchors} (expanded to {len(expanded)} topics, "
          f"{len(corpus.papers)} profile papers)")

    print("[2/5] Mining gap candidates ...")
    candidates = mine_gaps(corpus, expanded)
    try:
        vectors = EntityVectors.load(target.vdb_entities)
        semantic = {
            (c.topic_a, c.topic_c): vectors.side_similarity(
                corpus.topic_papers.get(c.topic_a, ()),
                corpus.topic_papers.get(c.topic_c, ()),
                fallback_a=c.topic_a, fallback_b=c.topic_c,
            )
            for c in candidates
        }
    except (OSError, KeyError, ValueError) as exc:
        degraded.append(f"entity vectors unavailable ({exc}) — semantic scores are 0")
        semantic = {}
    ranked = score_candidates(candidates, semantic)[: args.max_candidates]
    print(f"      {len(candidates)} candidates mined, top {len(ranked)} kept")

    print("[3/5] Novelty check (top venues) ...")
    if args.no_external:
        degraded.append("--no-external — novelty unverified")
        for c in ranked:
            c.novelty_label = "unverified"
    else:
        _invalidate_union_novelty_cache(target)
        for venue_db in target.venue_dbs:
            if not venue_db.exists():
                message = (
                    f"venue corpus missing ({venue_db.name}) — "
                    "run scripts/fetch_venue_corpus.py"
                )
                print(f"      WARNING: {message}", file=sys.stderr)
                degraded.append(f"{message}; novelty coverage reduced")
                continue
            built = read_built_at(venue_db)
            if built is None or (date.today() - built).days > STALE_AFTER_DAYS:
                degraded.append(
                    f"venue corpus stale ({venue_db.name}, built {built}) — "
                    "consider re-running scripts/fetch_venue_corpus.py"
                )
        checker = NoveltyChecker(
            target.novelty_cache,
            fetch=_target_venue_fetch(target),
        )
        for c in ranked:
            _check_candidate_novelty(checker, c)
        for corpus_note in getattr(checker.fetch, "notes", ()):
            if not corpus_note.endswith(": missing"):
                degraded.append(f"{corpus_note}; novelty coverage reduced")
        if checker.fetch.backoffs:
            print(f"      vocabulary back-off used for "
                  f"{len(checker.fetch.backoffs)} topic side(s)")
    storage_note = _apply_storage_coverage(corpus, ranked)
    if not args.no_external:
        note = unverified_note(ranked)
        if note:
            degraded.append(note)
    if storage_note:
        degraded.append(storage_note)
    survivors = [c for c in ranked if c.novelty_label != "dropped"][: args.max_hypotheses]
    print(f"      {len(survivors)} candidates survive novelty filtering")

    print(f"[4/5] Synthesizing {len(survivors)} hypotheses ...")
    client = make_llm_client()
    hypotheses = []
    for c in survivors:
        ctx_a = retrieve_context(f"{c.topic_a}: key findings, mechanisms, results",
                                 base_url=target.base_url)
        ctx_b = retrieve_context(f"{c.topic_c}: key findings, mechanisms, results",
                                 base_url=target.base_url)
        if ctx_a is None and ctx_b is None and "LightRAG" not in " ".join(degraded):
            degraded.append("LightRAG retrieval unavailable — grounding degraded to graph metadata")
        try:
            text = synthesize(client, c, ctx_a, ctx_b)
        except RuntimeError as exc:
            print(f"      WARNING: synthesis failed for {c.topic_a} × {c.topic_c}: {exc}",
                  file=sys.stderr)
            continue
        verdict = None
        if args.critic:
            try:
                verdict, text = apply_critic(client, c, text, ctx_a, ctx_b)
            except RuntimeError as exc:
                print(f"      WARNING: critic failed for {c.topic_a} × {c.topic_c}: {exc}",
                      file=sys.stderr)
        if verdict == "kill":
            continue
        hypotheses.append({"candidate": c, "text": text,
                           "critic_verdict": verdict, "critic_body": None})
        print(f"      ✓ {c.topic_a} × {c.topic_c}")

    print("[5/5] Writing dossier ...")
    n_papers = sum(1 for _, d in g.nodes(data=True) if d.get("entity_type") == "Paper")
    n_topics = sum(1 for _, d in g.nodes(data=True) if d.get("entity_type") == "Topic")
    flags = " ".join(f for f in (
        "--critic" if args.critic else "", "--no-external" if args.no_external else "",
        f"--since {args.since}" if args.since else "") if f)
    meta = {
        "topic": ", ".join(args.topic), "profiles": corpus.profile_ids,
        "run_date": date.today().isoformat(),
        "graph_stats": f"{n_papers} papers · {n_topics} topics · "
                       f"{len(corpus.papers)} in profile",
        "flags": flags, "anchors": anchors, "degraded": degraded,
    }
    output_profile = args.profile[0] if len(args.profile) == 1 else "combined"
    out_root = Path(args.out_dir) if args.out_dir else ROOT / "report-files" / "hypotheses"
    out_dir = out_root / output_profile
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filename tracks the first topic; a multi-topic run notes how many more.
    stem = args.topic[0] if len(args.topic) == 1 else f"{args.topic[0]}-plus{len(args.topic) - 1}"
    out = out_dir / out_name(stem, datetime.now().strftime("%Y-%m-%d-%H%M"))
    out.write_text(render_dossier(meta, hypotheses, candidates))
    print(f"Dossier written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

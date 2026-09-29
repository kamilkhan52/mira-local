#!/usr/bin/env python3
"""Corpus-wide hypothesis recommendation — ranked discovery agenda + top dossiers.

    python3 discover.py --profile memory-innovation

Spec: docs/superpowers/specs/2026-07-13-hypothesis-recommendation-design.md
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from mira.config import ROOT, iso_date_arg, make_llm_client
from mira.discovery.agenda import render_agenda, slug
from mira.discovery.canonicalize import build_merge_map, merge_topics_in_graph
from mira.discovery.feasibility import TIER_MULTIPLIERS, judge_feasibility
from mira.discovery.ledger import Ledger, pair_key, parse_pair
from mira.discovery.ranking import apply_hub_damping, final_score, semantic_scores
from mira.graph_target import DEFAULT_GRAPH, GRAPH_NAMES, GraphTarget, resolve_target
from mira.hypothesis.corpus import Corpus, load_graph
from mira.hypothesis.dossier import render_dossier
from mira.hypothesis.gaps import mine_gaps, score_candidates
from mira.hypothesis.novelty import NoveltyChecker, classify, unverified_note
from mira.hypothesis.synthesis import apply_critic, retrieve_context, synthesize
from mira.hypothesis.vectors import EntityVectors
from mira.venue_corpus import STALE_AFTER_DAYS, make_venue_fetch, read_built_at

NEAR_MISS_CAP = 30


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Corpus-wide discovery: ranked research agenda + top hypothesis dossiers.")
    ap.add_argument("--profile", help="Profile whose corpus slice to scan")
    ap.add_argument("--graph", choices=GRAPH_NAMES, default=DEFAULT_GRAPH,
                    help=f"Which LightRAG graph to read (default {DEFAULT_GRAPH})")
    ap.add_argument("--top-agenda", type=int, default=20,
                    help="Candidates carried into novelty check + agenda (default 20)")
    ap.add_argument("--top-dossiers", type=int, default=3,
                    help="Full dossiers for the highest-ranked survivors (default 3)")
    ap.add_argument("--since", type=iso_date_arg, default=None, help="YYYY-MM-DD")
    ap.add_argument("--no-external", action="store_true",
                    help="Skip the venue-corpus novelty check; novelty marked 'unverified'")
    ap.add_argument("--critic", action="store_true",
                    help="Run a critic LLM pass on each dossier; a killed "
                         "dossier promotes the next candidate")
    ap.add_argument("--ignore-ledger", action="store_true",
                    help="Clean-slate ranking: no dedup/demotion from run history")
    ap.add_argument("--set-status", nargs=2, metavar=("PAIR", "STATUS"),
                    help='Maintenance: update a ledger entry ("TOPIC_A|TOPIC_C" status) and exit')
    args = ap.parse_args()
    if not args.set_status and not args.profile:
        ap.error("--profile is required (unless using --set-status)")
    return args


def _set_status(target: GraphTarget, pair: str, status: str) -> int:
    ledger = Ledger.load(target.ledger_path)
    if ledger.warning:
        print(f"WARNING: {ledger.warning}", file=sys.stderr)
    parsed = parse_pair(pair)
    if parsed is None:
        print('ERROR: PAIR must be "TOPIC_A|TOPIC_C" (or the ledger key form '
              '"TOPIC_A || TOPIC_C")', file=sys.stderr)
        return 1
    key = pair_key(*parsed)
    try:
        found = ledger.set_status(key, status)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not found:
        print(f"ERROR: no ledger entry for '{key}'. Known keys:", file=sys.stderr)
        for k in sorted(ledger.entries):
            print(f"  {k}", file=sys.stderr)
        return 1
    ledger.save()
    print(f"{key} -> {status}")
    return 0


def main() -> int:
    args = _parse_args()
    target = resolve_target(args.graph)
    if args.set_status:
        return _set_status(target, *args.set_status)

    degraded: list[str] = []
    run_date = date.today()
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")

    print(f"[1/7] Loading graph from {target.graphml} ...")
    g = load_graph(target.graphml)
    try:
        vectors = EntityVectors.load(target.vdb_entities)
    except (OSError, KeyError, ValueError) as exc:
        degraded.append(f"entity vectors unavailable ({exc}) — "
                        "name-only canonicalization, semantic scores are 0")
        vectors = None

    print("[2/7] Canonicalizing topics ...")
    all_topics = [n for n, d in g.nodes(data=True) if d.get("entity_type") == "Topic"]
    merge_map = build_merge_map(all_topics, g, vectors)
    merge_topics_in_graph(g, merge_map)
    print(f"      {len(all_topics)} topics, {len(merge_map)} merged into canonicals")

    corpus = Corpus.build(g, (args.profile,), since=args.since)
    if not corpus.papers:
        print(f"ERROR: no papers found for profile '{args.profile}'.", file=sys.stderr)
        return 1

    print(f"[3/7] Mining gaps across all {len(corpus.topic_papers)} profile topics ...")
    candidates = mine_gaps(corpus, sorted(corpus.topic_papers))
    apply_hub_damping(candidates, corpus)
    semantic = semantic_scores(vectors, corpus, candidates) if vectors else {}
    ranked = score_candidates(candidates, semantic)
    print(f"      {len(candidates)} gap candidates mined")

    ledger = Ledger.load(target.ledger_path)
    if ledger.warning:
        degraded.append(ledger.warning)
    if args.ignore_ledger:
        eligible, pursuing_cands, suppressed = ranked, [], []
    else:
        eligible, pursuing_cands, suppressed = ledger.partition(ranked)
    top = eligible[: args.top_agenda]

    print(f"[4/7] Novelty check (top venues) on top {len(top)} ...")
    backoffs: dict[str, str] = {}
    if args.no_external:
        degraded.append("--no-external — novelty unverified")
        for c in top:
            c.novelty_label = "unverified"
    else:
        if not target.venue_db.exists():
            print("      WARNING: venue corpus missing — run scripts/fetch_venue_corpus.py",
                  file=sys.stderr)
            degraded.append("venue corpus missing — run scripts/fetch_venue_corpus.py; novelty unverified")
        else:
            built = read_built_at(target.venue_db)
            if built is None or (date.today() - built).days > STALE_AFTER_DAYS:
                degraded.append(f"venue corpus stale (built {built}) — "
                                "consider re-running scripts/fetch_venue_corpus.py")
        checker = NoveltyChecker(target.novelty_cache, fetch=make_venue_fetch(target.venue_db))
        for c in top:
            c.novelty_hits, c.external_titles = checker.check(c.topic_a, c.topic_c)
            c.novelty_label = classify(c.novelty_hits)
        note = unverified_note(top)
        if note:
            degraded.append(note)
        backoffs = checker.fetch.backoffs
        if backoffs:
            print(f"      vocabulary back-off used for {len(backoffs)} topic side(s)")
    survivors = [c for c in top if c.novelty_label != "dropped"]
    dropped = [c for c in top if c.novelty_label == "dropped"]
    print(f"      {len(survivors)} survive novelty filtering")

    print(f"[5/7] Feasibility judge on {len(survivors)} survivors ...")
    client = make_llm_client()
    feasibility: dict[tuple[str, str], tuple[str, str]] = {}
    for c in survivors:
        feasibility[(c.topic_a, c.topic_c)] = judge_feasibility(client, c)
    if any(t == "unrated" for t, _ in feasibility.values()):
        degraded.append("feasibility judge failed for some pairs — tier 'unrated'")

    def _final(c) -> float:
        tier, _ = feasibility[(c.topic_a, c.topic_c)]
        mult = 1.0 if args.ignore_ledger else ledger.multiplier(c.topic_a, c.topic_c)
        return final_score(c.combined_score, c.novelty_label,
                           TIER_MULTIPLIERS[tier], mult)

    survivors.sort(key=_final, reverse=True)

    print(f"[6/7] Synthesizing dossiers for top {args.top_dossiers} ...")
    out_dir = ROOT / "report-files" / "hypotheses" / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    n_papers = sum(1 for _, d in g.nodes(data=True) if d.get("entity_type") == "Paper")
    graph_stats = (f"{n_papers} papers · {len(all_topics)} topics "
                   f"({len(all_topics) - len(merge_map)} after merge) · "
                   f"{len(corpus.papers)} in profile")
    dossier_files: dict[tuple[str, str], str] = {}
    for c in survivors:
        if len(dossier_files) >= args.top_dossiers:
            break
        ctx_a = retrieve_context(f"{c.topic_a}: key findings, mechanisms, results",
                                 base_url=target.base_url)
        ctx_b = retrieve_context(f"{c.topic_c}: key findings, mechanisms, results",
                                 base_url=target.base_url)
        if ctx_a is None and ctx_b is None and not any("LightRAG" in d for d in degraded):
            degraded.append("LightRAG retrieval unavailable — grounding degraded to graph metadata")
        try:
            text = synthesize(client, c, ctx_a, ctx_b)
        except RuntimeError as exc:
            print(f"      WARNING: synthesis failed for {c.topic_a} × {c.topic_c}: "
                  f"{exc} — promoting next candidate", file=sys.stderr)
            degraded.append(f"synthesis failed for {c.topic_a} × {c.topic_c} — "
                            "next candidate promoted")
            continue
        verdict = None
        if args.critic:
            try:
                verdict, text = apply_critic(client, c, text, ctx_a, ctx_b)
            except RuntimeError as exc:
                print(f"      WARNING: critic failed for {c.topic_a} × {c.topic_c}: {exc}",
                      file=sys.stderr)
                verdict = None
            if verdict == "kill":
                print(f"      ✗ critic killed {c.topic_a} × {c.topic_c} "
                      "— promoting next candidate")
                continue
        fname = f"discovery-{slug(c.topic_a)}--{slug(c.topic_c)}-{stamp}.md"
        meta = {"topic": f"{c.topic_a} × {c.topic_c}", "profile": args.profile,
                "run_date": run_date.isoformat(), "graph_stats": graph_stats,
                "flags": "(discovery run)" + (" --critic" if args.critic else ""),
                "anchors": [c.topic_a, c.topic_c],
                "degraded": degraded}
        (out_dir / fname).write_text(render_dossier(
            meta, [{"candidate": c, "text": text,
                    "critic_verdict": verdict, "critic_body": None}], [c]))
        dossier_files[(c.topic_a, c.topic_c)] = fname
        print(f"      ✓ {fname}")

    print("[7/7] Writing agenda + updating ledger ...")
    items = []
    for c in survivors:
        tier, reason = feasibility[(c.topic_a, c.topic_c)]
        entry = ledger.entry(c.topic_a, c.topic_c)
        note = (f"seen in {entry['times_recommended']} previous runs"
                if entry else "new")
        items.append({"candidate": c, "final_score": _final(c), "tier": tier,
                      "tier_reason": reason, "ledger_note": note,
                      "dossier_file": dossier_files.get((c.topic_a, c.topic_c))})
    pursuing_meta = []
    for c in pursuing_cands:
        e = ledger.entry(c.topic_a, c.topic_c)
        pursuing_meta.append({"key": pair_key(c.topic_a, c.topic_c),
                              "first_recommended": e["first_recommended"],
                              "last_recommended": e["last_recommended"],
                              "best_score": e["best_score"]})
    near_misses = dropped + eligible[args.top_agenda: args.top_agenda + NEAR_MISS_CAP]
    flags = " ".join(f for f in (
        "--no-external" if args.no_external else "",
        "--critic" if args.critic else "",
        "--ignore-ledger" if args.ignore_ledger else "",
        f"--since {args.since}" if args.since else "") if f)
    meta = {"profile": args.profile, "run_stamp": stamp, "graph_stats": graph_stats,
            "flags": flags, "ledger_summary": ledger.summary(), "degraded": degraded}
    stage_counts = {"mined": len(candidates), "eligible": len(eligible),
                    "novelty-checked": len(top), "survivors": len(survivors),
                    "dossiers": len(dossier_files)}

    for item in items:
        c = item["candidate"]
        ledger.upsert(c.topic_a, c.topic_c, item["final_score"], run_date)
    ledger.save()

    agenda_path = out_dir / f"discovery-agenda-{stamp}.md"
    agenda_path.write_text(render_agenda(meta, items, pursuing_meta, suppressed,
                                         merge_map, near_misses, stage_counts,
                                         backoffs=backoffs))
    print(f"Agenda written: {agenda_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

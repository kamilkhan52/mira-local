#!/usr/bin/env python3
"""Benchmark Jev against the LLM on news-article selection (media_selection).

For each profile, the production path (the n8n-parity media selection in
mira.media: build_selection_prompt + the media_selection LLM + apply_selection)
picks up to MEDIA_MAX_SELECT (5) articles from the full list; Jev scores each
article on its own and code keeps the same number. Compares overlap and where
the LLM's picks land in Jev's ranking. Articles come from the crawler outputs
in configs/*-latest.json (or --articles).

  python3.11 scripts/bench_jev_media.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from mira import jev  # noqa: E402
from mira.paths import CONFIG_DIR, REPORT_FILES  # noqa: E402
from mira.config import make_llm_client  # noqa: E402
from mira.config import llm_call, model_for  # noqa: E402
from mira.media import (MEDIA_MAX_SELECT, _normalize_articles, apply_selection,  # noqa: E402
                        build_selection_prompt)


def llm_select(articles: list[dict], config: dict, client) -> list[dict]:
    """The production media-selection path over `articles`."""
    item = {"summary": {}, "article_count": len(articles),
            "articles": [dict(a, summary=a.get("content", "")) for a in articles]}
    user, system = build_selection_prompt(item, config)
    raw = llm_call(client, model_for(config, "media_selection"), system, user)
    return apply_selection(item, raw)["articles"]

OUT = REPORT_FILES / "bench_jev"


def load_articles(paths: list[str]) -> list[dict]:
    arts = []
    for p in paths:
        raw = json.loads(Path(p).read_text())
        source = (raw[0].get("siteName") or Path(p).stem) if raw else Path(p).stem
        arts += _normalize_articles(raw, source)
    return arts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", nargs="*", default=sorted(glob.glob(str(CONFIG_DIR / "*-latest.json"))))
    ap.add_argument("--n", type=int, default=MEDIA_MAX_SELECT)
    args = ap.parse_args()

    raw = json.loads((CONFIG_DIR / "memory-innovation-profile.json").read_text())
    profiles = raw["profiles"]
    articles = load_articles(args.articles)
    print(f"{len(articles)} articles from {len(args.articles)} files")
    client = make_llm_client()
    cache_path = OUT / "media_results.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows = []
    for prof in profiles:
        pid = prof["profile_id"]
        focus = prof["topic"]["focus"]
        guidance = prof.get("media", {}).get("selection_guidance", "")
        config = {"media": prof.get("media", {}), "topic": prof["topic"], "llm_models": raw["llm_models"]}

        key = f"{pid}|{len(articles)}"
        if key not in cache:
            t0 = time.perf_counter()
            llm_sel = llm_select(articles, config, client)
            llm_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            scores = [jev.judge_article(a, focus, guidance) for a in articles]  # sequential timing
            jev_seq_s = time.perf_counter() - t0
            cache[key] = {"llm_urls": [a.get("url") for a in llm_sel if isinstance(a, dict)],
                          "llm_s": llm_s, "jev_seq_s": jev_seq_s,
                          "jev_levels": [s["relevance_level"] for s in scores],
                          "jev_p50_s": sorted(s["latency_s"] for s in scores)[len(scores) // 2]}
            cache_path.write_text(json.dumps(cache, indent=1))
        c = cache[key]
        ranked = sorted(range(len(articles)), key=lambda k: -c["jev_levels"][k])
        jev_top = {articles[k]["url"] for k in ranked[:args.n]}
        llm_set = set(c["llm_urls"])
        rank_of = {articles[k]["url"]: r + 1 for r, k in enumerate(ranked)}
        llm_ranks = sorted(rank_of[u] for u in llm_set if u in rank_of)
        rows.append((pid, len(llm_set), len(jev_top & llm_set), llm_ranks, c))
        print(f"\n### {pid}")
        for k in ranked[:15]:
            a = articles[k]
            print(f"  {c['jev_levels'][k]:.2f} {'LLM' if a['url'] in llm_set else '   '} {a['source'][:10]:10} {a['title'][:90]}")

    L = [f"# Jev vs LLM — news article selection (top {args.n})", "",
         f"{len(articles)} articles. LLM = production media_selection ({raw['llm_models']['media_selection']}) "
         f"over the full list; Jev = one Score per article, top {args.n} by score.", "",
         f"| profile | LLM picks | overlap with Jev top {args.n} | Jev ranks of LLM picks | LLM time | Jev time (sequential / p50 per article) |",
         "|---|---|---|---|---|---|"]
    for pid, n_llm, ov, ranks, c in rows:
        L.append(f"| {pid} | {n_llm} | {ov} | {ranks} | {c['llm_s']:.1f}s | "
                 f"{c['jev_seq_s']:.1f}s / {c['jev_p50_s']:.2f}s |")
    md = "\n".join(L)
    (OUT / "media_report.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()

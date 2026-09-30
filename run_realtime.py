#!/usr/bin/env python3
"""MIRA Live — hourly realtime monitor (see mira/realtime.py).

  python3.11 run_realtime.py                 # all profiles with subscribers
  python3.11 run_realtime.py --profile memory-innovation --dry-run
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="MIRA Live — realtime paper/news alerts")
    ap.add_argument("--profile", action="append", help="Profile id (repeatable); default: all with subscribers")
    ap.add_argument("--dry-run", action="store_true",
                    help="Judge and write the alert HTML, but send no email and leave the ledger unchanged")
    ap.add_argument("--no-news", action="store_true", help="Skip the news crawlers")
    ap.add_argument("--no-llm", action="store_true",
                    help="No LLM summaries; use the Jev-selected key sentence instead")
    args = ap.parse_args()

    from mira import jev, realtime
    from mira.config import make_llm_client

    if not jev.enabled():
        sys.exit("TYPESAFE_API_KEY not set in .env or environment")
    settings = realtime.load_settings()
    profiles = args.profile or [p for p, subs in settings["subscribers"].items() if subs]
    if not profiles:
        sys.exit("No profiles with subscribers in configs/realtime.json")

    client = None
    if not args.no_llm:
        try:
            client = make_llm_client()
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: no LLM client ({e}); summaries will be key sentences")

    news = []
    if not args.no_news:
        print("Crawling news...")
        news = realtime.crawl_news(settings["news_lookback_days"], settings.get("news_sources"))
    for pid in profiles:
        print(f"[{pid}]")
        try:
            res = realtime.run_profile(pid, settings, news, client, dry_run=args.dry_run)
        except Exception as e:  # noqa: BLE001 — one profile must not stop the others
            print(f"  ERROR: {pid} failed — {e}")
            continue
        print(f"  {res}")


if __name__ == "__main__":
    main()

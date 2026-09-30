#!/usr/bin/env python3
"""One-off digest from the command line (same code path as the Prefect
`digest` flow; runs without a Prefect server using an ephemeral one).

  python run.py --profile cxl-research --mode monthly --test-mode --no-email
  python run.py --trigger "CXL Monthly Trigger" --current-date 2026-09-01
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description="MIRA research digest")
    ap.add_argument("--profile", help="Profile id (e.g. cxl-research); default from --trigger or config")
    ap.add_argument("--mode", help="daily | weekly | monthly (default: the profile's default mode)")
    ap.add_argument("--trigger", dest="trigger_name", help="Route like an n8n trigger name instead")
    ap.add_argument("--current-date", help="Pretend today is YYYY-MM-DD (n8n current_date_override)")
    ap.add_argument("--start-date", help="Explicit window start YYYY-MM-DD (requires --end-date)")
    ap.add_argument("--end-date", help="Explicit window end YYYY-MM-DD")
    ap.add_argument("--lookback-days", type=int)
    ap.add_argument("--max-limit", type=int, help="Cap on arXiv results")
    ap.add_argument("--test-mode", action="store_true", help="Write the report under tests/ (not trend history)")
    ap.add_argument("--no-cache", dest="llm_cache_bypass", action="store_true", help="Bypass the LLM stage cache")
    ap.add_argument("--no-trend", dest="trend_enabled", action="store_const", const=False, default=None)
    ap.add_argument("--no-media", dest="include_media", action="store_false")
    ap.add_argument("--no-email", dest="send_email", action="store_false", help="Build everything, send nothing")
    ap.add_argument("--no-pdf", dest="pdf", action="store_false")
    ap.add_argument("--to", dest="recipients", action="append", help="Recipient (repeatable)")
    ap.add_argument("--jev-level", default="off", choices=["off", "prescreen", "gate", "replace"],
                    help="off: all LLM (like n8n); prescreen: Jev skips clearly irrelevant papers; "
                         "gate: Jev decides relevance, LLM only on what passes; replace: Jev instead "
                         "of the per-paper LLM stages")
    ap.add_argument("--jev-prescreen", action="store_true", help="Same as --jev-level prescreen")
    ap.add_argument("--subject-tag", help="Text prepended to the email subject (test runs)")
    ap.add_argument("--graph", action="store_true", help="Also ingest into the LightRAG graph")
    args = ap.parse_args()
    if args.start_date and not args.end_date:
        ap.error("--end-date is required with --start-date")

    from mira_flows.digest import digest_flow
    print(json.dumps(digest_flow(**vars(args)), indent=2))


if __name__ == "__main__":
    main()

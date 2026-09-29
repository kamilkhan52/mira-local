#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="MIRA — Memory Innovation Research Assistant")
    parser.add_argument("--mode",    required=True, choices=["daily", "weekly"], help="Run mode")
    parser.add_argument("--profile", required=True, help="Profile ID (e.g. memory-innovation)")
    parser.add_argument("--start-date", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date",   help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--no-media",  action="store_true", help="Skip media crawlers")
    parser.add_argument("--no-email",  action="store_true", help="Generate HTML only, do not send email")
    parser.add_argument("--no-graph",  action="store_true", help="Skip knowledge graph ingestion")
    parser.add_argument("--no-cache",  action="store_true",
                        help="Bypass the shared LLM cache (same as MIRA_LLM_CACHE_BYPASS=1): "
                             "force fresh LLM calls even when cached results exist")
    parser.add_argument("--jev-prescreen", action="store_true",
                        help="Screen out clearly irrelevant papers with Jev (TypeSafe) before "
                             "PDF extraction and the LLM stages; needs TYPESAFE_API_KEY")
    args = parser.parse_args()

    if args.no_cache:
        os.environ["MIRA_LLM_CACHE_BYPASS"] = "1"

    if args.start_date and not args.end_date:
        parser.error("--end-date is required when --start-date is provided")

    from mira.config import load_config, make_llm_client
    from mira.fetch   import fetch_papers, extract_first_pages
    from mira.pipeline import run_pipeline
    from mira.media   import fetch_media
    from mira.report  import (
        generate_report, to_html, save_report, send_email, get_trend_section
    )

    try:
        print(f"[1/9] Loading config... profile={args.profile} mode={args.mode}")
        config = load_config(args.profile, args.mode, start_date=args.start_date, end_date=args.end_date)
        client = make_llm_client()
    except Exception as exc:
        sys.exit(f"[1/9] FAILED — config load error: {exc}")

    try:
        print(f"[2/9] Fetching arXiv papers ({config['start_date_iso']} → {config['end_date_iso']})...")
        papers = fetch_papers(config)
        print(f"  {len(papers)} papers found")
    except Exception as exc:
        sys.exit(f"[2/9] FAILED — arXiv fetch error: {exc}")

    screened: list[dict] = []
    if args.jev_prescreen:
        from mira import jev
        if not jev.enabled():
            sys.exit("[2/9] FAILED — --jev-prescreen needs TYPESAFE_API_KEY in .env or environment")
        if args.profile not in jev.PRESCREEN_CUTOFFS:
            print(f"  Jev pre-screen: no validated cutoff for {args.profile}; not screening")
        papers, screened = jev.prescreen(papers, config)
        print(f"  Jev pre-screen: {len(screened)} screened out, {len(papers)} continue to the LLM stages")

    try:
        print(f"[3/9] Extracting first-page PDFs... (this may take a few minutes)")
        papers = extract_first_pages(papers)
    except Exception as exc:
        sys.exit(f"[3/9] FAILED — first-page PDF extraction error: {exc}")

    try:
        print(f"[4/9] Classifying papers...")
        result = run_pipeline(papers, config, client)
        selected  = result["selected"]
        remaining = result["remaining"]
        total     = result["total_scanned"] + len(screened)
        print(f"  {total} processed, {len(selected) + len(remaining)} passed thresholds, {len(selected)} selected")
    except Exception as exc:
        sys.exit(f"[4/9] FAILED — pipeline (classify/select/analyze) error: {exc}")

    trend_section = ""
    if config["mode_cfg"].get("trend_enabled"):
        try:
            print(f"[5/9] Running trend analysis...")
            trend_section = get_trend_section(config, client)
        except Exception as exc:
            print(f"[5/9] WARNING — trend analysis error (continuing): {exc}")
            trend_section = ""
    else:
        print(f"[5/9] Trend analysis disabled for {args.mode} mode")

    media: list[dict] = []
    if not args.no_media:
        try:
            print(f"[6/9] Crawling media sources...")
            media = fetch_media(config, client)
            print(f"  {len(media)} articles after filtering")
        except Exception as exc:
            print(f"[6/9] WARNING — media crawl error (continuing): {exc}")
            media = []
    else:
        print(f"[6/9] Skipping media (--no-media)")

    try:
        print(f"[7/9] Generating report...")
        report_json = generate_report(selected, remaining, media, config, client, trend_section, total)
        subject = report_json.get("subject", f"MIRA Digest - {config['current_date']}")
        body    = report_json.get("body", "")
        html    = to_html(body, subject, config)
        html_path = save_report(html, report_json, config)
        print(f"  Report saved: {html_path}")
    except Exception as exc:
        sys.exit(f"[7/9] FAILED — report generation error: {exc}")

    if args.no_email:
        print(f"[8/9] Skipping email (--no-email)")
    else:
        try:
            print(f"[8/9] Sending email to {config['recipient_email']}...")
            send_email(html, subject, config)
            print(f"  Sent.")
        except Exception as exc:
            print(f"[8/9] WARNING — email send error: {exc}")
            print(f"  Email failed. Report saved at: {html_path}. You can open and forward it manually.")

    if not args.no_graph:
        try:
            print(f"[9/9] Ingesting into knowledge graph...")
            from mira.graph_ingest import ingest_report
            ingest_report(selected, media, config)
            print(f"  Done. Query at http://localhost:9621/webui/")
        except Exception as exc:
            print(f"[9/9] WARNING — graph ingestion error (continuing): {exc}")
    else:
        print(f"[9/9] Skipping graph ingestion (--no-graph)")


if __name__ == "__main__":
    main()

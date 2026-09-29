"""Research digest flow: the port of the n8n "Memory Innovation Research
Assistant" workflow. Each n8n stage group is a Prefect task so the UI shows
per-stage state, logs, timing and retries."""
from __future__ import annotations

from prefect import flow, task

from mira import pipeline
from mira.config import load_config, make_llm_client
from mira.fetch import extract_first_pages, fetch_papers


@task(name="load-config")
def load_run_config(**kwargs) -> dict:
    return load_config(**kwargs)


@task(name="fetch-arxiv")
def fetch(config: dict) -> list[dict]:
    papers = fetch_papers(config)
    print(f"{len(papers)} papers from arXiv ({config['start_date_iso']} → {config['end_date_iso']})")
    return papers


@task(name="jev-prescreen")
def prescreen(papers: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    from mira import jev
    if not jev.enabled():
        raise RuntimeError("jev_prescreen requested but TYPESAFE_API_KEY is not set")
    kept, screened = jev.prescreen(papers, config)
    print(f"Jev pre-screen: {len(screened)} screened out, {len(kept)} continue")
    return kept, screened


@task(name="first-page-pdfs")
def first_pages(papers: list[dict]) -> list[dict]:
    return extract_first_pages(papers)


@task(name="classify-select-analyze")
def run_pipeline(papers: list[dict], config: dict) -> dict | None:
    try:
        return pipeline.run_pipeline(papers, config, make_llm_client())
    except pipeline.NoEligiblePapers as e:
        # n8n stops without a report when nothing passes the thresholds.
        print(f"No eligible papers — no report this run ({e})")
        return None


@task(name="media", retries=1, retry_delay_seconds=60)
def media(config: dict):
    from mira.media import run_media
    return run_media(config, make_llm_client())


@task(name="report-and-trend")
def report(result: dict, media_out, config: dict, pdf: bool) -> dict:
    from mira.report import produce_report
    return produce_report(result, media_out, config, make_llm_client(), pdf=pdf)


@task(name="send-email", retries=2, retry_delay_seconds=120)
def deliver(report_out: dict, config: dict, recipients: list[str] | None, dry_run: bool) -> None:
    from mira.report import deliver_report
    deliver_report(report_out, config, recipients=recipients, dry_run=dry_run)


@task(name="graph-ingest")
def graph_ingest(selected: list[dict], media_out, config: dict) -> None:
    from mira.graph_ingest import ingest_report
    articles = (media_out or {}).get("media_articles", []) if isinstance(media_out, dict) else media_out or []
    ingest_report(selected, articles, config)


@flow(name="digest", log_prints=True)
def digest_flow(
    profile: str | None = None,
    mode: str | None = None,
    trigger_name: str | None = None,
    current_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int | None = None,
    max_limit: int | None = None,
    test_mode: bool = False,
    llm_cache_bypass: bool = False,
    trend_enabled: bool | None = None,
    include_media: bool = True,
    send_email: bool = True,
    recipients: list[str] | None = None,
    jev_prescreen: bool = False,
    graph: bool = False,
    pdf: bool = True,
) -> dict:
    """Fetch → first pages → classify → filter/rank → select → deep analysis →
    media → report + trend → email (→ optional LightRAG graph ingest).

    profile/mode: e.g. cxl-research/monthly; mode defaults to the profile's
    default. trigger_name routes like the n8n trigger names instead.
    current_date, lookback_days, max_limit, test_mode, llm_cache_bypass,
    trend_enabled: the n8n backfill-webhook overrides. test_mode writes the
    report under tests/ (excluded from trend history).
    send_email=False builds everything but sends nothing.
    """
    config = load_run_config(
        profile=profile, mode=mode, start_date=start_date, end_date=end_date,
        trigger_name=trigger_name, current_date=current_date, lookback_days=lookback_days,
        max_limit=max_limit, test_mode=test_mode, llm_cache_bypass=llm_cache_bypass,
        trend_enabled=trend_enabled)
    media_future = media.submit(config) if include_media else None

    papers = fetch(config)
    screened: list[dict] = []
    if jev_prescreen:
        papers, screened = prescreen(papers, config)
    papers = first_pages(papers)
    result = run_pipeline(papers, config)
    media_out = media_future.result() if media_future is not None else None
    if result is None:
        return {"status": "no-eligible-papers", "profile": config["profile_id"]}
    result["relevant_pool"] = result.get("selection_pool")
    result["total_scanned"] += len(screened)

    report_out = report(result, media_out, config, pdf)
    deliver(report_out, config, recipients, not send_email)
    if graph:
        graph_ingest(result["selected"], media_out, config)
    return {"status": "ok", "profile": config["profile_id"], "subject": report_out["subject"],
            "selected": len(result["selected"]), "record_path": str(report_out.get("record_path")),
            "html_path": str(report_out.get("html_path")), "pdf_path": str(report_out.get("pdf_path")),
            "emailed": send_email}

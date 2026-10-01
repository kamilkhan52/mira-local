"""Research digest flow: the port of the n8n "Memory Innovation Research
Assistant" workflow. Each n8n stage group is a Prefect task so the UI shows
per-stage state, logs, timing and retries. Every run also writes a summary
(stage times, LLM and Jev cost per stage, paper counts, selection) to
data/runs/."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from prefect import flow, task

from mira import pipeline, usage
from mira.config import load_config, make_llm_client
from mira.fetch import extract_first_pages, fetch_papers
from mira.paths import DATA_DIR


@task(name="load-config")
def load_run_config(**kwargs) -> dict:
    return load_config(**kwargs)


@task(name="fetch-arxiv")
def fetch(config: dict) -> list[dict]:
    with usage.stage("arXiv fetch"):
        papers = fetch_papers(config)
    print(f"{len(papers)} papers from arXiv ({config['start_date_iso']} → {config['end_date_iso']})")
    return papers


@task(name="jev-prescreen")
def prescreen(papers: list[dict], config: dict, level: str) -> tuple[list[dict], list[dict]]:
    from mira import jev
    if not jev.enabled():
        raise RuntimeError("jev_level requires TYPESAFE_API_KEY")
    with usage.stage("Jev screening"):
        kept, screened = jev.prescreen(papers, config, level=level)
    print(f"Jev {level}: {len(screened)} screened out, {len(kept)} continue to the LLM stages")
    return kept, screened


@task(name="jev-replace")
def jev_replace(papers: list[dict], config: dict) -> list[dict]:
    """Jev instead of the per-paper LLM stages: relevance for all papers, then
    first pages + credibility only for papers past the relevance threshold."""
    from mira import jev
    if not jev.enabled():
        raise RuntimeError("jev_level requires TYPESAFE_API_KEY")
    with usage.stage("Jev relevance"):
        papers = jev.replace_classification(papers, config)
    rmin = config["thresholds"]["relevance_score_min"]
    relevant = [p for p in papers if p["relevance_score"] >= rmin]
    with usage.stage("first-page PDFs"):
        extract_first_pages(relevant)
    with usage.stage("Jev credibility"):
        jev.judge_credibility_for(relevant, config)
    print(f"Jev replace: {len(relevant)} of {len(papers)} papers past relevance; credibility judged for those")
    return papers


@task(name="first-page-pdfs")
def first_pages(papers: list[dict]) -> list[dict]:
    with usage.stage("first-page PDFs"):
        return extract_first_pages(papers)


@task(name="classify-select-analyze")
def run_pipeline(papers: list[dict], config: dict, classify: bool = True) -> dict | None:
    try:
        with usage.stage("classify + select + analyze"):
            return pipeline.run_pipeline(papers, config, make_llm_client(), classify=classify)
    except pipeline.NoEligiblePapers as e:
        # n8n stops without a report when nothing passes the thresholds.
        print(f"No eligible papers — no report this run ({e})")
        return None


@task(name="media", retries=1, retry_delay_seconds=60)
def media(config: dict):
    from mira.media import run_media
    t0 = time.perf_counter()
    out = run_media(config, make_llm_client())
    return out, time.perf_counter() - t0


@task(name="report-and-trend")
def report(result: dict, media_out, config: dict, pdf: bool) -> dict:
    from mira.report import produce_report
    with usage.stage("report + trend + PDF"):
        return produce_report(result, media_out, config, make_llm_client(), pdf=pdf)


@task(name="send-email", retries=2, retry_delay_seconds=120)
def deliver(report_out: dict, config: dict, recipients: list[str] | None, dry_run: bool) -> None:
    from mira.report import deliver_report
    with usage.stage("email"):
        deliver_report(report_out, config, recipients=recipients, dry_run=dry_run)


@task(name="graph-ingest")
def graph_ingest(selected: list[dict], media_out, config: dict) -> None:
    from mira.graph_ingest import ingest_report
    articles = (media_out or {}).get("media_articles", []) if isinstance(media_out, dict) else media_out or []
    ingest_report(selected, articles, config)


def _short(p: dict) -> dict:
    return {"id": p.get("raw_id") or p.get("arxiv_id") or p.get("id"), "title": p.get("title"),
            "relevance_score": p.get("relevance_score"), "credibility_tier": p.get("credibility_tier")}


def _variant(jev_level: str) -> str:
    from mira import jev
    if jev_level == "off":
        return "all-llm"
    return f"jev-{jev_level}" if jev.BACKEND == "typesafe" else f"{jev.BACKEND}-{jev_level}"


def _write_summary(summary: dict) -> str:
    out = DATA_DIR / "runs"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{summary['started_at'].replace(':', '-')}__{summary['profile']}__{summary['variant']}.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return str(path)


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
    jev_level: str = "off",
    jev_backend: str = "typesafe",
    jev_prescreen: bool = False,
    subject_tag: str | None = None,
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
    jev_level: off (all LLM, like n8n) | prescreen (Jev skips clearly
    irrelevant papers) | gate (Jev decides relevance; LLM stages only for what
    it passes) | replace (Jev instead of the per-paper LLM stages).
    jev_backend: typesafe (Jev, hosted) | nimble | kev (local decision models
    serving the same API; see mira/jev.py BACKENDS).
    jev_prescreen=True is the same as jev_level="prescreen".
    subject_tag: text prepended to the email subject (e.g. for test runs).
    send_email=False builds everything but sends nothing.
    """
    from mira import jev
    if jev_prescreen and jev_level == "off":
        jev_level = "prescreen"
    if jev_level not in jev.JEV_LEVELS:
        raise ValueError(f"jev_level must be one of {jev.JEV_LEVELS}")
    jev.set_backend(jev_backend)
    if jev_level != "off" and not jev.cutoffs_for(profile or "memory-innovation"):
        raise ValueError(f"no calibrated {jev_backend} cutoffs for profile {profile}")
    usage.reset()
    started = datetime.now(timezone.utc)
    t_run = time.perf_counter()

    config = load_run_config(
        profile=profile, mode=mode, start_date=start_date, end_date=end_date,
        trigger_name=trigger_name, current_date=current_date, lookback_days=lookback_days,
        max_limit=max_limit, test_mode=test_mode, llm_cache_bypass=llm_cache_bypass,
        trend_enabled=trend_enabled)
    media_future = media.submit(config) if include_media else None

    try:
        return _run(config, media_future, jev_level, recipients, send_email, subject_tag, graph, pdf,
                    started, t_run)
    except Exception as e:
        # Keep the evidence (stage times and spend so far) for failed runs too.
        _write_summary({"started_at": started.isoformat(timespec="seconds"),
                        "profile": config["profile_id"], "mode": config["mode"],
                        "variant": _variant(jev_level),
                        "status": "failed", "error": repr(e),
                        "wall_seconds": round(time.perf_counter() - t_run, 1), **usage.snapshot()})
        raise


def _run(config, media_future, jev_level, recipients, send_email, subject_tag, graph, pdf, started, t_run):
    papers = fetch(config)
    fetched = len(papers)
    screened: list[dict] = []
    if jev_level in ("prescreen", "gate"):
        papers, screened = prescreen(papers, config, jev_level)
    if jev_level == "replace":
        papers = jev_replace(papers, config)
        result = run_pipeline(papers, config, classify=False)
    else:
        papers = first_pages(papers)
        result = run_pipeline(papers, config)
    media_out, media_seconds = media_future.result() if media_future is not None else (None, 0.0)

    summary = {
        "started_at": started.isoformat(timespec="seconds"), "profile": config["profile_id"],
        "mode": config["mode"], "variant": _variant(jev_level),
        "window": [config["start_date_iso"], config["end_date_iso"]],
        "papers_fetched": fetched, "papers_screened_by_jev": len(screened),
        "media_seconds": round(media_seconds, 1),
    }
    if result is None:
        summary.update({"status": "no-eligible-papers", "wall_seconds": round(time.perf_counter() - t_run, 1),
                        **usage.snapshot()})
        summary["summary_path"] = _write_summary(summary)
        return summary
    result["relevant_pool"] = result.get("selection_pool")
    result["total_scanned"] += len(screened)

    report_out = report(result, media_out, config, pdf)
    if subject_tag:
        report_out["subject"] = f"{subject_tag} {report_out['subject']}"
    deliver(report_out, config, recipients, not send_email)
    if graph:
        graph_ingest(result["selected"], media_out, config)

    summary.update({
        "status": "ok", "subject": report_out["subject"],
        "wall_seconds": round(time.perf_counter() - t_run, 1),
        "papers_classified": len(result["classified"]),
        "papers_passing": len(result["ranked"]),
        "ranked_pool": [_short(p) for p in result["ranked"]],
        "selected": [_short(p) for p in result["selected"]],
        "record_path": str(report_out.get("record_path")), "html_path": str(report_out.get("html_path")),
        "pdf_path": str(report_out.get("pdf_path")), "emailed": send_email,
        **usage.snapshot(),
    })
    summary["summary_path"] = _write_summary(summary)
    print(f"Run summary: {summary['summary_path']}  wall {summary['wall_seconds']}s  "
          f"LLM ${summary['llm_cost_usd']}  Jev ${summary['jev_cost_usd']}")
    return {k: summary[k] for k in ("status", "profile", "variant", "subject", "wall_seconds",
                                    "llm_cost_usd", "jev_cost_usd", "papers_passing", "summary_path")}

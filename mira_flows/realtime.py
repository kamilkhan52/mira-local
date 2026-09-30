"""MIRA Live: hourly Jev-triaged alerts (wraps mira.realtime)."""
from __future__ import annotations

from prefect import flow, task

from mira import jev, realtime


@task(retries=1, retry_delay_seconds=60)
def crawl_news(lookback_days: int) -> list[dict]:
    return realtime.crawl_news(lookback_days)


@task(retries=1, retry_delay_seconds=120)
def alert_profile(profile_id: str, settings: dict, news: list[dict],
                  llm_summaries: bool, dry_run: bool) -> dict:
    client = None
    if llm_summaries:
        from mira.config import make_llm_client
        try:
            client = make_llm_client()
        except Exception as e:  # noqa: BLE001 — fall back to Jev key sentences
            print(f"No LLM client ({e}); summaries will be key sentences")
    return realtime.run_profile(profile_id, settings, news, client, dry_run=dry_run)


@flow(name="realtime", log_prints=True)
def realtime_flow(profiles: list[str] | None = None, include_news: bool = True,
                  include_papers: bool | None = None,
                  llm_summaries: bool = True, dry_run: bool = False) -> list[dict]:
    """Breaking news (and optionally new arXiv papers), triaged by Jev, emailed
    to subscribers. include_papers: default from configs/realtime.json (off).

    profiles: default = every profile with subscribers in configs/realtime.json.
    dry_run: judge and write the alert HTML but send nothing and keep the ledger.
    """
    if not jev.enabled():
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    settings = realtime.load_settings()
    if include_papers is not None:
        settings["include_papers"] = include_papers
    profiles = profiles or [p for p, subs in settings["subscribers"].items() if subs]
    news = crawl_news(settings["news_lookback_days"]) if include_news else []
    results = []
    for pid in profiles:
        results.append(alert_profile(pid, settings, news, llm_summaries, dry_run))
    return results

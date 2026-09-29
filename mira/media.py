from __future__ import annotations
from mira.paths import TEMP_DIR, CRAWLERS_DIR
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
_MICRON_DIR = CRAWLERS_DIR


def _run_crawler(script_name: str, config: dict, source_key: str, list_url: str | None = None) -> list[dict]:
    output_file = TEMP_DIR / f"{source_key}-latest.json"
    (TEMP_DIR).mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "DATE_FROM": config["start_date_iso"],
        "DATE_TO": config["end_date_iso"],
        "OUTPUT_PATH": str(output_file),
        "MAX_ARTICLES": "20",
    }
    if list_url:
        env["LIST_URL"] = list_url

    try:
        result = subprocess.run(
            ["npx", "tsx", script_name],
            env=env,
            cwd=str(_MICRON_DIR),
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  WARNING: {source_key} crawler error — {e}")
        return []

    if result.returncode != 0:
        print(f"  WARNING: {source_key} crawler failed — {result.stderr[:150]}")
        return []

    if output_file.exists():
        try:
            return json.loads(output_file.read_text())
        except Exception:
            return []
    return []


def _normalize_articles(raw: list[dict], source: str) -> list[dict]:
    return [
        {
            "source": source,
            "title": a.get("title") or a.get("listTitle", ""),
            "url": a.get("url", ""),
            "content": a.get("content", ""),
            "date": a.get("listDate") or str(a.get("crawledAt") or "")[:10],
        }
        for a in raw
        if a.get("title") or a.get("listTitle")
    ]


def _filter_articles(articles: list[dict], config: dict, client) -> list[dict]:
    from mira.config import llm_call, model_for, record_parse_failure

    model = model_for(config, "media_selection")
    guidance = config["media"].get("selection_guidance", "")
    system = "You are filtering news articles for a research digest. Return only valid JSON."
    user = (
        f"Select the 10 most relevant articles for a {config['topic']['focus']} digest.\n"
        f"Selection guidance: {guidance}\n\n"
        f"Articles:\n{json.dumps(articles, indent=2)}\n\n"
        f"Return a JSON array of selected article objects with the same fields."
    )
    raw = llm_call(client, model, system, user)
    try:
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\n?```\s*$", "", raw.strip(), flags=re.MULTILINE)
        return json.loads(raw.strip())
    except Exception:
        record_parse_failure("media_selection")
        return articles[:10]


def _summarize_articles(articles: list[dict], config: dict, client) -> list[dict]:
    from mira.config import llm_call, model_for, record_parse_failure

    model = model_for(config, "media_summary")
    system = "You are summarizing news articles. Return only valid JSON."
    user = (
        f"Summarize each article in 2-3 sentences focusing on its significance for "
        f"{config['topic']['focus']}.\n\n"
        f"Articles:\n{json.dumps(articles, indent=2)}\n\n"
        f"Return a JSON array where each object has the original fields plus a 'summary' field."
    )
    raw = llm_call(client, model, system, user)
    try:
        raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\n?```\s*$", "", raw.strip(), flags=re.MULTILINE)
        return json.loads(raw.strip())
    except Exception:
        record_parse_failure("media_summary")
        for a in articles:
            a["summary"] = a.get("content", "")[:200]
        return articles


def fetch_media(config: dict, client) -> list[dict]:
    eetimes_url = config["media"].get("eetimes", {}).get("list_url")
    crawlers = [
        ("ee-times-crawler.ts",    "eetimes",      "EE Times",     eetimes_url),
        ("semianalysis-crawler.ts","semianalysis", "SemiAnalysis", None),
        ("trendforce-crawler.ts",  "trendforce",   "TrendForce",   None),
        ("digitimes-crawler.ts",   "digitimes",    "Digitimes",    None),
    ]

    all_articles: list[dict] = []
    for script, key, source, list_url in crawlers:
        raw = _run_crawler(script, config, key, list_url)
        all_articles.extend(_normalize_articles(raw, source))
        print(f"  {source}: {len(raw)} articles")

    if not all_articles:
        return []

    if len(all_articles) > 15:
        all_articles = _filter_articles(all_articles, config, client)

    return _summarize_articles(all_articles, config, client)

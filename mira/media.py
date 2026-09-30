"""Media intelligence: port of the n8n "Media - *" nodes.

Crawl (EE Times, SemiAnalysis, TrendForce; Digitimes opt-in) -> map each
source to the pipeline shape -> keep articles dated inside the run window ->
when there are more than 15, let the LLM pick 5 -> summarize -> the same
object "Media - Prepare Media Output For Parent" hands to the report stage.
"""
from __future__ import annotations
from mira.paths import TEMP_DIR, CRAWLERS_DIR
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
_MICRON_DIR = CRAWLERS_DIR

# (script, source key, display name). n8n crawls exactly these three.
MEDIA_SOURCES = (
    ("ee-times-crawler.ts", "eetimes", "EE Times"),
    ("semianalysis-crawler.ts", "semianalysis", "SemiAnalysis"),
    ("trendforce-crawler.ts", "trendforce", "TrendForce"),
)
# CLI-only source, off by default: profile media.digitimes.enabled, env
# MIRA_MEDIA_DIGITIMES=1, or run_media(..., include_digitimes=True).
DIGITIMES_SOURCE = ("digitimes-crawler.ts", "digitimes", "Digitimes")

MEDIA_MAX_ARTICLES = 0          # n8n: MAX_ARTICLES=0 (everything in the window)
MEDIA_CRAWL_TIMEOUT = 600       # n8n: `timeout 600 sh -c ...`
MEDIA_SELECT_THRESHOLD = 15     # "Media - IF More Than 15 Articles"
MEDIA_MAX_SELECT = 5            # "Media - Build Selection Prompt" maxSelect
MEDIA_MAX_SUMMARY = 6000        # Map * to Pipeline maxSummary
MEDIA_SELECTION_SNIPPET = 500
MEDIA_SUMMARIZE_CONTENT = 3000
DEFAULT_TOPIC_FOCUS = "memory technology and semiconductors"


# --------------------------------------------------------------- crawling --

def _run_crawler(script_name: str, config: dict, source_key: str, list_url: str | None = None,
                 *, max_articles: int | str = 20, timeout: float = 180) -> list[dict]:
    """Run one TypeScript crawler with `npx tsx` and return its JSON array.

    The output file is removed first so a failed crawl can never serve the
    previous run's articles (n8n's `cat` step would). A non-zero exit still
    reads whatever the crawler managed to write, like n8n's continueOnFail.
    Defaults (20 articles, 180 s) are the realtime monitor's; the digest
    passes n8n's 0 / 600 s."""
    output_file = TEMP_DIR / f"{source_key}-latest.json"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        output_file.unlink()
    except FileNotFoundError:
        pass

    env = {
        **os.environ,
        "DATE_FROM": config["start_date_iso"],
        "DATE_TO": config["end_date_iso"],
        "OUTPUT_PATH": str(output_file),
        "OUTPUT_DIR": str(TEMP_DIR / "crawler-output"),
        "MAX_ARTICLES": str(max_articles),
        # Deprecated alias; never let an inherited value trigger a POST.
        # CRAWLER_WEBHOOK_URL (if the operator set one) is passed through.
        "N8N_WEBHOOK_URL": "",
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
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  WARNING: {source_key} crawler error — {e}")
        result = None

    if result is not None and result.returncode != 0:
        print(f"  WARNING: {source_key} crawler failed — {result.stderr[-300:]}")

    if output_file.exists():
        try:
            data = json.loads(output_file.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []
    return []


def _normalize_articles(raw: list[dict], source: str) -> list[dict]:
    """Legacy flat shape (realtime monitor, graph ingest)."""
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


# ------------------------------------------------------------ JS helpers --

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _js_date(value) -> datetime | None:
    """`new Date(String(value).trim())` for the formats the crawlers and
    models produce, as an aware UTC datetime (None when invalid). Date-only
    ISO strings are UTC midnight; other date-only forms ("September 25,
    2026", "09.17.2026", "09/17/2026") are LOCAL midnight — V8's rules."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    local_tz = datetime.now().astimezone().tzinfo
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    iso = re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?", s,
                       flags=re.IGNORECASE)
    if iso:  # V8 also accepts lowercase t/z (normDate lowercases first)
        try:
            dt = datetime.fromisoformat(s.upper().replace("Z", "+00:00").replace(" ", "T"))
        except ValueError:
            return None
        return (dt if dt.tzinfo else dt.replace(tzinfo=local_tz)).astimezone(timezone.utc)
    m = re.fullmatch(r"([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})", s)
    if m and m.group(1)[:3].lower() in _MONTHS:
        month, day, year = _MONTHS[m.group(1)[:3].lower()], int(m.group(2)), int(m.group(3))
    else:
        m = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", s)
        if not m:
            try:
                return parsedate_to_datetime(s).astimezone(timezone.utc)
            except (TypeError, ValueError, IndexError):
                return None
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(year, month, day, tzinfo=local_tz).astimezone(timezone.utc)
    except ValueError:
        return None


def _js_iso(dt: datetime) -> str:
    """Date.prototype.toISOString()."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{dt.microsecond // 1000:03d}Z"


def _to_published(value) -> str:
    dt = _js_date(value)
    return _js_iso(dt) if dt else ""


def _parse_date_only(value) -> str:
    """Media - Split Summary and Articles: parseDateOnly."""
    if not value:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    dt = _js_date(s)
    return _js_iso(dt)[:10] if dt else ""


def _norm_date(value) -> str:
    """Media - Parse Summarized Articles: normDate."""
    if not value or not isinstance(value, str):
        return ""
    t = value.strip().lower()
    if not t or "not" in t or t == "n/a" or t == "null":
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return t
    dt = _js_date(t)
    return _js_iso(dt)[:10] if dt else ""


def _apply_template(template: str, variables: dict) -> str:
    from mira.pipeline import _apply_template_n8n
    return _apply_template_n8n(template, variables)


# --------------------------------------------------------------- mapping --

def map_to_pipeline(raw: list[dict], source_key: str, run_date: str) -> list[dict]:
    """Media - Map <Source> to Pipeline."""
    out = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict):
            continue
        out.append({
            "id": r.get("url") or "",
            "title": (r.get("title") or r.get("listTitle") or "").strip(),
            "summary": (r.get("content") or "").strip()[:MEDIA_MAX_SUMMARY],
            "author": [r["listAuthor"]] if r.get("listAuthor") else [],
            "published": _to_published(r.get("listDate")),
            "category": [source_key],
            "run_date": run_date,
            "source": source_key,
            "url": r.get("url") or "",
            "listAuthor": r.get("listAuthor") or "",
            "listDate": r.get("listDate") or "",
        })
    return out


def filter_to_window(articles: list[dict], config: dict) -> dict:
    """Media - Split Summary and Articles: keep articles whose date (date ||
    listDate || published) falls in the run window. Undated articles are
    dropped. Returns {summary, articles, article_count}."""
    date_from = config.get("start_date_iso") or ""
    date_to = config.get("end_date_iso") or ""
    period_range = config.get("period_range") or ""
    if " to " in period_range:
        lo, hi = period_range.split(" to ", 1)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", lo or ""):
            date_from = lo
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", hi or ""):
            date_to = hi

    def in_window(a: dict) -> bool:
        if not date_from or not date_to:
            return True
        d = _parse_date_only(a.get("date") or a.get("listDate") or a.get("published") or "")
        return bool(d) and date_from <= d <= date_to

    kept = [a for a in articles if in_window(a)]
    rng = f"{date_from} to {date_to}" if date_from and date_to else period_range
    return {"summary": {"period_range": rng, "date_from": date_from, "date_to": date_to,
                        "total_articles": len(kept)},
            "articles": kept, "article_count": len(kept)}


# ------------------------------------------------------------- prompts --

def _topic_focus(config: dict) -> str:
    return (config.get("topic") or {}).get("focus") or config.get("topic_focus") or DEFAULT_TOPIC_FOCUS


def _media_prompts(config: dict) -> dict:
    return (config.get("prompts") or {}).get("media") or {}


def build_selection_prompt(item: dict, config: dict) -> tuple[str, str]:
    """Media - Build Selection Prompt: (user, system). The profile's
    media.selection_guidance is NOT used — n8n ignores it too; override the
    prompt with prompts.media.selection_user / selection_system."""
    topic_focus = _topic_focus(config)
    period_label = config.get("period_label") or "this period"
    period_range = (item.get("summary") or {}).get("period_range") or config.get("period_range") or "the period"
    articles = item.get("articles") or []
    articles_json = json.dumps([{
        "index": i, "source": a.get("source"), "title": a.get("title"),
        "summary": (a.get("summary") or "")[:MEDIA_SELECTION_SNIPPET],
        "url": a.get("url"), "listDate": a.get("listDate"),
    } for i, a in enumerate(articles)], indent=2, ensure_ascii=False)
    default_system = (
        f"Select the top {MEDIA_MAX_SELECT} most newsworthy articles about {topic_focus}. "
        f"Try to include articles from multiple sources (EE Times, SemiAnalysis, TrendForce). "
        f"Output ONLY valid JSON.")
    default_user = (
        f"Select up to {MEDIA_MAX_SELECT} articles from the following {len(articles)} articles about "
        f"{topic_focus} for {period_label} ({period_range}).\n\nArticles:\n{articles_json}\n\n"
        f'Output ONLY valid JSON: {{"selected_indices": [0, 1, 2, ...]}}')
    variables = {"topic_focus": topic_focus, "period_label": period_label,
                 "period_range": period_range, "article_count": len(articles),
                 "max_select": MEDIA_MAX_SELECT, "articles_json": articles_json}
    prompts = _media_prompts(config)
    # n8n runs applyTemplate over the default prompt as well, so any "{{x}}"
    # inside an article title/snippet is blanked — reproduced.
    system = _apply_template(prompts.get("selection_system") or default_system, variables)
    user = _apply_template(prompts.get("selection_user") or default_user, variables)
    return user, system


def apply_selection(item: dict, raw_output) -> dict:
    """Media - Apply Selection: selected_indices (or selected_papers[].index),
    else a regex over the raw text; nothing usable -> the first 5."""
    articles = item.get("articles") or []
    indices: list = []
    parsed = None
    if isinstance(raw_output, dict):
        parsed = raw_output
    elif isinstance(raw_output, str):
        from mira.pipeline import _parse_json_object
        try:
            parsed = _parse_json_object(raw_output)
        except ValueError:
            parsed = None
    if isinstance(parsed, dict):
        if isinstance(parsed.get("selected_indices"), list):
            indices = parsed["selected_indices"]
        elif isinstance(parsed.get("selected_papers"), list):
            indices = [p.get("index") or 0 for p in parsed["selected_papers"] if isinstance(p, dict)]
            indices = [i for i in indices if isinstance(i, (int, float)) and not isinstance(i, bool)]
    if not indices:
        text = raw_output if isinstance(raw_output, str) else json.dumps(raw_output or {})
        m = re.search(r'"selected_indices"\s*:\s*\[([^\]]+)\]', str(text))
        if m:
            try:
                indices = json.loads("[" + m.group(1) + "]")
            except json.JSONDecodeError:
                indices = []
    wanted = [i for i in indices if isinstance(i, (int, float)) and not isinstance(i, bool)]
    selected = [a for i, a in enumerate(articles) if i in wanted]
    if not selected:
        selected = articles[:MEDIA_MAX_SELECT]
    return {"summary": item.get("summary"), "articles": selected, "article_count": len(selected)}


def build_summarize_prompt(item: dict, config: dict) -> tuple[str, str]:
    """Media - Build Summarize Prompt: (user, system)."""
    topic_focus = _topic_focus(config)
    period_label = config.get("period_label") or "this period"
    period_range = (item.get("summary") or {}).get("period_range") or config.get("period_range") or "the period"
    articles = item.get("articles") or []
    listing = "\n\n".join(
        f"--- Article {i + 1} [{(a.get('source') or '').upper()}] ---\n"
        f"Title: {a.get('title') or 'Untitled'}\n"
        f"Date: {a.get('listDate') or ''}\n"
        f"URL: {a.get('url') or ''}\n"
        f"Content:\n{(a.get('summary') or '')[:MEDIA_SUMMARIZE_CONTENT]}"
        for i, a in enumerate(articles))
    default_user = (
        f"Normalize each news article for {topic_focus} in {period_label} ({period_range}). "
        f"For each, output one object in the SAME ORDER with:\n"
        f"- title: article title (unchanged)\n"
        f"- short_summary: 2-3 clear sentences about the main point (no author/date lines, no tables)\n"
        f"- date: YYYY-MM-DD or empty string\n"
        f"- source: copy the source tag from the article header (eetimes / semianalysis / trendforce)\n\n"
        f"Return ONLY a JSON array. No code fences.\n\n{listing}")
    default_system = ("Output only a JSON array of objects with keys: title, short_summary, date, source. "
                      "Same number and order as articles. No markdown.")
    variables = {"topic_focus": topic_focus, "period_label": period_label, "period_range": period_range,
                 "article_count": len(articles), "articles_text": listing}
    prompts = _media_prompts(config)
    user = _apply_template(prompts.get("summarize_user") or default_user, variables)
    system = _apply_template(prompts.get("summarize_system") or default_system, variables)
    return user, system


def parse_summarized(item: dict, raw_output) -> dict:
    """Media - Parse Summarized Articles: merge the model's per-article
    objects (by position) over the originals."""
    articles = item.get("articles") or []
    if not articles:
        return {"summary": item.get("summary"), "articles": [], "article_count": 0}
    parsed: list = []
    raw = ""
    if isinstance(raw_output, list):
        parsed = raw_output
    elif isinstance(raw_output, dict):
        if isinstance(raw_output.get("articles"), list):
            parsed = raw_output["articles"]
        else:
            raw = json.dumps(raw_output)
    elif isinstance(raw_output, str):
        raw = raw_output
    if raw:
        cleaned = re.sub(r"^\s*```(?:json)?\s*\n?", "", raw, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.IGNORECASE).strip()
        try:
            p = json.loads(cleaned)
            if isinstance(p, list):
                parsed = p
            elif isinstance(p, dict) and p.get("articles"):
                parsed = p["articles"]
            else:
                parsed = [p]
        except json.JSONDecodeError:
            parsed = []
    merged = []
    for i, orig in enumerate(articles):
        s = parsed[i] if i < len(parsed) and isinstance(parsed[i], dict) else {}
        title = s.get("title") if s.get("title") is not None else orig.get("title")
        short = s.get("short_summary")
        merged.append({
            "title": title if title is not None else "",
            "short_summary": short if short is not None else (orig.get("summary") or "")[:400],
            "date": _norm_date(s.get("date")) or _norm_date(orig.get("listDate") or orig.get("date") or ""),
            "url": orig.get("url") if orig.get("url") is not None else "",
            "source": s.get("source") or orig.get("source") or "unknown",
        })
    return {"summary": item.get("summary"), "articles": merged, "article_count": len(merged)}


def prepare_media_output(item: dict) -> dict:
    """Media - Prepare Media Output For Parent."""
    summary = item.get("summary") or {}
    articles = item.get("articles") if isinstance(item.get("articles"), list) else []
    period_range = summary.get("period_range") or " to ".join(
        x for x in (summary.get("date_from"), summary.get("date_to")) if x) or ""
    if articles:
        lines = []
        for idx, a in enumerate(articles):
            title = a.get("title") or f"Untitled {idx + 1}"
            link = f"[{title}]({a['url']})" if a.get("url") else title
            source = str(a.get("source") or "unknown")
            when = f" ({a['date']})" if a.get("date") else ""
            text = str(a.get("short_summary") or "").strip()
            lines.append(f"- {link} — {source}{when}\n  {text}")
        markdown = "\n".join(lines)
    else:
        markdown = "- No media articles available for this period."
    return {
        "media_intelligence": {"period_range": period_range, "article_count": len(articles),
                               "articles": articles},
        "media_period_range": period_range,
        "media_article_count": len(articles),
        "media_articles": articles,
        "media_markdown": markdown,
    }


# ------------------------------------------------------------ pipeline --

def _digitimes_enabled(config: dict, include_digitimes: bool | None) -> bool:
    if include_digitimes is not None:
        return include_digitimes
    if ((config.get("media") or {}).get("digitimes") or {}).get("enabled"):
        return True
    return os.environ.get("MIRA_MEDIA_DIGITIMES", "").strip().lower() in ("1", "true", "yes", "on")


def crawl_sources(config: dict, include_digitimes: bool | None = None) -> list[dict]:
    """Run the crawlers (in parallel) and map each to the pipeline shape, in
    source order. Profile media list_url(s)/list_url_template/source_caps are
    ignored, as in n8n (each crawler's own default list page is used)."""
    sources = list(MEDIA_SOURCES)
    if _digitimes_enabled(config, include_digitimes):
        sources.append(DIGITIMES_SOURCE)
    run_date = config.get("current_date") or date.today().isoformat()

    def crawl(src):
        script, key, name = src
        raw = _run_crawler(script, config, key, max_articles=MEDIA_MAX_ARTICLES,
                           timeout=MEDIA_CRAWL_TIMEOUT)
        print(f"  {name}: {len(raw)} articles")
        return map_to_pipeline(raw, key, run_date)

    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        mapped = list(pool.map(crawl, sources))
    return [a for chunk in mapped for a in chunk]


def run_media(config: dict, client, include_digitimes: bool | None = None,
              articles: list[dict] | None = None) -> dict:
    """The whole media branch. Returns Prepare Media Output For Parent's
    object: {media_intelligence, media_period_range, media_article_count,
    media_articles: [{title, short_summary, date, url, source}], media_markdown}.
    `articles` (pipeline-shaped) skips crawling — for tests and reruns."""
    from mira.config import llm_call, model_for

    if articles is None:
        articles = crawl_sources(config, include_digitimes)
    item = filter_to_window(articles, config)

    if item["article_count"] > MEDIA_SELECT_THRESHOLD:
        user, system = build_selection_prompt(item, config)
        try:
            raw = llm_call(client, model_for(config, "media_selection"), system, user, schema="media_selection")
        except Exception as e:  # noqa: BLE001 — n8n routes the error output to Apply Selection
            print(f"  WARNING: media selection failed — {e}. Using the first {MEDIA_MAX_SELECT}.")
            raw = {}
        item = apply_selection(item, raw)
    else:
        item = {"summary": item["summary"], "articles": item["articles"],
                "article_count": len(item["articles"])}

    if item["articles"]:
        user, system = build_summarize_prompt(item, config)
        try:
            raw = llm_call(client, model_for(config, "media_summary"), system, user)
        except Exception as e:  # noqa: BLE001 — error output also feeds the parser
            print(f"  WARNING: media summarization failed — {e}. Using article excerpts.")
            raw = ""
        item = parse_summarized(item, raw)
    return prepare_media_output(item)


def fetch_media(config: dict, client, include_digitimes: bool | None = None) -> list[dict]:
    """Backward-compatible list of media articles ({title, short_summary,
    date, url, source} plus a `summary` alias of short_summary for older
    consumers such as graph ingestion). run_media() returns the full
    n8n-shaped object."""
    result = run_media(config, client, include_digitimes)
    return [{**a, "summary": a.get("short_summary", "")} for a in result["media_articles"]]

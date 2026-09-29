"""Report generation and delivery — port of the back half of the n8n workflow
"Memory Innovation Research Assistant" (iz3yMcSlkWIQhRmn).

n8n node order reproduced by ``produce_report`` (see _connections.json):

  Compute Paper Statistics ─┐
  Prepare Data For Report → Aggregate ─┼→ Wait For All Inputs → Combine Report Data
  Split Selected and Remaining Papers ─┘
  → Build Report Prompt → Generate Report (+ Parse Report Output) → Validate Report Output
  → Persist Current Report → Ensure Report Directories → Write Report File
  → Fetch Prior Reports → Load Prior Reports → Trend Enabled?
      yes → Build Trend Prompt → Trend Analysis Agent → Append Trend Section ─┐
      no ────────────────────────────────────────────────────────────────────┤
  → Post Process Email Data2 → Convert Markdown to HTML2 → Apply Email Styling
  → Prepare & Convert HTML → Write HTML File → Generate PDF → (Read PDF File)
and ``deliver_report``: Build Email with PDF → Send Email2.

Public API
  produce_report(pipeline_result, media, config, client, *, now=None, pdf=True) -> dict
  deliver_report(result, config, recipients=None, *, dry_run=False, attach_pdf=True) -> dict
  send_email(html, subject, config, recipients=None, attachments=None, dry_run=False)
  to_html(body_markdown, subject, config)      (legacy CLI/realtime renderer, unchanged)

The legacy CLI functions (generate_report, get_trend_section, save_report,
_build_stats, match_themes, format_theme_label) are kept at the bottom of this
module so run.py keeps working until it switches to produce_report.
"""
from __future__ import annotations

import functools
import json
import math
import os
import re
import smtplib
import ssl
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import markdown as md

from mira.graph_ingest import normalize_topic
from mira.paths import CRAWLERS_DIR, REPORT_FILES
from mira.render import (
    UNDEF, js_iso_now, js_json, js_null_number, js_number, js_round,
    js_str, markdown_to_html, resolve_palette, style_email,
)

ROOT = Path(__file__).parent.parent

# n8n "OpenRouter Chat Model5" (Parse Report Output) and "OpenRouter Chat
# Model6" (Structured Output Parser (Trend) autoFix) are hard-coded to this
# model in the live workflow and have no llm_models key. Optional overrides:
# llm_models.report_parse / llm_models.trend_fix.
PARSE_REPORT_MODEL_DEFAULT = "anthropic/claude-sonnet-5"
TREND_FIX_MODEL_DEFAULT = "anthropic/claude-sonnet-5"

# Persist Current Report hard-codes the prior-report window (the profile's
# modes.<mode>.trend_window_days is NOT read by n8n).
PRIOR_REPORT_WINDOW_DAYS = 60

PDF_ATTACHMENT_NAME = "research-report.pdf"  # n8n Build Email with PDF


class ReportValidationError(ValueError):
    """n8n "Validate Report Output" threw — the run stops (no record, no email)."""


class ReportParseError(ValueError):
    """Neither the report text nor the Parse Report Output fallback chain
    yielded a {subject, body} object (n8n: the chain's output parser throws)."""


# --------------------------------------------------------------------------
# JS-semantics helpers local to the Code-node ports
# --------------------------------------------------------------------------

def _truthy(v) -> bool:
    """JavaScript truthiness ([] and {} are truthy, NaN/0/''/null falsy)."""
    if v is None or v is UNDEF or v is False:
        return False
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return not (v == 0 or math.isnan(v))
    if isinstance(v, str):
        return v != ""
    return True


def _or(*vals):
    """``a || b || c``."""
    for v in vals[:-1]:
        if _truthy(v):
            return v
    return vals[-1]


def _nullish(*vals):
    """``a ?? b ?? c``."""
    for v in vals[:-1]:
        if v is not None and v is not UNDEF:
            return v
    return vals[-1]


def _get(obj, *path):
    """Optional chaining ``obj?.a?.b`` → value or UNDEF."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return UNDEF
        cur = cur[key]
    return cur


def _to_num(v):
    """n8n ``toNum``: ``Number(v)`` when finite, else None (JS null)."""
    if v is UNDEF:
        return None
    n = js_null_number(v)
    return n if math.isfinite(n) else None


def _round1(v) -> float:
    return js_round((v or 0) * 10) / 10


def _normalize_id(raw) -> str:
    s = js_str(raw) if _truthy(raw) else ""
    s = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", s)
    s = re.sub(r"\.pdf$", "", s, flags=re.I)
    return s.strip()


def _slug(text) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower())
    return re.sub(r"^-+|-+$", "", s)


def _key(v) -> str:
    """Property key a JS object would use for ``counts[v]``."""
    return v if isinstance(v, str) else js_str(v)


_ARRAY_INDEX_RE = re.compile(r"0|[1-9]\d*")


def _js_entries(d: dict) -> list[tuple[str, object]]:
    """``Object.entries`` order: array-index keys ascending, then insertion."""
    idx = sorted((k for k in d if _ARRAY_INDEX_RE.fullmatch(k) and int(k) < 2**32 - 1), key=int)
    rest = [k for k in d if k not in set(idx)]
    return [(k, d[k]) for k in idx + rest]


def _js_sort(items: list, cmp) -> list:
    """Stable ``Array.prototype.sort(cmp)``; NaN comparator results count as 0."""
    def wrapped(a, b):
        r = cmp(a, b)
        if r is None or (isinstance(r, float) and math.isnan(r)):
            return 0
        return -1 if r < 0 else (1 if r > 0 else 0)
    return sorted(items, key=functools.cmp_to_key(wrapped))


def _apply_template(template, variables: dict) -> str:
    """n8n applyTemplate: ``{{ key }}`` → value; unknown/null → ''."""
    def repl(m: re.Match) -> str:
        v = variables.get(m.group(1).strip(), UNDEF)
        return "" if v is None or v is UNDEF else js_str(v)
    return re.sub(r"\{\{\s*([^}]+)\s*\}\}", repl, template or "")


def _parse_bool(value, fallback: bool = False) -> bool:
    if value is None or value == "":
        return fallback
    return value in (True, "true", 1, "1", "yes", "on") and value is not False


# --------------------------------------------------------------------------
# Set Run Mode — the subset the back half reads, derived from config
# --------------------------------------------------------------------------

def _duration_label(days) -> str:
    d = js_number(days)
    d = 0 if not math.isfinite(d) else d
    if d <= 0:
        return "recent period"
    if d == 1:
        return "last day"
    if d == 7:
        return "last week"
    if d % 30 == 0:
        months = js_round(d / 30)
        return "last month" if months == 1 else f"last {js_str(months)} months"
    if d % 7 == 0 and d < 60:
        weeks = js_round(d / 7)
        return "last week" if weeks == 1 else f"last {js_str(weeks)} weeks"
    return f"last {js_str(d)} days"


def _title_case(text: str) -> str:
    return " ".join((w[0].upper() + w[1:]) if w else "" for w in (text or "").split(" ")).strip()


def resolve_run_mode(config: dict) -> dict:
    """The "Set Run Mode" fields the back half reads, from our config dict.

    Explicit config keys win (period_label, period_title, period_range,
    test_mode, trend_enabled, current_date); anything missing is derived the
    way Set Run Mode derives it. digestLabel is ALWAYS the n8n formula
    ``"<topic.name> Digest (<periodTitle>)"`` (n8n ignores modes.*.digest_label).
    """
    topic = config.get("topic") or {}
    mode = config.get("mode") or config.get("default_mode") or "weekly"
    mode_cfg = config.get("mode_cfg") or {}

    current_date = str(config.get("current_date") or config.get("end_date_iso") or "")[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", current_date):
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lookback = config.get("lookback_days")
    if lookback is None and config.get("start_date_iso") and config.get("end_date_iso"):
        try:
            lookback = (date.fromisoformat(config["end_date_iso"])
                        - date.fromisoformat(config["start_date_iso"])).days + 1
        except ValueError:
            lookback = None
    if lookback is None:
        lookback = mode_cfg.get("lookback_days")
    if lookback is None:
        lookback = 1 if mode == "daily" else 7 if mode == "weekly" else 30

    period_label = config.get("period_label") or _duration_label(lookback)
    period_title = config.get("period_title") or _title_case(period_label)
    period_range = config.get("period_range")
    if not period_range:
        range_days = max(0, int(js_number(lookback) or 0) - 1)
        start = date.fromisoformat(current_date) - timedelta(days=range_days)
        period_range = f"{start.isoformat()} to {current_date}"

    trend_raw = config.get("trend_enabled")
    if trend_raw is None:
        trend_raw = mode_cfg.get("trend_enabled", True)
    trend_enabled = _parse_bool(trend_raw, True)

    profile_id = config.get("profile_id") or topic.get("name") or "default"
    report_max = js_number(mode_cfg.get("report_max_selection") or (5 if mode == "daily" else 10))
    report_range_label = mode_cfg.get("report_selection_range_label") or (
        f"{mode_cfg.get('report_selection_min') or 0}-{mode_cfg.get('report_selection_max') or 0}")
    if report_range_label == "0-0":
        report_range_label = "3-5" if mode == "daily" else "7-10"

    return {
        "mode": mode,
        "isTestMode": _parse_bool(config.get("test_mode"), False),
        "trend_enabled": trend_enabled,
        "profileId": profile_id,
        "profileSlug": _slug(profile_id) or "default",
        "lookbackDays": lookback,
        "periodLabel": period_label,
        "periodTitle": period_title,
        "periodRange": period_range,
        "digestLabel": f"{topic.get('name') or 'Research'} Digest ({period_title})",
        "reportSelectionRangeLabel": report_range_label,
        "reportMaxSelection": report_max,
        "topicName": topic.get("name") or "Research",
        "topicFocus": topic.get("focus") or "research",
        "assistantSignature": topic.get("assistant_signature") or "Research Assistant",
        "currentDate": current_date,
    }


# --------------------------------------------------------------------------
# Adapters: flat Python paper dicts → the n8n item shapes the Code nodes read
# --------------------------------------------------------------------------

def _n8n_id(p: dict) -> str:
    """n8n's ``item.json.id`` is the full arXiv Atom id URL (with version)."""
    raw = p.get("raw_id")
    if raw:
        return str(raw)
    pid = str(p.get("id") or p.get("arxiv_id") or "")
    if not pid or pid.startswith("http"):
        return pid
    return f"http://arxiv.org/abs/{pid}"


def _author_field(p: dict):
    # "Prep Data for Pipeline" stores author as a *string*-typed Set field, so
    # the array lands JSON-stringified (the n8n caches show '"Name"' keys).
    if "author" in p:
        return p["author"]
    authors = p.get("authors")
    if isinstance(authors, list):
        return js_json(authors, indent=None)
    return UNDEF


def classified_item(p: dict) -> dict:
    """One "Merge Classification Results1" item (affiliation + classification)."""
    nid = _n8n_id(p)
    item = {
        "id": nid,
        "title": p.get("title", UNDEF),
        "summary": p.get("summary", UNDEF),
        "author": _author_field(p),
        "published": p.get("published", UNDEF),
        "category": p.get("category", p.get("categories", UNDEF)),
        "arxiv_id": nid,
        "affiliations": _or(p.get("affiliations"), []),
        "author_affiliations": _or(p.get("author_affiliations"), {}),
        "credibility_tier": _or(p.get("credibility_tier"), 0),
        "credibility_reasoning": _or(p.get("credibility_reasoning"), ""),
        "output": {
            "arxiv_id": nid,
            "primary_topic": _or(p.get("primary_topic"), ""),
            "secondary_topics": _or(p.get("secondary_topics"), []),
            "potential_impact": _or(p.get("potential_impact"), ""),
            "relevance_score": _or(p.get("relevance_score"), 0),
            "key_findings": _or(p.get("key_findings"), ""),
            "actionable": _or(p.get("actionable"), ""),
        },
    }
    return item


def selection_pool_item(p: dict) -> dict:
    """One "Format for Selection Agent" all_papers entry."""
    it = classified_item(p)
    out = it["output"]
    return {
        "arxiv_id": it["id"],
        "title": it["title"],
        "authors": it["author"],
        "affiliations": it["affiliations"],
        "credibility_tier": it["credibility_tier"],
        "credibility_reasoning": it["credibility_reasoning"],
        "primary_topic": out["primary_topic"],
        "secondary_topics": out["secondary_topics"],
        "potential_impact": out["potential_impact"],
        "relevance_score": out["relevance_score"],
        "key_findings": out["key_findings"],
        "abstract": _or(it["summary"], _get(out, "summary")),
    }


def analysis_succeeded(p: dict) -> bool:
    """Did deep analysis produce a usable result for this selected paper?

    In n8n a failed/unparseable Deep Analysis item never reaches the
    Aggregate (the merge on output.arxiv_id drops it) and Combine Report Data
    synthesizes a fallback for it. Our pipeline keeps failed papers in
    ``selected``; treat ``analysis_error``/``analysis_fallback`` or an empty
    large+short summary as "missing from the aggregate"."""
    if p.get("analysis_error") or p.get("analysis_fallback"):
        return False
    return bool(p.get("large_summary") or p.get("short_summary"))


def selection_payload(selected: list[dict], remaining: list[dict]) -> dict:
    """"Validate Selection Output"-shaped selected/remaining metadata."""
    sel = []
    for i, p in enumerate(selected):
        rank = js_number(p.get("priority_rank"))
        sel.append({
            "arxiv_id": _n8n_id(p),
            "selection_reasoning": str(p.get("selection_reasoning") or "").strip(),
            "priority_rank": rank if math.isfinite(rank) and p.get("priority_rank") is not None else i + 1,
        })
    rem = [{"arxiv_id": _n8n_id(p), "exclusion_reasoning": str(p.get("exclusion_reasoning") or "").strip()}
           for p in remaining]
    return {"selected_papers": [s for s in sel if s["arxiv_id"]],
            "remaining_papers": [r for r in rem if r["arxiv_id"]]}


def prepare_data_for_report(merged: dict) -> dict:
    """n8n "Prepare Data For Report" (full_text intentionally excluded)."""
    sel = merged.get("selected_papers") or {}
    out = merged.get("output") or {}
    return {
        "arxiv_id": _or(_get(sel, "arxiv_id"), _get(out, "arxiv_id"), merged.get("id", UNDEF)),
        "title": merged.get("title", UNDEF),
        "author": merged.get("author", UNDEF),
        "affiliations": merged.get("affiliations", UNDEF),
        "credibility_tier": merged.get("credibility_tier", UNDEF),
        "primary_topic": _get(out, "primary_topic"),
        "secondary_topics": _get(out, "secondary_topics"),
        "potential_impact": _get(out, "potential_impact"),
        "relevance_score": _get(out, "relevance_score"),
        "selection_reasoning": _get(sel, "selection_reasoning"),
        "large_summary": _or(_get(out, "large_summary"), _get(out, "short_summary")),
        "short_summary": _or(_get(out, "short_summary"), _get(out, "large_summary")),
        "pdf_analysis_performed": _get(out, "pdf_analysis_performed"),
    }


def _merged_analysis_item(p: dict, sel_meta: dict) -> dict:
    """Paper item after "Merge" + "Merge Analysis with Paper Data" (deep-merge:
    the Deep Analysis output fields land inside ``output``)."""
    item = classified_item(p)
    item["selected_papers"] = sel_meta
    item["output"] = {
        **item["output"],
        "arxiv_id": sel_meta.get("arxiv_id") or item["id"],
        "large_summary": p.get("large_summary", UNDEF),
        "short_summary": p.get("short_summary", UNDEF),
        "pdf_analysis_performed": p.get("pdf_analysis_performed", UNDEF),
    }
    return item


def normalize_media(media) -> dict:
    """"Media - Prepare Media Output For Parent" fields from whatever the media
    stage returned: None (media did not run → n8n's ``{}``), a list of
    articles, the parent-output dict, or the sub-workflow {summary, articles}."""
    if media is None:
        return {}
    if isinstance(media, list):
        item = {"articles": media, "summary": {}}
    elif isinstance(media, dict) and "media_articles" in media:
        return media
    elif isinstance(media, dict):
        item = media
    else:
        return {}
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
            dt = f" ({a['date']})" if a.get("date") else ""
            text = str(a.get("short_summary") or "").strip()
            lines.append(f"- {link} — {source}{dt}\n  {text}")
        media_md = "\n".join(lines)
    else:
        media_md = "- No media articles available for this period."
    return {
        "media_intelligence": {"period_range": period_range, "article_count": len(articles), "articles": articles},
        "media_period_range": period_range,
        "media_article_count": len(articles),
        "media_articles": articles,
        "media_markdown": media_md,
    }


# --------------------------------------------------------------------------
# Compute Paper Statistics
# --------------------------------------------------------------------------

def compute_paper_statistics(all_papers: list[dict], abstracts_analyzed: int) -> dict:
    """Port of "Compute Paper Statistics" over Merge Classification Results1
    items (ALL classified papers, before the relevance/credibility filter)."""
    rel = [v for v in (_to_num(_get(p, "output", "relevance_score")) for p in all_papers) if v is not None]
    cred = [v for v in (_to_num(p.get("credibility_tier", UNDEF)) for p in all_papers) if v is not None]
    avg_rel = sum(rel) / len(rel) if rel else 0
    avg_cred = sum(cred) / len(cred) if cred else 0
    high = sum(1 for p in all_papers if (_to_num(p.get("credibility_tier", UNDEF)) or 0) >= 8
               and _to_num(p.get("credibility_tier", UNDEF)) is not None)

    topic_counts: dict[str, int] = {}
    for p in all_papers:
        t = _key(_or(_get(p, "output", "primary_topic"), "Unknown"))
        topic_counts[t] = topic_counts.get(t, 0) + 1
    dist = _js_sort(_js_entries(topic_counts), lambda a, b: b[1] - a[1])[:10]

    impact = {"Breakthrough": 0, "High": 0, "Medium": 0, "Low": 0}
    for p in all_papers:
        k = _or(_get(p, "output", "potential_impact"), "Unknown")
        if isinstance(k, str) and k in impact:
            impact[k] += 1

    return {"stats": {
        "abstracts_analyzed": abstracts_analyzed,
        "total_relevant_papers": len(all_papers),
        "avg_relevance_score": _round1(avg_rel),
        "avg_credibility_tier": _round1(avg_cred),
        "high_credibility_count": high,
        "topic_distribution": [{"topic": t, "count": c} for t, c in dist],
        "impact_distribution": impact,
    }}


# --------------------------------------------------------------------------
# Combine Report Data
# --------------------------------------------------------------------------

_FALLBACK_REASON = ("Deep analysis returned an empty completion or parser failed before merge; "
                    "synthesized fallback from selected paper metadata.")
_REMAINING_POLICY = (
    "Priority order: analyzed-but-not-selected first, then non-analyzed remaining papers. "
    "Selected papers whose Deep Analysis failed are retained with analysis_error and "
    "pdf_analysis_performed=false. Within each source: P1 relevance_score==10 OR "
    "credibility_tier==10; P2 sort by relevance_score + credibility_tier; cap=20")


def _paper_id(p) -> str:
    return _normalize_id(_or(_get(p, "selected_papers", "arxiv_id"), _get(p, "output", "arxiv_id"),
                             _get(p, "arxiv_id"), _get(p, "id")))


def combine_report_data(*, aggregated: list[dict], stats: dict, split_payload: dict,
                        all_classified: list[dict], relevant_pool: list[dict],
                        media_raw: dict, run_mode: dict) -> dict:
    """Port of "Combine Report Data". Inputs are n8n-shaped:

    aggregated      Aggregate.data — Prepare Data For Report items
    stats           Compute Paper Statistics .stats
    split_payload   {"selected_papers": [...], "remaining_papers": [...]}
    all_classified  Merge Classification Results1 items
    relevant_pool   Format for Selection Agent .all_papers
    media_raw       Media - Prepare Media Output For Parent json ({} if absent)
    """
    aggregated = aggregated if isinstance(aggregated, list) else []
    selected_papers = split_payload.get("selected_papers")
    selected_papers = selected_papers if isinstance(selected_papers, list) else []
    remaining_papers = _or(split_payload.get("remaining_papers"), [])

    report_max = max(1, js_number(_or(run_mode.get("reportMaxSelection"), 10)))
    report_max = int(report_max) if math.isfinite(report_max) else 1

    media_articles = media_raw.get("media_articles") if isinstance(media_raw.get("media_articles"), list) else []
    media_period_range = js_str(_or(run_mode.get("periodRange"), media_raw.get("media_period_range"), ""))
    mac = media_raw.get("media_article_count")
    media_article_count = mac if isinstance(mac, (int, float)) and not isinstance(mac, bool) and math.isfinite(mac) \
        else len(media_articles)
    media_markdown = media_raw.get("media_markdown") if isinstance(media_raw.get("media_markdown"), str) else ""

    def rank_candidates(a, b):
        a1, b1 = (1 if a["p1"] else 0), (1 if b["p1"] else 0)
        if b1 != a1:
            return b1 - a1
        if b["p2"] != a["p2"]:
            return b["p2"] - a["p2"]
        if b["relevance"] != a["relevance"]:
            return b["relevance"] - a["relevance"]
        if b["credibility"] != a["credibility"]:
            return b["credibility"] - a["credibility"]
        return a["priorityRank"] - b["priorityRank"]

    selected_meta_by_id: dict[str, dict] = {}
    for p in selected_papers:
        if isinstance(p, dict) and _truthy(p.get("arxiv_id")):
            selected_meta_by_id[_normalize_id(p["arxiv_id"])] = p

    all_by_id: dict[str, dict] = {}
    for p in all_classified:
        pid = _normalize_id(_or(p.get("id", UNDEF), p.get("arxiv_id", UNDEF), _get(p, "output", "arxiv_id")))
        if pid:
            all_by_id[pid] = p

    selected_ids: dict[str, None] = {}
    for p in selected_papers:
        pid = _normalize_id(_get(p, "arxiv_id"))
        if pid:
            selected_ids.setdefault(pid, None)

    selected_rows = [all_by_id[i] for i in selected_ids if i in all_by_id]
    aggregated_ids = {pid for pid in (_paper_id(p) for p in aggregated) if pid}

    def build_fallback(pid: str) -> dict:
        meta = selected_meta_by_id.get(pid) or {}
        source = all_by_id.get(pid) or {}
        src_out = source.get("output") or {}
        arxiv_id = _or(meta.get("arxiv_id", UNDEF), src_out.get("arxiv_id", UNDEF),
                       source.get("arxiv_id", UNDEF), source.get("id", UNDEF), pid)
        title = _or(source.get("title", UNDEF), meta.get("title", UNDEF), "")
        kf = src_out.get("key_findings", UNDEF)
        key_findings = "; ".join(js_str(x) for x in kf) if isinstance(kf, list) else _or(kf, "")
        summary_context = " ".join(js_str(x) for x in (src_out.get("summary", UNDEF),
                                                       src_out.get("potential_impact", UNDEF),
                                                       key_findings) if _truthy(x))
        fallback_summary = " ".join(x for x in (
            "Selected for full-text analysis, but the Deep Analysis LLM returned no usable structured output.",
            "The report should still mention this paper because it passed relevance, credibility, and selection filters.",
            f"Classification context: {summary_context}" if summary_context else "",
        ) if x)
        return {
            **source,
            "selected_papers": {**meta, "analysis_fallback": True},
            "arxiv_id": arxiv_id,
            "id": _or(source.get("id", UNDEF), arxiv_id),
            "title": title,
            "author": _or(source.get("author", UNDEF), meta.get("authors", UNDEF), meta.get("author", UNDEF), []),
            "affiliations": _or(source.get("affiliations", UNDEF), meta.get("affiliations", UNDEF), []),
            "credibility_tier": _nullish(source.get("credibility_tier", UNDEF), meta.get("credibility_tier", UNDEF), None),
            "relevance_score": _nullish(src_out.get("relevance_score", UNDEF), source.get("relevance_score", UNDEF),
                                        meta.get("relevance_score", UNDEF)),
            "output": {
                **src_out,
                "arxiv_id": arxiv_id,
                "large_summary": fallback_summary,
                "short_summary": fallback_summary,
                "pdf_analysis_performed": False,
                "analysis_error": _FALLBACK_REASON,
                "analysis_fallback": True,
            },
            "pdf_analysis_performed": False,
            "analysis_error": _FALLBACK_REASON,
            "analysis_fallback": True,
            "report_recovery_source": "selected_missing_from_deep_analysis_aggregate",
        }

    fallback_papers = [p for p in (build_fallback(i) for i in selected_ids if i not in aggregated_ids)
                       if _normalize_id(_or(p.get("arxiv_id", UNDEF), p.get("id", UNDEF)))]
    agg_for_report = list(aggregated) + fallback_papers

    rows = []
    for idx, paper in enumerate(agg_for_report):
        pid = _normalize_id(_or(_get(paper, "selected_papers", "arxiv_id"), _get(paper, "output", "arxiv_id"),
                                _get(paper, "arxiv_id"), _get(paper, "id"), ""))
        relevance = js_null_number(_nullish(_get(paper, "relevance_score"),
                                            _get(paper, "output", "relevance_score"), 0))
        credibility = js_null_number(_nullish(_get(paper, "credibility_tier"), 0))
        meta = selected_meta_by_id.get(pid) or {}
        prio = js_null_number(_nullish(meta.get("priority_rank", UNDEF),
                                       _get(paper, "selected_papers", "priority_rank"), idx + 1))
        rows.append({
            "id": pid, "paper": paper, "relevance": relevance, "credibility": credibility,
            "p1": relevance == 10 or credibility == 10, "p2": relevance + credibility,
            "priorityRank": prio if math.isfinite(prio) else idx + 1,
        })
    ranked = _js_sort([r for r in rows if r["id"]], rank_candidates)

    report_selected = [{**r["paper"], "report_selection_rank": i + 1, "report_selected": True}
                       for i, r in enumerate(ranked[:report_max])]
    report_remaining = [{**r["paper"], "report_selected": False} for r in ranked[report_max:]]
    report_remaining_ids = {_paper_id(p) for p in report_remaining}

    def rel_of(p):
        return js_null_number(_nullish(_get(p, "output", "relevance_score"), _get(p, "relevance_score"), 0))

    analyzed_remaining_context = []
    for p in report_remaining:
        r = rel_of(p)
        c = js_null_number(_nullish(_get(p, "credibility_tier"), 0))
        analyzed_remaining_context.append({
            "arxiv_id": _or(_get(p, "selected_papers", "arxiv_id"), _get(p, "output", "arxiv_id"),
                            _get(p, "arxiv_id"), _get(p, "id"), ""),
            "title": _or(_get(p, "title"), ""),
            "affiliations": _or(_get(p, "affiliations"), []),
            # n8n quirk: aggregate items are flattened (no .output), so these
            # three are always empty for analyzed-not-selected entries.
            "primary_topic": _or(_get(p, "output", "primary_topic"), ""),
            "secondary_topics": _or(_get(p, "output", "secondary_topics"), []),
            "potential_impact": _or(_get(p, "output", "potential_impact"), ""),
            "relevance_score": r,
            "credibility_tier": c,
            "exclusion_reasoning": "Analyzed in full text but not included in final deep-dive cap.",
            "source": "analyzed_not_selected",
            "ranking_flags": {
                "p1_relevance_or_credibility_is_10": r == 10 or c == 10,
                "p2_relevance_plus_credibility": r + c,
            },
        })

    report_selected_ids: dict[str, None] = {}
    for p in report_selected:
        pid = _paper_id(p)
        if pid:
            report_selected_ids.setdefault(pid, None)
    report_selected_rows = [all_by_id[i] for i in report_selected_ids if i in all_by_id]
    remaining_ids_ordered: dict[str, None] = {}
    for p in report_remaining:
        remaining_ids_ordered.setdefault(_paper_id(p), None)
    analyzed_not_selected_rows = [all_by_id[i] for i in remaining_ids_ordered if i in all_by_id]

    def summarize_cohort(rows_):
        rel = [v for v in (_to_num(_get(r, "output", "relevance_score")) for r in rows_) if v is not None]
        cred = [v for v in (_to_num(r.get("credibility_tier", UNDEF)) for r in rows_) if v is not None]
        return {
            "count": len(rows_),
            "avg_relevance_score": _round1(sum(rel) / len(rel) if rel else 0),
            "avg_credibility_tier": _round1(sum(cred) / len(cred) if cred else 0),
            "high_credibility_count": sum(1 for r in rows_ if (_to_num(r.get("credibility_tier", UNDEF)) is not None
                                                               and _to_num(r.get("credibility_tier", UNDEF)) >= 8)),
        }

    def topic_distribution(rows_):
        counts: dict[str, int] = {}
        for r in rows_:
            t = _key(_or(_get(r, "output", "primary_topic"), "Unknown"))
            counts[t] = counts.get(t, 0) + 1
        return [{"topic": t, "count": c} for t, c in _js_sort(_js_entries(counts), lambda a, b: b[1] - a[1])]

    all_topic_dist = topic_distribution(all_classified)
    sel_topic_map = {t["topic"]: t["count"] for t in topic_distribution(report_selected_rows)}
    themes = [{"topic": t["topic"], "all_count": t["count"], "selected_count": sel_topic_map.get(t["topic"]) or 0}
              for t in all_topic_dist[:8]]

    remaining_map: dict[str, dict] = {}
    for p in (remaining_papers if isinstance(remaining_papers, list) else []):
        if isinstance(p, dict) and _truthy(p.get("arxiv_id")):
            remaining_map[js_str(p["arxiv_id"])] = p

    fallback_ctx = []
    for p in relevant_pool:
        key = js_str(p.get("arxiv_id", UNDEF))
        if key not in remaining_map or _normalize_id(p.get("arxiv_id", UNDEF)) in report_remaining_ids:
            continue
        r = js_null_number(_nullish(p.get("relevance_score", UNDEF), 0))
        c = js_null_number(_nullish(p.get("credibility_tier", UNDEF), 0))
        fallback_ctx.append({
            "arxiv_id": p.get("arxiv_id", UNDEF),
            "title": _or(p.get("title", UNDEF), ""),
            "affiliations": _or(p.get("affiliations", UNDEF), []),
            "primary_topic": _or(p.get("primary_topic", UNDEF), ""),
            "secondary_topics": _or(p.get("secondary_topics", UNDEF), []),
            "potential_impact": _or(p.get("potential_impact", UNDEF), ""),
            "relevance_score": r,
            "credibility_tier": c,
            "exclusion_reasoning": _or(remaining_map[key].get("exclusion_reasoning", UNDEF), ""),
            "source": "not_analyzed_remaining",
            "ranking_flags": {"p1_relevance_or_credibility_is_10": r == 10 or c == 10,
                              "p2_relevance_plus_credibility": r + c},
        })

    def fallback_cmp(a, b):
        a1 = 1 if a["ranking_flags"]["p1_relevance_or_credibility_is_10"] else 0
        b1 = 1 if b["ranking_flags"]["p1_relevance_or_credibility_is_10"] else 0
        if b1 != a1:
            return b1 - a1
        a2 = js_null_number(_or(a["ranking_flags"]["p2_relevance_plus_credibility"], 0))
        b2 = js_null_number(_or(b["ranking_flags"]["p2_relevance_plus_credibility"], 0))
        if b2 != a2:
            return b2 - a2
        if b["relevance_score"] != a["relevance_score"]:
            return b["relevance_score"] - a["relevance_score"]
        return b["credibility_tier"] - a["credibility_tier"]

    ranked_fallback = _js_sort(fallback_ctx, fallback_cmp)
    remaining_context = (analyzed_remaining_context + ranked_fallback)[:20]

    full_text_count = len(agg_for_report)
    report_selected_count = len(report_selected)
    not_selected_count = len(report_remaining)

    cohort = {
        "all_papers": summarize_cohort(all_classified),
        "full_text_selected_papers": summarize_cohort(selected_rows),
        "report_selected_papers": summarize_cohort(report_selected_rows),
        "analyzed_not_selected_papers": summarize_cohort(analyzed_not_selected_rows),
    }
    top_themes = "; ".join(f"{t['topic']} (all {js_str(t['all_count'])}, selected {js_str(t['selected_count'])})"
                           for t in themes[:5])
    impact_breakdown = ", ".join(f"{js_str(c)} {k}" for k, c in _js_entries(stats["impact_distribution"])
                                 if js_null_number(c) > 0)
    numbers = {
        "abstracts_analyzed": stats.get("abstracts_analyzed"),
        "full_text_analyzed_count": full_text_count,
        "report_selected_count": report_selected_count,
        "analyzed_not_selected_count": not_selected_count,
        "top_themes_all_vs_selected": top_themes,
        "all_avg_relevance": cohort["all_papers"]["avg_relevance_score"],
        "all_avg_credibility": cohort["all_papers"]["avg_credibility_tier"],
        "selected_avg_relevance": cohort["report_selected_papers"]["avg_relevance_score"],
        "selected_avg_credibility": cohort["report_selected_papers"]["avg_credibility_tier"],
        "impact_breakdown": impact_breakdown,
    }

    period_label = run_mode.get("periodLabel") or "this period"
    sources: dict[str, None] = {}
    for a in media_articles:
        if isinstance(a, dict) and _truthy(a.get("source")):
            sources.setdefault(js_str(a["source"]), None)
    media_sources = ", ".join(sources)
    media_stat_line = (f"\n- **{js_str(media_article_count)} media intelligence articles** tracked"
                       f"{f' from {media_sources}' if media_sources else ''} for {media_period_range}"
                       ) if media_article_count > 0 else ""
    n = numbers
    numbers_section = (
        f"- **{js_str(n['abstracts_analyzed'])} paper abstracts analyzed** {period_label}\n"
        f"- **{js_str(n['full_text_analyzed_count'])} papers underwent full-text analysis**\n"
        f"- **{js_str(n['report_selected_count'])} papers selected** for the final digest deep dive\n"
        f"- **{js_str(n['analyzed_not_selected_count'])} analyzed papers** are available for Also Worth Noting\n"
        f"- **Top research themes (all vs selected):** {n['top_themes_all_vs_selected']}\n"
        f"- **Average scores (all papers):** relevance {js_str(n['all_avg_relevance'])}/10, "
        f"credibility {js_str(n['all_avg_credibility'])}/10\n"
        f"- **Average scores (selected papers):** relevance {js_str(n['selected_avg_relevance'])}/10, "
        f"credibility {js_str(n['selected_avg_credibility'])}/10\n"
        f"- **Impact breakdown (all papers):** {n['impact_breakdown']}{media_stat_line}"
    )

    return {
        "stats": stats,
        "cohort_stats": cohort,
        "stats_dashboard": {
            "counts": {
                "abstracts_analyzed": stats.get("abstracts_analyzed"),
                "all_papers_considered": full_text_count,
                "selected_papers": report_selected_count,
                "full_text_analyzed_count": full_text_count,
                "report_selected_count": report_selected_count,
                "analyzed_not_selected_count": not_selected_count,
                "deep_analyses": full_text_count,
                "deep_analysis_fallback_count": len(fallback_papers),
            },
            "score_cards": {
                "all_papers": {
                    "avg_relevance_score": cohort["all_papers"]["avg_relevance_score"],
                    "avg_credibility_tier": cohort["all_papers"]["avg_credibility_tier"],
                },
                "selected_papers": {
                    "avg_relevance_score": cohort["report_selected_papers"]["avg_relevance_score"],
                    "avg_credibility_tier": cohort["report_selected_papers"]["avg_credibility_tier"],
                },
            },
            "themes": themes,
        },
        "analyzed_papers": report_selected,
        "full_text_analyzed_papers": agg_for_report,
        "deep_analysis_fallback_papers": fallback_papers,
        "report_selected_papers": report_selected,
        "report_remaining_analyzed_papers": analyzed_remaining_context,
        "remaining_relevant_papers": remaining_papers,
        "remaining_context_papers": remaining_context,
        "remaining_context_policy": _REMAINING_POLICY,
        "this_period_numbers": numbers_section,
        "media_intelligence": {
            "period_range": media_period_range,
            "article_count": media_article_count,
            "articles": media_articles,
            "markdown": media_markdown,
        },
        "summary": {
            "abstracts_analyzed": stats.get("abstracts_analyzed"),
            "total_analyzed": full_text_count,
            "total_report_selected": report_selected_count,
            "total_analyzed_not_selected": not_selected_count,
            "total_deep_analysis_fallbacks": len(fallback_papers),
            "total_remaining_shown": len(remaining_context),
            "total_relevant": stats.get("total_relevant_papers"),
            "total_all_considered": len(selected_rows),
            "total_selected": report_selected_count,
        },
    }


# --------------------------------------------------------------------------
# Build Report Prompt
# --------------------------------------------------------------------------

def build_report_prompt(combined: dict, config: dict, run_mode: dict) -> tuple[str, str]:
    """Port of "Build Report Prompt" → (report_prompt, report_system).

    Only the variables n8n supplies are substituted; every other placeholder
    becomes '' — including {{mode_label}} and {{topic_short_label}}, which the
    live profiles use in the subject line (hence the live subjects' double
    spaces: "MIRA  Digest - Top  Papers ..."). The system prompt is NOT
    templated, only its summary field is swapped."""
    period_label = run_mode.get("periodLabel") or "recent period"
    period_title = run_mode.get("periodTitle") or "Period"
    period_range = run_mode.get("periodRange") or ""
    topic = config.get("topic") or {}
    prompts = (config.get("prompts") or {}).get("report") or {}

    summary_mode = str(config.get("summary_mode") or "short").lower()
    summary_field = "large_summary" if summary_mode == "large" else "short_summary"

    def swap(v) -> str:
        return re.sub("large_summary", summary_field, re.sub("short_summary", summary_field, v or ""))

    template = swap(prompts.get("user") or "")
    system = swap(prompts.get("system") or "")

    mi = combined.get("media_intelligence") or {}
    articles = mi.get("articles") if isinstance(mi.get("articles"), list) else []
    media_json = js_json({
        "period_range": _or(mi.get("period_range", UNDEF), period_range, "the selected period"),
        "article_count": _or(mi.get("article_count", UNDEF), len(articles)),
        "articles": articles,
    })
    media_section = (
        "**## Media Intelligence**\n\nUse ONLY the media intelligence data below. Do NOT fabricate or infer "
        "articles.\n- Group articles by source when possible.\n- For each article include title (as markdown "
        "link using url), source, date (if present), and a concise 1-2 sentence summary.\n- If there are no "
        "articles, write: \"No media intelligence articles available for this period.\"\n\n"
        f"Media Intelligence Data:\n{media_json}")

    summary = combined.get("summary") or {}
    prompt = _apply_template(template, {
        "period_label": period_label,
        "period_title": period_title,
        "period_range": period_range,
        "topic_focus": topic.get("focus") or "research",
        "topic_name": topic.get("name") or "Research",
        "stats_json": js_json(combined.get("stats", UNDEF)) if combined.get("stats") is not None else UNDEF,
        "cohort_stats_json": js_json(_or(combined.get("cohort_stats", UNDEF), {})),
        "stats_dashboard_json": js_json(_or(combined.get("stats_dashboard", UNDEF), {})),
        "analyzed_papers_json": js_json(combined["analyzed_papers"]) if "analyzed_papers" in combined else UNDEF,
        "analyzed_not_selected_json": js_json(_or(combined.get("report_remaining_analyzed_papers", UNDEF), [])),
        "remaining_count": _nullish(summary.get("total_remaining_shown", UNDEF), 0),
        "remaining_papers_json": js_json(_or(combined.get("remaining_context_papers", UNDEF), [])),
        "remaining_selection_policy": _or(combined.get("remaining_context_policy", UNDEF), ""),
        "report_selection_range_label": _or(run_mode.get("reportSelectionRangeLabel"), "7-10"),
        "report_max_selection": _or(run_mode.get("reportMaxSelection"), 10),
        "this_period_numbers": combined.get("this_period_numbers", UNDEF),
        "digest_label": _or(run_mode.get("digestLabel"), ""),
        "current_date": run_mode.get("currentDate"),
        "assistant_signature": topic.get("assistant_signature") or "Research Assistant",
        "media_intelligence_json": media_json,
        "media_intelligence_section": media_section,
    })
    return f"{prompt}\n\n{media_section}", system


# --------------------------------------------------------------------------
# Generate Report → Parse Report Output → Validate Report Output
# --------------------------------------------------------------------------

_REPORT_SCHEMA = (
    '{\n  "type": "object",\n  "properties": {\n'
    '    "subject": {"type": "string", "description": "The email subject line for the research digest"},\n'
    '    "body": {"type": "string", "description": "The full email body in Markdown format"}\n'
    '  },\n  "required": ["subject", "body"]\n}')
_TREND_SCHEMA = ('{\n  "type": "object",\n  "properties": {\n    "trend_section_markdown": {"type": "string"}\n'
                 '  },\n  "required": ["trend_section_markdown"]\n}')


def _format_instructions(schema: str) -> str:
    # Approximates LangChain's StructuredOutputParser.getFormatInstructions().
    return ("You must format your output as a JSON value that adheres to a given \"JSON Schema\" instance.\n\n"
            "Your output will be parsed and type-checked according to the provided schema instance, so make "
            "sure all fields in your output match the schema exactly and there are no trailing commas!\n\n"
            "Here is the JSON Schema instance your output must adhere to. Include the enclosing markdown "
            f"codeblock:\n```json\n{schema}\n```\n")


def _extract_json_object(text) -> dict | None:
    """JSON object from an LLM completion: mira.config.parse_json_response,
    then a lenient pass (fences anywhere, raw control characters in strings).
    Unwraps n8n's ``{"output": {...}}`` parser wrapper."""
    from mira.config import parse_json_response

    if isinstance(text, dict):
        obj = text
    else:
        text = str(text or "")
        obj = None
        try:
            obj = parse_json_response(text)
        except Exception:  # noqa: BLE001
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.S)
            candidates = [m.group(1)] if m else []
            i, j = text.find("{"), text.rfind("}")
            if i != -1 and j > i:
                candidates.append(text[i:j + 1])
            for cand in candidates:
                try:
                    obj = json.loads(cand, strict=False)
                    break
                except Exception:  # noqa: BLE001
                    continue
    if isinstance(obj, dict) and set(obj) == {"output"} and isinstance(obj["output"], dict):
        obj = obj["output"]
    return obj if isinstance(obj, dict) else None


def _report_fields(obj: dict | None) -> dict | None:
    if obj and isinstance(obj.get("subject"), str) and isinstance(obj.get("body"), str):
        return {"subject": obj["subject"], "body": obj["body"]}
    return None


def generate_report_text(report_prompt: str, report_system: str, config: dict, client) -> str:
    """n8n "Generate Report" agent (model llm_models.report; the node retries
    3x — mira.config.llm_call retries 3x as well)."""
    from mira import config as cfg

    return cfg.llm_call(client, cfg.model_for(config, "report"), report_system, report_prompt)


def parse_report_output(raw: str, config: dict, client) -> dict:
    """"Parse Report Output": {subject, body} from the agent's text.

    n8n always pipes the agent text through an LLM chain (claude-sonnet-5 +
    Structured Output Parser). We parse deterministically first and only call
    that chain when the text is not a usable JSON object (e.g. raw Markdown).
    Raises ReportParseError if the chain's answer does not parse either."""
    from mira import config as cfg

    fields = _report_fields(_extract_json_object(raw))
    if fields:
        return fields
    try:
        cfg.record_parse_failure("report")
    except Exception:  # noqa: BLE001
        pass
    model = (config.get("llm_models") or {}).get("report_parse") or PARSE_REPORT_MODEL_DEFAULT
    prompt = (
        "Extract the subject and body from the following report output. The text should contain a JSON "
        "object with \"subject\" and \"body\" keys. If the text contains JSON (with or without markdown code "
        "fences), extract it. If the text is a raw Markdown report without JSON wrapping, use the first line "
        "as the subject and the rest as the body.\n\nReport output:\n"
        f"{raw}\n\n{_format_instructions(_REPORT_SCHEMA)}")
    answer = cfg.llm_call(client, model, "", prompt)
    fields = _report_fields(_extract_json_object(answer))
    if not fields:
        raise ReportParseError(f"Parse Report Output could not extract subject/body. Preview: {str(answer)[:300]!r}")
    return fields


def validate_report_output(output: dict) -> None:
    """Port of "Validate Report Output" — raises ReportValidationError."""
    subject = js_str(output.get("subject") or "").strip()
    body = js_str(output.get("body") or "").strip()
    errors = []
    if not subject or len(subject) < 10:
        errors.append(f"Subject too short or empty (got {len(subject)} chars)")
    if "example subject" in subject.lower():
        errors.append("Subject contains placeholder text")
    if not body or len(body) < 200:
        errors.append(f"Body too short or empty (got {len(body)} chars)")
    if "example body" in body.lower() or "this is an example" in body.lower():
        errors.append("Body contains placeholder text")
    if "##" not in body:
        errors.append("Body missing expected Markdown headings (##)")
    if errors:
        raise ReportValidationError(f"Report validation failed: {'; '.join(errors)}")


# --------------------------------------------------------------------------
# Persist Current Report / Write Report File
# --------------------------------------------------------------------------

def report_dirs(run_mode: dict) -> tuple[Path, Path]:
    """(report_dir, prior_reports_dir): tests/<slug> in test mode, prod/<slug>
    otherwise; prior reports are always read from prod/<slug>."""
    slug = run_mode["profileSlug"]
    prod = REPORT_FILES / "prod" / slug
    return (REPORT_FILES / "tests" / slug if run_mode.get("isTestMode") else prod), prod


def persist_current_report(output: dict, config: dict, run_mode: dict,
                           now: datetime | None = None) -> dict:
    """Port of "Persist Current Report" + "Ensure Report Directories" +
    "Select Report Record for File" + "Convert Report Record to File" +
    "Write Report File". Writes ``[record]`` (2-space JSON, no trailing
    newline) and returns {"report_record", "report_dir", "prior_reports_dir",
    "report_file" (Path), "prior_report_window_days"}."""
    topic = config.get("topic") or {}
    topic_name = run_mode.get("topicName") or topic.get("name") or "Research"
    safe_topic = _slug(topic_name)
    current_date = run_mode.get("currentDate") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    iso = js_iso_now(now)
    timestamp = re.sub(r"[:.]", "-", iso)
    report_dir, prior_dir = report_dirs(run_mode)
    report_id = f"report-{safe_topic}-{current_date}-{timestamp}"
    file_path = report_dir / f"{report_id}.json"
    is_test = run_mode.get("isTestMode") is True

    record = {
        "report_id": report_id,
        "profile_id": config.get("profile_id") or run_mode.get("profileId") or "",
        "topic_name": topic_name,
        "topic_focus": run_mode.get("topicFocus") or topic.get("focus") or "",
        "period_label": run_mode.get("periodLabel") or "",
        "period_title": run_mode.get("periodTitle") or "",
        "period_range": run_mode.get("periodRange") or "",
        "run_date": current_date,
        "created_at": iso,
        "subject": output.get("subject") or "",
        "body_markdown": output.get("body") or "",
        "is_test": is_test,
        # n8n stored its container path (/report-files/...); we store where
        # the file actually is. Nothing downstream reads this field.
        "report_file": str(file_path),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    prior_dir.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps([record], indent=2, ensure_ascii=False), encoding="utf-8")
    return {"report_record": record, "report_dir": report_dir, "prior_reports_dir": prior_dir,
            "report_file": file_path, "prior_report_window_days": PRIOR_REPORT_WINDOW_DAYS}


# --------------------------------------------------------------------------
# Fetch Prior Reports / Load Prior Reports
# --------------------------------------------------------------------------

_TEST_WORDS_RE = re.compile(r"\b(test|debug|demo|draft|tmp|sandbox|trial)\b", re.ASCII)


def _parse_js_date(value) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    normalized = f"{value}T00:00:00Z" if len(value) == 10 else value
    try:
        dt = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def is_likely_test_report(report: dict) -> bool:
    """n8n heuristic: ANY of test|debug|demo|draft|tmp|sandbox|trial as a whole
    word in subject/topic/body marks the report as a test (so a real digest
    that says e.g. "stress test" is also excluded from trend input)."""
    blob = " ".join(js_str(x) for x in (report.get("subject"), report.get("topic_name"),
                                        report.get("body_markdown")) if _truthy(x)).lower()
    return bool(_TEST_WORDS_RE.search(blob))


def load_prior_reports(prior_dir: Path, record: dict, window_days: int = PRIOR_REPORT_WINDOW_DAYS,
                       run_date_fallback: str | None = None) -> list[dict]:
    """Port of "Fetch Prior Reports" (glob ``report-*.json``) + "Load Prior
    Reports": same profile/topic, not test, not the current report, run_date
    (or created_at) within [run_date - window_days, run_date]; newest first."""
    topic_name = record.get("topic_name") or ""
    profile_id = record.get("profile_id") or ""
    current_id = record.get("report_id") or ""
    current_dt = _parse_js_date(record.get("run_date") or run_date_fallback or "") or datetime.now(timezone.utc)
    window = js_number(window_days or 60)
    cutoff = current_dt - timedelta(days=window if math.isfinite(window) else 60)

    reports = []
    for path in sorted(Path(prior_dir).glob("report-*.json")) if Path(prior_dir).is_dir() else []:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for rep in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(rep, dict):
                continue
            if current_id and rep.get("report_id") == current_id:
                continue
            if topic_name and rep.get("topic_name") and rep.get("topic_name") != topic_name:
                continue
            if profile_id and rep.get("profile_id") != profile_id:
                continue
            if rep.get("is_test") is True or is_likely_test_report(rep):
                continue
            dt = _parse_js_date(rep.get("run_date") or rep.get("created_at") or "")
            if not dt or dt < cutoff or dt > current_dt:
                continue
            reports.append(rep)

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return sorted(reports, key=lambda r: _parse_js_date(r.get("run_date") or r.get("created_at") or "") or epoch,
                  reverse=True) if reports else reports


# --------------------------------------------------------------------------
# Build Trend Prompt → Trend Analysis Agent → Append Trend Section
# --------------------------------------------------------------------------

DEFAULT_TREND_TEMPLATE = """You are analyzing multi-week trends across research digests.

Topic: {{topic_name}} ({{topic_focus}})
Window: last {{window_days}} days
Prior report count: {{prior_report_count}}

Current report (JSON):
{{current_report_json}}

Prior reports (JSON array):
{{prior_reports_json}}

TASK:
Write a concise trend section (150-250 words) summarizing:
- The most recurring topics and why they are rising
- The most notable papers or institutions across the window (mention titles if available)
- How the trend evolved from earlier reports to the current report (if the range is small, like 90 days, there may be no noticable trend of change, but you should comment on what you see across weeks that stand out)

IMPORTANT: NEVER use LaTeX, MathJax, or dollar-sign math notation. Write all equations and formulas in plain text.

Output ONLY valid JSON (no code fences):
{
  "trend_section_markdown": "..."
}"""


def build_trend_prompt(output: dict, prior_reports: list[dict], config: dict, run_mode: dict,
                       window_days: int = PRIOR_REPORT_WINDOW_DAYS) -> tuple[str, str]:
    """Port of "Build Trend Prompt" → (trend_prompt, trend_system)."""
    topic = config.get("topic") or {}
    trend_cfg = (config.get("prompts") or {}).get("trend") or {}
    template = trend_cfg.get("user") or DEFAULT_TREND_TEMPLATE
    system_template = trend_cfg.get("system") or ""
    current = {"run_date": run_mode.get("currentDate") or "", "subject": output.get("subject") or "",
               "body_markdown": output.get("body") or ""}
    prior = [{"run_date": r.get("run_date") or r.get("created_at") or "", "subject": r.get("subject") or "",
              "body_markdown": r.get("body_markdown") or ""} for r in prior_reports]
    topic_name = run_mode.get("topicName") or topic.get("name") or "Research"
    topic_focus = run_mode.get("topicFocus") or topic.get("focus") or "research"
    prompt = _apply_template(template, {
        "topic_name": topic_name,
        "topic_focus": topic_focus,
        "window_days": window_days or 60,
        "prior_report_count": len(prior_reports) or 0,
        "current_report_json": js_json(current),
        "prior_reports_json": js_json(prior),
    })
    system = _apply_template(system_template, {"topic_name": topic_name, "topic_focus": topic_focus})
    return prompt, system


def run_trend_analysis(trend_prompt: str, trend_system: str, config: dict, client) -> str:
    """"Trend Analysis Agent" (llm_models.trend) with the Structured Output
    Parser (Trend) autoFix pass (claude-sonnet-5). Returns the stripped
    trend_section_markdown; raises ValueError when both passes fail."""
    from mira import config as cfg

    raw = cfg.llm_call(client, cfg.model_for(config, "trend"), trend_system, trend_prompt)
    obj = _extract_json_object(raw)
    if obj and isinstance(obj.get("trend_section_markdown"), str):
        return obj["trend_section_markdown"].strip()
    try:
        cfg.record_parse_failure("trend")
    except Exception:  # noqa: BLE001
        pass
    fix_model = (config.get("llm_models") or {}).get("trend_fix") or TREND_FIX_MODEL_DEFAULT
    fix_prompt = (
        f"Instructions:\n--------------\n{_format_instructions(_TREND_SCHEMA)}--------------\n"
        f"Completion:\n--------------\n{raw}\n--------------\n\n"
        "Above, the Completion did not satisfy the constraints given in the Instructions.\n"
        "Error:\n--------------\nCould not parse a JSON object with a string \"trend_section_markdown\".\n"
        "--------------\n\nPlease try again. Please only respond with an answer that satisfies the "
        "constraints laid out in the Instructions:")
    fixed = _extract_json_object(cfg.llm_call(client, fix_model, "", fix_prompt))
    if fixed and isinstance(fixed.get("trend_section_markdown"), str):
        return fixed["trend_section_markdown"].strip()
    raise ValueError("Trend Analysis Agent output did not contain trend_section_markdown")


def append_trend_section(body: str, trend_section: str, window_days: int = PRIOR_REPORT_WINDOW_DAYS) -> str:
    """Port of "Append Trend Section"."""
    trend_section = (trend_section or "").strip()
    if not trend_section:
        return body or ""
    heading = f"## Multi-Week Trend Watch (Last {js_str(window_days or 60)} Days)"
    return f"{body or ''}\n\n{heading}\n\n{trend_section}"


# --------------------------------------------------------------------------
# Post Process → Markdown → Styling → HTML file → PDF
# --------------------------------------------------------------------------

def post_process_email_data(output: dict) -> tuple[str, str]:
    """"Post Process Email Data2": subject, and the body with every literal
    backslash-n sequence turned into a real newline."""
    subject = output.get("subject")
    subject = "" if subject is None else js_str(subject)
    return subject, str(output.get("body") or "").replace("\\n", "\n")


def render_email_html(body_markdown: str, subject: str, config: dict, run_mode: dict | None = None,
                      stats_dashboard: dict | None = None) -> str:
    """"Convert Markdown to HTML2" + "Apply Email Styling" → html_body."""
    run_mode = run_mode or resolve_run_mode(config)
    raw_digest = run_mode.get("digestLabel") or ""
    header_title = run_mode.get("topicName") or "Memory Technology Research"
    digest_label = raw_digest if raw_digest and "${" not in raw_digest else \
        f"{header_title} Digest ({run_mode.get('periodTitle') or 'Period'})"
    email_cfg = config.get("email_cfg") or {}
    return style_email(
        markdown_to_html(body_markdown), subject,
        header_title=header_title,
        digest_label=digest_label,
        current_date=run_mode.get("currentDate"),
        colors=resolve_palette(config),
        footer_blurb=email_cfg.get("footer_blurb"),
        stats_dashboard=stats_dashboard,
    )


def html_output_paths(now: datetime | None = None) -> tuple[Path, Path]:
    """"Prepare & Convert HTML": report-files/report-memory-<UTC ts>.{html,pdf}
    (n8n hard-codes "memory" for every profile and writes to the root)."""
    ts = re.sub(r"[:.]", "-", js_iso_now(now))[:19]
    return REPORT_FILES / f"report-memory-{ts}.html", REPORT_FILES / f"report-memory-{ts}.pdf"


def pdf_enabled() -> bool:
    return os.environ.get("MIRA_PDF", "1").strip().lower() not in ("0", "false", "no", "off")


def generate_pdf(html_path: Path, pdf_path: Path, timeout: int = 180) -> tuple[Path | None, str | None]:
    """"Generate PDF": ``npx tsx generate-pdf.ts <html> <pdf>`` in crawlers/.

    Failure-tolerant (n8n would abort the run): returns (pdf_path, None) on
    success, (None, reason) on any failure — missing npx/tsx/Chrome, non-zero
    exit, timeout, or no/empty output file."""
    pdf_path = Path(pdf_path)
    try:
        pdf_path.unlink(missing_ok=True)
        proc = subprocess.run(
            ["npx", "tsx", "generate-pdf.ts", str(Path(html_path).resolve()), str(pdf_path.resolve())],
            cwd=str(CRAWLERS_DIR), capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return None, f"generate-pdf.ts exited {proc.returncode}: {(proc.stderr or proc.stdout)[-400:].strip()}"
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        return None, "generate-pdf.ts produced no PDF"
    return pdf_path, None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _flat_list(v) -> list[dict]:
    return [p for p in (v or []) if isinstance(p, dict)]


def _canonicalize_selection(split: dict, papers: list[dict]) -> dict:
    """Map selection arxiv_ids given in another form (bare "2609.00001",
    versioned "2609.00001v1", abs/pdf URL) onto the paper's n8n id, so the
    id-keyed joins in Combine Report Data line up. Unknown ids are kept."""
    lookup: dict[str, str] = {}
    for p in papers:
        nid = _n8n_id(p)
        norm = _normalize_id(nid)
        for key in (norm, re.sub(r"v\d+$", "", norm), str(p.get("id") or "")):
            if key:
                lookup.setdefault(key, nid)

    def fix(entries):
        out = []
        for e in entries or []:
            if not isinstance(e, dict):
                continue
            norm = _normalize_id(e.get("arxiv_id"))
            nid = lookup.get(norm) or lookup.get(re.sub(r"v\d+$", "", norm))
            out.append({**e, "arxiv_id": nid} if nid else dict(e))
        return out

    return {**split, "selected_papers": fix(split.get("selected_papers")),
            "remaining_papers": fix(split.get("remaining_papers"))}


def build_report_inputs(pipeline_result: dict, media, config: dict, run_mode: dict | None = None) -> dict:
    """Everything up to and including Combine Report Data (no LLM, no I/O).

    pipeline_result keys (flat paper dicts as produced by mira.pipeline):
      selected        analyzed selected papers (required)
      remaining       relevant papers not selected (required)
      total_scanned   abstracts analyzed ("Remove Duplicates" count); or pass
                      abstracts_analyzed explicitly
      all_classified  ALL classified papers incl. below-threshold (n8n
                      "Merge Classification Results1"). Optional; when absent
                      the "all papers" statistics fall back to selected+remaining.
      relevant_pool   filtered+ranked pool shown to the selection agent (n8n
                      "Format for Selection Agent"); default remaining+selected.
      selection       {"selected_papers": [{arxiv_id, selection_reasoning,
                      priority_rank}], "remaining_papers": [{arxiv_id,
                      exclusion_reasoning}]}; default derived from the flat lists.
    """
    run_mode = run_mode or resolve_run_mode(config)
    selected = _flat_list(pipeline_result.get("selected"))
    remaining = _flat_list(pipeline_result.get("remaining"))
    all_flat = pipeline_result.get("all_classified") or pipeline_result.get("classified")
    if not all_flat:
        seen: dict[str, dict] = {}
        for p in selected + remaining:
            seen.setdefault(_normalize_id(_n8n_id(p)), p)
        all_flat = list(seen.values())
    all_items = [classified_item(p) for p in _flat_list(all_flat)]

    split = pipeline_result.get("selection")
    if split:
        split = _canonicalize_selection(split, _flat_list(all_flat) + selected + remaining)
    else:
        split = selection_payload(selected, remaining)
    meta_by_id = {_normalize_id(m.get("arxiv_id")): m for m in split.get("selected_papers") or []}
    aggregated = []
    for p in selected:
        if not analysis_succeeded(p):
            continue
        nid = _normalize_id(_n8n_id(p))
        meta = meta_by_id.get(nid)
        if meta is None:
            continue  # n8n's merge-by-arxiv_id drops papers the selection did not list
        aggregated.append(prepare_data_for_report(_merged_analysis_item(p, meta)))

    pool_flat = pipeline_result.get("relevant_pool") or (remaining + selected)
    pool = [selection_pool_item(p) for p in _flat_list(pool_flat)]
    abstracts = pipeline_result.get("abstracts_analyzed", pipeline_result.get("total_scanned", 0))

    media_raw = normalize_media(media)
    stats = compute_paper_statistics(all_items, abstracts)["stats"]
    combined = combine_report_data(
        aggregated=aggregated, stats=stats, split_payload=split, all_classified=all_items,
        relevant_pool=pool, media_raw=media_raw, run_mode=run_mode)
    return {"run_mode": run_mode, "combined": combined,
            "n8n_inputs": {"aggregated": aggregated, "split_payload": split, "all_classified": all_items,
                           "relevant_pool": pool, "media_raw": media_raw, "abstracts_analyzed": abstracts}}


def produce_report(pipeline_result: dict, media, config: dict, client, *,
                   now: datetime | None = None, pdf: bool = True) -> dict:
    """Run the n8n back half end to end (LLM calls, record, trend, HTML, PDF).

    Returns {"subject", "body_markdown", "html", "html_path", "pdf_path",
    "record_path", "report_record", "trend_applied", "trend_error",
    "prior_report_count", "pdf_error", "stats_dashboard", "run_mode"}.
    Raises ReportParseError / ReportValidationError where n8n would stop.
    """
    prepared = build_report_inputs(pipeline_result, media, config)
    run_mode, combined = prepared["run_mode"], prepared["combined"]

    report_prompt, report_system = build_report_prompt(combined, config, run_mode)
    raw = generate_report_text(report_prompt, report_system, config, client)
    output = parse_report_output(raw, config, client)
    validate_report_output(output)

    persisted = persist_current_report(output, config, run_mode, now=now)
    record = persisted["report_record"]
    window = persisted["prior_report_window_days"]
    prior = load_prior_reports(persisted["prior_reports_dir"], record, window, run_mode.get("currentDate"))

    trend_applied, trend_error = False, None
    if run_mode.get("trend_enabled") is True:
        trend_prompt, trend_system = build_trend_prompt(output, prior, config, run_mode, window)
        try:
            section = run_trend_analysis(trend_prompt, trend_system, config, client)
        except Exception as exc:  # noqa: BLE001 — n8n would fail the run here; we keep the digest
            trend_error = f"{type(exc).__name__}: {exc}"
            print(f"  WARNING: trend analysis failed, sending digest without it — {trend_error}")
            section = ""
        if section:
            output = {**output, "body": append_trend_section(output.get("body") or "", section, window)}
            trend_applied = True

    subject, body_markdown = post_process_email_data(output)
    html = render_email_html(body_markdown, subject, config, run_mode, combined.get("stats_dashboard"))

    html_path, pdf_path = html_output_paths(now)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")

    pdf_out, pdf_error = None, None
    if pdf and pdf_enabled():
        pdf_out, pdf_error = generate_pdf(html_path, pdf_path)
        if pdf_error:
            print(f"  WARNING: PDF generation failed, continuing without attachment — {pdf_error}")
    else:
        pdf_error = "disabled"

    return {
        "subject": subject,
        "body_markdown": body_markdown,
        "html": html,
        "html_path": html_path,
        "pdf_path": pdf_out,
        "record_path": persisted["report_file"],
        "report_record": record,
        "trend_applied": trend_applied,
        "trend_error": trend_error,
        "prior_report_count": len(prior),
        "pdf_error": pdf_error,
        "stats_dashboard": combined.get("stats_dashboard"),
        "run_mode": run_mode,
    }


# --------------------------------------------------------------------------
# Delivery (Build Email with PDF → Send Email2), generic SMTP
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SmtpSettings:
    host: str
    port: int
    security: str  # "starttls" | "ssl" | "none"
    user: str | None
    password: str | None
    sender: str


def smtp_settings(environ: dict | None = None) -> SmtpSettings:
    """SMTP settings from the environment.

    MIRA_SMTP_HOST, MIRA_SMTP_PORT (default 587), MIRA_SMTP_USER,
    MIRA_SMTP_PASSWORD, MIRA_SMTP_FROM (default MIRA_SMTP_USER),
    MIRA_SMTP_SECURITY (starttls|ssl|none; default ssl on port 465, else
    starttls). Legacy fallback when MIRA_SMTP_HOST is unset: GMAIL_USER +
    GMAIL_APP_PASSWORD via smtp.gmail.com:587 STARTTLS."""
    env = os.environ if environ is None else environ
    host = (env.get("MIRA_SMTP_HOST") or "").strip()
    if host:
        port_raw = (env.get("MIRA_SMTP_PORT") or "").strip() or "587"
        try:
            port = int(port_raw)
        except ValueError:
            raise EnvironmentError(f"MIRA_SMTP_PORT must be an integer, got {port_raw!r}") from None
        security = (env.get("MIRA_SMTP_SECURITY") or "").strip().lower() or ("ssl" if port == 465 else "starttls")
        if security not in ("starttls", "ssl", "none"):
            raise EnvironmentError(f"MIRA_SMTP_SECURITY must be starttls, ssl or none, got {security!r}")
        user = (env.get("MIRA_SMTP_USER") or "").strip() or None
        password = env.get("MIRA_SMTP_PASSWORD") or None
        sender = (env.get("MIRA_SMTP_FROM") or "").strip() or user
        if not sender:
            raise EnvironmentError("Set MIRA_SMTP_FROM (or MIRA_SMTP_USER) for the sender address")
        return SmtpSettings(host, port, security, user, password, sender)
    gmail_user = (env.get("GMAIL_USER") or "").strip()
    gmail_password = env.get("GMAIL_APP_PASSWORD") or ""
    if gmail_user and gmail_password:
        return SmtpSettings("smtp.gmail.com", 587, "starttls", gmail_user, gmail_password, gmail_user)
    raise EnvironmentError(
        "No SMTP configuration: set MIRA_SMTP_HOST (+ MIRA_SMTP_PORT/USER/PASSWORD/FROM), "
        "or the legacy GMAIL_USER and GMAIL_APP_PASSWORD")


def _split_addresses(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        items = [str(x) for x in value]
    else:
        items = str(value or "").split(",")
    return [a.strip() for a in items if a and a.strip()]


def resolve_recipients(recipients=None, config: dict | None = None, environ: dict | None = None) -> list[str]:
    """recipients argument → MIRA_RECIPIENTS (comma-separated) →
    RECIPIENT_EMAIL → config["recipient_email"]."""
    env = os.environ if environ is None else environ
    for source in (recipients, env.get("MIRA_RECIPIENTS"), env.get("RECIPIENT_EMAIL"),
                   (config or {}).get("recipient_email")):
        found = _split_addresses(source)
        if found:
            return found
    return []


def build_message(html: str, subject: str, sender: str, recipients: list[str],
                  attachments: list | None = None) -> MIMEMultipart:
    """HTML email; with attachments a multipart/mixed wrapper. Attachment items
    are paths or (path, filename) tuples."""
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html, "html", "utf-8"))
    if attachments:
        msg = MIMEMultipart("mixed")
        msg.attach(alt)
        for item in attachments:
            path, name = (item if isinstance(item, tuple) else (item, Path(item).name))
            part = MIMEApplication(Path(path).read_bytes(),
                                   _subtype="pdf" if str(name).lower().endswith(".pdf") else "octet-stream")
            part.add_header("Content-Disposition", "attachment", filename=str(name))
            msg.attach(part)
    else:
        msg = alt
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    return msg


def send_email(html: str, subject: str, config: dict, recipients=None, attachments: list | None = None,
               dry_run: bool = False, environ: dict | None = None) -> dict:
    """Send one HTML email over SMTP (see smtp_settings).

    Recipients: the ``recipients`` argument, else ``config["recipient_email"]``
    (mira.realtime passes one subscriber per call this way), else
    MIRA_RECIPIENTS / RECIPIENT_EMAIL. ``dry_run`` builds the message but
    does not connect. Returns {"sent", "recipients", "attachments", "host"}."""
    to = _split_addresses(recipients) or _split_addresses((config or {}).get("recipient_email")) \
        or resolve_recipients(None, None, environ)
    if not to:
        raise ValueError("No email recipients: pass recipients, or set MIRA_RECIPIENTS / RECIPIENT_EMAIL")
    names = [str(a[1] if isinstance(a, tuple) else Path(a).name) for a in (attachments or [])]
    if dry_run:
        return {"sent": False, "dry_run": True, "recipients": to, "attachments": names, "host": None}
    s = smtp_settings(environ)
    msg = build_message(html, subject, s.sender, to, attachments)
    context = ssl.create_default_context()
    if s.security == "ssl":
        server = smtplib.SMTP_SSL(s.host, s.port, context=context, timeout=60)
    else:
        server = smtplib.SMTP(s.host, s.port, timeout=60)
    with server:
        server.ehlo()
        if s.security == "starttls":
            server.starttls(context=context)
            server.ehlo()
        if s.user and s.password:
            server.login(s.user, s.password)
        server.sendmail(s.sender, to, msg.as_string())
    return {"sent": True, "dry_run": False, "recipients": to, "attachments": names, "host": s.host}


def deliver_report(result: dict, config: dict, recipients=None, *, dry_run: bool = False,
                   attach_pdf: bool = True, environ: dict | None = None) -> dict:
    """"Build Email with PDF" + "Send Email2" for a produce_report() result.

    Recipients: argument → MIRA_RECIPIENTS → RECIPIENT_EMAIL →
    config["recipient_email"]. Attaches the PDF as research-report.pdf when
    one was generated (n8n built the attachment but never passed it to
    Send Email2)."""
    to = resolve_recipients(recipients, config, environ)
    if not to:
        raise ValueError("No email recipients: pass recipients, or set MIRA_RECIPIENTS / RECIPIENT_EMAIL")
    pdf_path = result.get("pdf_path")
    attachments = [(Path(pdf_path), PDF_ATTACHMENT_NAME)] if attach_pdf and pdf_path and Path(pdf_path).is_file() \
        else []
    return send_email(result["html"], result["subject"], config, recipients=to, attachments=attachments,
                      dry_run=dry_run, environ=environ)


# --------------------------------------------------------------------------
# Legacy CLI path (run.py) — kept unchanged until run.py moves to
# produce_report/deliver_report. to_html is also used by mira.realtime.
# --------------------------------------------------------------------------


def match_themes(paper: dict, main_themes: list[str]) -> list[str]:
    """Ordered, de-duplicated canonical themes a paper touches.

    Matches the paper's ``primary_topic`` (first) and ``secondary_topics`` (in
    order) against ``main_themes`` using ``normalize_topic(...).casefold()`` on
    both sides, so qualified variants like
    ``"Photonic Integrated Circuits (Silicon Photonics)"`` collapse onto their
    canonical base name. Returns the canonical display names (from
    ``main_themes``, not the paper's raw strings); free-form topics that match
    no canonical theme are ignored.
    """
    lookup = {normalize_topic(t).casefold(): t for t in main_themes}
    candidates = [paper.get("primary_topic", "")]
    candidates.extend(paper.get("secondary_topics", []) or [])

    matched: list[str] = []
    for candidate in candidates:
        if not candidate or not str(candidate).strip():
            continue
        key = normalize_topic(str(candidate)).casefold()
        display = lookup.get(key)
        if display is not None and display not in matched:
            matched.append(display)
    return matched


def format_theme_label(themes: list[str]) -> str:
    """Format the parenthetical theme tag; '' when ``themes`` is empty.

    ``[]`` -> ``""``, ``[A]`` -> ``"(A)"``, ``[A, B]`` -> ``"(A and B)"``,
    ``[A, B, C, ...]`` -> ``"(A, B and C)"``.
    """
    if not themes:
        return ""
    if len(themes) == 1:
        return f"({themes[0]})"
    return f"({', '.join(themes[:-1])} and {themes[-1]})"


_HTML_WRAPPER = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
</head>
<body style="margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;background-color:#e0f2fe;color:#1a202c;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color:#e0f2fe;">
    <tr><td align="center" style="padding:40px 20px;">
      <table role="presentation" width="680" cellspacing="0" cellpadding="0" border="0"
        style="background-color:#ffffff;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.08);overflow:hidden;max-width:100%;">
        <tr><td style="background-color:{header_color};padding:40px 48px;text-align:center;">
          <p style="margin:0 0 8px 0;color:rgba(255,255,255,0.9);font-size:14px;font-weight:500;text-transform:uppercase;letter-spacing:2px;">{header_label}</p>
          <h1 style="margin:0;color:#ffffff;font-size:28px;font-weight:700;letter-spacing:-0.5px;">{topic_name}</h1>
          <p style="margin:16px 0 0 0;color:rgba(255,255,255,0.95);font-size:15px;font-weight:500;">{date_str}</p>
        </td></tr>
        <tr><td style="background-color:{accent_color};height:6px;"></td></tr>
        <tr><td style="padding:48px;">
          <div style="color:#2d3748;font-size:16px;line-height:1.7;">
            {body_html}
          </div>
        </td></tr>
        <tr><td style="padding:24px 48px;border-top:1px solid #e2e8f0;text-align:center;">
          <p style="margin:0;color:#718096;font-size:13px;">Digest of {topic_name} research from arXiv</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def _build_stats(selected: list[dict], remaining: list[dict], total_scanned: int) -> dict:
    all_relevant = selected + remaining
    topics: dict[str, int] = {}
    for p in all_relevant:
        t = p.get("primary_topic", "Unknown")
        topics[t] = topics.get(t, 0) + 1
    top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]

    all_institutions: list[str] = []
    for p in all_relevant:
        all_institutions.extend(p.get("affiliations", []))
    inst_counts: dict[str, int] = {}
    for inst in all_institutions:
        inst_counts[inst] = inst_counts.get(inst, 0) + 1
    top_inst = sorted(inst_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "papers_scanned": total_scanned,
        "papers_relevant": len(all_relevant),
        "papers_deep_analyzed": len(selected),
        "pdf_success_count": sum(1 for p in selected if p.get("pdf_analysis_performed")),
        "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
        "top_institutions": [{"institution": i, "count": c} for i, c in top_inst],
    }


def _build_this_period_numbers(stats: dict, config: dict) -> str:
    topics_str = ", ".join(t["topic"] for t in stats["top_topics"])
    inst_str = ", ".join(f"{i['institution']} ({i['count']})" for i in stats["top_institutions"])
    return (
        f"- {stats['papers_scanned']} paper abstracts analyzed\n"
        f"- {stats['papers_relevant']} deemed relevant "
        f"(score ≥ {config['thresholds']['relevance_score_min']}, "
        f"credibility ≥ {config['thresholds']['credibility_tier_min']})\n"
        f"- {stats['papers_deep_analyzed']} papers received full-text deep analysis\n"
        f"- Top research themes: {topics_str}\n"
        f"- high-credibility institutions: {inst_str}"
    )


def get_trend_section(config: dict, client) -> str:
    from mira.config import llm_call, model_for

    reports_dir = REPORT_FILES / "prod" / config["profile_id"]
    if not reports_dir.exists():
        return ""

    report_files = sorted(reports_dir.glob("*.json"), reverse=True)[:12]
    if not report_files:
        return ""

    prior_reports = []
    for f in report_files:
        try:
            prior_reports.append(json.loads(f.read_text()))
        except Exception:
            continue

    if not prior_reports:
        return ""

    model = model_for(config, "trend")
    system = "You are a research analyst identifying trends. Be concise and specific."
    trimmed_reports = []
    budget = 30000
    for r in prior_reports:
        serialized = json.dumps(r, indent=2)
        if budget - len(serialized) < 0:
            break
        trimmed_reports.append(r)
        budget -= len(serialized)
    user = (
        f"Analyze the following {len(trimmed_reports)} prior {config['topic']['focus']} research digests "
        f"(spanning up to 60 days) and write 2-3 paragraphs identifying key emerging trends, "
        f"institutional activity patterns, and notable shifts in research focus.\n\n"
        f"Prior reports (JSON):\n{json.dumps(trimmed_reports, indent=2)}\n\n"
        f"Current date: {config['current_date']}"
    )
    try:
        return llm_call(client, model, system, user)
    except Exception:
        return ""


def generate_report(
    selected: list[dict],
    remaining: list[dict],
    media: list[dict],
    config: dict,
    client,
    trend_section: str,
    total_scanned: int,
) -> dict:
    from mira.config import (apply_template, parse_json_response, llm_call,
                             model_for, record_parse_failure)

    stats = _build_stats(selected, remaining, total_scanned)
    this_period_numbers = _build_this_period_numbers(stats, config)

    mode_cfg = config["mode_cfg"]
    prompt = config["prompts"]["report"]
    remaining_capped = sorted(
        remaining, key=lambda p: p.get("relevance_score", 0) + p.get("credibility_tier", 0), reverse=True
    )[:20]

    enriched_selected = []
    for paper in selected:
        enriched = dict(paper)
        enriched["theme_label"] = format_theme_label(match_themes(paper, config.get("themes", [])))
        enriched_selected.append(enriched)

    user = apply_template(prompt["user"], {
        "period_label": config["mode"],
        "topic_focus": config["topic"]["focus"],
        "stats_json": json.dumps(stats, indent=2),
        "analyzed_papers_json": json.dumps(enriched_selected, indent=2),
        "remaining_count": str(len(remaining)),
        "remaining_selection_policy": "Top 20 by relevance_score + credibility_tier",
        "remaining_papers_json": json.dumps(remaining_capped, indent=2),
        "period_title": config["mode"].capitalize(),
        "this_period_numbers": this_period_numbers,
        "assistant_signature": config["topic"].get("assistant_signature", "MIRA"),
        "mode_label": config["mode"].capitalize(),
        "topic_short_label": config["topic"].get("short_label", config["topic"]["name"]),
        "current_date": config["current_date"],
    })
    system = apply_template(prompt.get("system", ""), {"topic_focus": config["topic"]["focus"]})

    if media:
        user += f"\n\n## MEDIA INTELLIGENCE\n```json\n{json.dumps(media, indent=2)}\n```"
    if trend_section:
        user += f"\n\n## TREND ANALYSIS\n{trend_section}"

    model = model_for(config, "report")
    raw = llm_call(client, model, system, user)

    try:
        result = parse_json_response(raw)
    except Exception:
        record_parse_failure("report")
        result = {"subject": f"MIRA Digest - {config['current_date']}", "body": raw}

    return result


def to_html(body_markdown: str, subject: str, config: dict) -> str:
    email_cfg = config["email_cfg"]
    colors = email_cfg.get("colors", {})
    body_html = md.markdown(
        body_markdown,
        extensions=["extra", "nl2br", "sane_lists"],
    )
    d = datetime.strptime(config["current_date"], "%Y-%m-%d")
    day_str = f"{d.strftime('%A, %B')} {d.day}, {d.year}"
    return _HTML_WRAPPER.format(
        title=subject,
        header_label=email_cfg.get("header_label", "Research Digest"),
        topic_name=config["topic"]["name"],
        date_str=day_str,
        header_color=colors.get("primary", "#0891b2"),
        accent_color=colors.get("accent", "#14b8a6"),
        body_html=body_html,
    )


def save_report(html: str, report_json: dict, config: dict) -> Path:
    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    slug = config["topic"].get("short_label", config["profile_id"]).lower().replace(" ", "-")

    html_dir = REPORT_FILES
    html_dir.mkdir(exist_ok=True)
    html_path = html_dir / f"report-{slug}-{ts}.html"
    html_path.write_text(html, encoding="utf-8")

    json_dir = REPORT_FILES / "prod" / config["profile_id"]
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"report-{slug}-{config['current_date']}-{ts}.json"
    json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    return html_path

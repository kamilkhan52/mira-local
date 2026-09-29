from __future__ import annotations
from mira.paths import REPORT_FILES
import json
import re
import smtplib
import ssl
import os
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import markdown as md

from mira.graph_ingest import normalize_topic

ROOT = Path(__file__).parent.parent


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


def send_email(html: str, subject: str, config: dict) -> None:
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not gmail_user:
        raise EnvironmentError("GMAIL_USER environment variable is not set in .env")
    if not gmail_password:
        raise EnvironmentError("GMAIL_APP_PASSWORD environment variable is not set in .env")
    recipient = config["recipient_email"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls(context=context)
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, recipient, msg.as_string())

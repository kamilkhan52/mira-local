"""Realtime monitor ("MIRA Live"): breaking industry news (and optionally new
arXiv papers), triaged by Jev, emailed to subscribers.

Each run (hourly via run_realtime.py):
  1. fetch recent arXiv papers per profile and crawl the news sources once;
  2. drop anything already in the profile's ledger (seen before);
  3. Jev judges each new item — paper relevance/topic from title+abstract, then
     credibility from the first-page header for papers that pass relevance;
     article relevance from title+content;
  4. items past the Jev-calibrated cutoffs are ranked; the top ones get a short
     summary written for the profile's preferences (LLM; Jev-selected key
     sentence if the LLM is unavailable) and the digest is emailed.

Every judged item is appended to judged.jsonl so the realtime triage can be
compared with the weekly LLM workflow on the same papers.
"""
from __future__ import annotations
from mira.paths import REALTIME_DIR, CONFIG_DIR

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from mira import jev

ROOT = Path(__file__).parent.parent
# Keep MIRA_DATA_DIR on a disk background jobs can write to (macOS blocks
# launchd jobs from external volumes without Full Disk Access).
STATE_DIR = REALTIME_DIR

# Jev relevance-level cutoffs per profile, calibrated against the cached LLM
# baseline on two disjoint samples (scripts/bench_jev.py, seeds 42+43):
#   gate      — agreement-maximising equivalent of the weekly relevance gate
#               (86-92% agreement with the LLM's pass/fail)
#   priority  — equivalent of LLM relevance >= 7 (marked "high priority")
#   cred_gate — credibility level (first-page header) equivalent of the
#               weekly credibility gate
ALERT_CUTOFFS = jev.CUTOFFS  # calibrated in mira/jev.py

NEWS_CRAWLERS = [
    ("ee-times-crawler.ts",     "eetimes",      "EE Times"),
    ("semianalysis-crawler.ts", "semianalysis", "SemiAnalysis"),
    ("trendforce-crawler.ts",   "trendforce",   "TrendForce"),
    ("digitimes-crawler.ts",    "digitimes",    "Digitimes"),
]


def load_settings() -> dict:
    s = json.loads((CONFIG_DIR / "realtime.json").read_text())
    subs = {}
    for pid, lst in s.get("subscribers", {}).items():
        resolved = []
        for entry in lst:
            m = re.fullmatch(r"\$\{(\w+)\}", entry.strip())
            val = os.environ.get(m.group(1), "") if m else entry.strip()
            if val:
                resolved.append(val)
        subs[pid] = resolved
    s["subscribers"] = subs
    return s


# ---------------------------------------------------------------- ledger --

class Ledger:
    """Ids/URLs already judged for a profile, so nothing is alerted twice."""

    def __init__(self, profile_id: str):
        self.dir = STATE_DIR / profile_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "seen.json"
        try:
            self.seen: dict[str, str] = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            self.seen = {}

    def is_new(self, key: str) -> bool:
        return key not in self.seen

    def mark(self, keys) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        for k in keys:
            self.seen.setdefault(k, now)

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.seen))
        tmp.replace(self.path)

    def log_judged(self, rows: list[dict]) -> None:
        with (self.dir / "judged.jsonl").open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


# --------------------------------------------------------------- sources --

def fetch_recent_papers(config: dict, lookback_days: int, max_results: int) -> list[dict]:
    from mira.fetch import fetch_papers
    end = datetime.now()
    cfg = dict(config)
    cfg["arxiv"] = {**config["arxiv"], "max_results": max_results}
    cfg["start_date"] = (end - timedelta(days=lookback_days)).strftime("%Y%m%d")
    cfg["end_date"] = end.strftime("%Y%m%d")
    return fetch_papers(cfg)


def crawl_news(lookback_days: int) -> list[dict]:
    from mira.media import _normalize_articles, _run_crawler
    end = datetime.now()
    window = {"start_date_iso": (end - timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
              "end_date_iso": end.strftime("%Y-%m-%d")}
    articles = []
    for script, key, source in NEWS_CRAWLERS:
        raw = _run_crawler(script, window, key)
        articles += [a for a in _normalize_articles(raw, source) if a.get("url")]
        print(f"  {source}: {len(raw)} articles")
    return articles


# ---------------------------------------------------------------- triage --

def _parallel(fn, items, workers=16):
    def safe(x):
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001 — one failure must not stop the run
            print(f"  WARNING: Jev call failed — {e}")
            return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(safe, items))


def triage_papers(papers: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    """Returns (passing papers ranked, all judged rows for the log). Papers whose
    Jev call failed are left out of both, so they are retried next run."""
    from mira.fetch import extract_first_pages

    pid = config["profile_id"]
    cut = ALERT_CUTOFFS.get(pid, {"decide": 1.0, "priority": 2.0, "cred_gate": 1.0})
    rubric = jev.profile_rubric(config)
    judged = _parallel(lambda p: jev.judge_paper(p, rubric), papers)
    rows, candidates = [], []
    for p, j in zip(papers, judged):
        if j is None:
            continue
        p["jev"] = j
        rows.append(p)
        if j["relevance_level"] >= cut["decide"]:
            candidates.append(p)

    # Credibility needs the first page; only fetch it for relevant papers.
    if candidates:
        extract_first_pages(candidates)
        creds = _parallel(lambda p: jev.judge_credibility(
            config["topic"]["focus"], first_page_text=p.get("first_page_text", "")), candidates)
        for p, c in zip(candidates, creds):
            p["jev_cred"] = c
    passing = [p for p in candidates
               if p.get("first_page_text", "").strip() == "" or p.get("jev_cred") is None
               or p["jev_cred"]["credibility_level"] >= cut["cred_gate"]]
    for p in passing:
        p["priority"] = p["jev"]["relevance_level"] >= cut["priority"]
    passing.sort(key=lambda p: (p["jev"]["relevance_level"],
                                (p.get("jev_cred") or {}).get("credibility_level", 0)), reverse=True)
    alerted = {p["id"] for p in passing}
    log = [{"kind": "paper", "id": p["id"], "title": p["title"], "published": p.get("published"),
            "relevance_level": p["jev"]["relevance_level"],
            "relevance_score": p["jev"]["relevance_score"],
            "primary_topic": p["jev"]["primary_topic"],
            "credibility_level": (p.get("jev_cred") or {}).get("credibility_level"),
            "alerted": p["id"] in alerted} for p in rows]
    return passing, log


def triage_news(articles: list[dict], config: dict, min_level: float) -> tuple[list[dict], list[dict]]:
    guidance = config.get("media", {}).get("selection_guidance", "")
    focus = config["topic"]["focus"]
    judged = _parallel(lambda a: jev.judge_article(a, focus, guidance), articles)
    rows, passing = [], []
    for a, j in zip(articles, judged):
        if j is None:
            continue
        a = dict(a, jev=j)
        rows.append(a)
        if j["relevance_level"] >= min_level:
            passing.append(a)
    passing.sort(key=lambda a: a["jev"]["relevance_level"], reverse=True)
    alerted = {a["url"] for a in passing}
    log = [{"kind": "news", "id": a["url"], "title": a["title"], "source": a["source"],
            "relevance_level": a["jev"]["relevance_level"], "alerted": a["url"] in alerted} for a in rows]
    return passing, log


# -------------------------------------------------------------- summaries --

def _preferences(config: dict) -> str:
    r = jev.profile_rubric(config)
    lines = [f"Team: {r['team']}.", f"Focus area: {r['focus']}."]
    lines += [f"- {g}" for g in r["guidance"]]
    guidance = config.get("media", {}).get("selection_guidance")
    if guidance:
        lines.append(f"- News preferences: {guidance}")
    return "\n".join(lines)


def summarize(item: dict, kind: str, config: dict, client, model: str, max_tokens: int) -> tuple[str, str]:
    """(summary, how) — how is 'llm' or 'key sentence' (Jev-selected fallback)."""
    text = item.get("summary") if kind == "paper" else (item.get("content") or "")[:6000]
    if client is not None:
        system = ("You write short alerts for a research team. Their preferences:\n"
                  f"{_preferences(config)}\n"
                  "Write 2-3 plain sentences: what is new, and why it matters (or not) for "
                  "this team specifically. No preamble, no markdown, no hype.")
        user = f"{'Paper' if kind == 'paper' else 'Article'}: {item['title']}\n\n{text}"
        try:
            from mira import usage
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_body={"usage": {"include": True}})
            usage.record_llm("summarize", model, getattr(resp, "usage", None), time.perf_counter() - t0)
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out, "llm"
        except Exception as e:  # noqa: BLE001 — fall back to extraction
            print(f"  WARNING: summary LLM failed ({str(e)[:120]}); using key sentence")
    return key_sentence(text or "", config), "key sentence"


def key_sentence(text: str, config: dict) -> str:
    """Jev picks the sentence most relevant to the team (select, don't generate)."""
    from typesafe_sdk import Choice
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30][:40]
    if len(sents) < 2:
        return sents[0] if sents else ""
    r = jev.profile_rubric(config)
    try:
        ans = jev.client().system_one(
            {"sentences": {f"s{i}": s for i, s in enumerate(sents)}},
            {"pick": Choice(
                instructions={"team": r["team"],
                              "question": "Which sentence in `sentences` best states the finding "
                                          "that matters most to `team`?"},
                criteria={f"s{i}": None for i in range(len(sents))})})
        return sents[int(ans.choices["pick"].choice[1:])]
    except Exception:  # noqa: BLE001
        return sents[0]


# ------------------------------------------------------------------ email --

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def build_digest(config: dict, papers: list[dict], listed: list[dict], news: list[dict],
                 n_scanned: int, n_news_scanned: int, n_more: int = 0) -> tuple[str, str]:
    """(subject, markdown body)."""
    focus_name = config["topic"].get("short_label") or config["topic"]["name"]
    now = datetime.now()
    parts = []
    if papers or listed:
        parts.append(f"{len(papers) + len(listed)} paper{'s' if len(papers) + len(listed) != 1 else ''}")
    if news:
        parts.append(f"{len(news)} news")
    subject = f"[MIRA Live] {focus_name}: {', '.join(parts)} — {now:%b %d %H:%M}"

    L = [f"**Live update** · {now:%A %b %d, %H:%M} · {n_scanned} new papers and "
         f"{n_news_scanned} new articles scanned",
         ""]
    if papers:
        L += ["## New papers", ""]
        for p in papers:
            badge = " · **High priority**" if p.get("priority") else ""
            cred = p.get("jev_cred")
            L += [f"### [{_clean(p['title'])}](https://arxiv.org/abs/{p['id']})",
                  f"*arXiv {p['id']} · {p.get('published', '')} · "
                  f"{', '.join(p.get('authors', [])[:4])}{' et al.' if len(p.get('authors', [])) > 4 else ''}*",
                  "",
                  f"Topic: {p['jev']['primary_topic']} · Relevance {p['jev']['relevance_score']}/10"
                  + (f" · Credibility {cred['credibility_tier']}/10" if cred else "") + badge,
                  "",
                  f"**Summary{' (key sentence)' if p['summary_how'] != 'llm' else ''}:** {p['alert_summary']}",
                  "",
                  f"**Abstract:** {_clean(p['summary'])}",
                  ""]
    if listed:
        L += ["## Also relevant", ""]
        for p in listed:
            L.append(f"- [{_clean(p['title'])}](https://arxiv.org/abs/{p['id']}) — "
                     f"{p['jev']['primary_topic']} · relevance {p['jev']['relevance_score']}/10")
        if n_more:
            L.append(f"- …and {n_more} more relevant paper{'s' if n_more != 1 else ''} not shown")
        L.append("")
    if news:
        L += ["## News", ""]
        for a in news:
            L += [f"### [{_clean(a['title'])}]({a['url']})",
                  f"*{a['source']}{' · ' + a['date'] if a.get('date') else ''}*",
                  "",
                  f"**Summary{' (key sentence)' if a['summary_how'] != 'llm' else ''}:** {a['alert_summary']}",
                  "",
                  f"**Excerpt:** {_clean(a.get('content', ''))[:500]}…",
                  ""]
    L += ["---",
          "*Triage by Jev (TypeSafe); relevance and credibility use cutoffs calibrated "
          "against the weekly workflow's LLM judgments. Summaries are written for this "
          "profile's preferences.*"]
    return subject, "\n".join(L)


# -------------------------------------------------------------------- run --

def run_profile(profile_id: str, settings: dict, news_pool: list[dict] | None,
                client, *, dry_run: bool = False) -> dict:
    from mira.config import load_config, shared_llm_model
    from mira.report import send_email, to_html

    config = load_config(profile_id)  # profile default mode; realtime only needs topic/prompts/email
    ledger = Ledger(profile_id)
    t0 = time.perf_counter()

    if settings.get("include_papers", False):
        papers = fetch_recent_papers(config, settings["arxiv_lookback_days"], settings["arxiv_max_results"])
        new_papers = [p for p in papers if ledger.is_new(p["id"])]
        print(f"  arXiv: {len(papers)} in window, {len(new_papers)} new")
        passing, paper_log = triage_papers(new_papers, config)
    else:  # news-first: papers arrive once a day and wait for the weekly digest
        new_papers, passing, paper_log = [], [], []

    news_new = [a for a in (news_pool or []) if ledger.is_new(a["url"])]
    news_pass, news_log = triage_news(news_new, config, settings["news_min_level"])
    print(f"  Jev: {len(passing)} papers and {len(news_pass)} articles pass")

    featured = passing[:settings["max_featured_papers"]]
    listed = passing[settings["max_featured_papers"]:
                     settings["max_featured_papers"] + settings["max_listed_papers"]]
    news = news_pass[:settings["max_featured_news"]]

    model = shared_llm_model("realtime_summary", shared_llm_model("media_summary", "anthropic/claude-sonnet-5"))
    for item, kind in [(p, "paper") for p in featured] + [(a, "news") for a in news]:
        item["alert_summary"], item["summary_how"] = summarize(
            item, kind, config, client, model, settings["summary_max_tokens"])

    result = {"profile": profile_id, "papers_new": len(new_papers), "papers_alerted": len(passing),
              "news_new": len(news_new), "news_alerted": len(news_pass),
              "seconds": round(time.perf_counter() - t0, 1), "email": None}
    if featured or listed or news:
        n_more = max(0, len(passing) - len(featured) - len(listed))
        subject, body = build_digest(config, featured, listed, news, len(new_papers), len(news_new), n_more)
        html = to_html(body, subject, config)
        out = ledger.dir / f"alert-{datetime.now():%Y%m%dT%H%M%S}.html"
        out.write_text(html, encoding="utf-8")
        result["html"] = str(out)
        subs = settings["subscribers"].get(profile_id, [])
        if dry_run:
            result["email"] = f"dry run — not sent to {len(subs)} subscriber(s)"
        else:
            sent = 0
            for sub in subs:
                try:
                    send_email(html, subject, {**config, "recipient_email": sub})
                    sent += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  WARNING: email to subscriber failed — {e}")
            result["email"] = f"sent to {sent}/{len(subs)} subscriber(s)"
            if sent == 0 and subs:
                # Keep items unseen so the next run retries the alert.
                return result

    if not dry_run:
        ledger.mark(r["id"] for r in paper_log)
        ledger.mark(r["id"] for r in news_log)
        ledger.log_judged(paper_log + news_log)
        ledger.save()
    return result

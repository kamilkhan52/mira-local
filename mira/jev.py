"""Jev (TypeSafe System One) judgments for the paper-triage stages.

Jev returns typed judgments with probabilities in ~150 ms; it does not generate
text. It is used here only for the judgment-shaped parts of the per-paper
stages — relevance, primary topic, impact level, actionable, and institutional
credibility — never for extraction (affiliations) or prose (key_findings,
reasoning), which stay with the LLM.

Questions are built from the profile's own prompts so each profile keeps its
focus, priorities and topic taxonomy. Enabled when TYPESAFE_API_KEY is set.
"""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache

JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")

# Relevance levels, low to high. The expectation over these (0..4) maps onto the
# LLM's 1-10 relevance_score bands: 1-2, 3-4, 5-6, 7-8, 9-10.
RELEVANCE_LEVELS = [
    "Unrelated: the paper is not about the team's focus area at all.",
    "Tangential: the paper touches the team's focus area only incidentally, "
    "with no concrete implication for it.",
    "Adjacent: the paper is mainly about something else but has explicit, "
    "concrete implications for the team's focus area.",
    "Directly relevant: the paper is substantially about the team's focus area.",
    "Core: the paper is a central contribution to the team's focus area and is "
    "directly applicable to practical development work in it.",
]

# Credibility levels, low to high. Examples are the affiliation prompt's own
# reference lists. Expectation (0..3) maps onto the LLM's 1-10 credibility_tier
# bands: 1-3 (LOW), 4-5 and 6-7 (MEDIUM), 8-10 (HIGH).
CREDIBILITY_LEVELS = [
    "Low: unknown or obscure institutions, no clear institutional affiliation, "
    "or individual researchers without institutional backing.",
    "Lower medium: a known but less prominent university or company with some "
    "research capability.",
    "Upper medium: an established research university with strong CS/EE "
    "programs, a government research lab (e.g. Sandia, Lawrence Livermore), "
    "a tech company with an R&D division, or an established startup in the field.",
    "High: a leading company or research lab (e.g. Samsung, Micron, SK Hynix, "
    "Intel, AMD, NVIDIA, Google Research, Microsoft Research, IBM Research, "
    "Meta AI, Apple, Huawei, imec, TSMC, Fraunhofer) or a top university "
    "(e.g. MIT, Stanford, CMU, UC Berkeley, ETH Zurich, KAIST, Tsinghua, NUS, "
    "Georgia Tech, UIUC, Princeton, Harvard).",
]
_CRED_BANDS = [2, 4.5, 6.5, 9]  # representative tier per level, for the 1-10 mapping

IMPACT_OPTIONS = {
    "Low": "Incremental or narrow; unlikely to change practice.",
    "Medium": "A solid contribution that could influence some designs or studies.",
    "High": "A significant advance likely to influence product or research directions.",
    "Breakthrough": "A step change that could reshape the field.",
}
ACTIONABLE_OPTIONS = {
    "Yes": "The team could act on this work now (adopt, evaluate, or respond to it).",
    "Maybe": "Possibly worth acting on, depending on further evaluation.",
    "No": "Nothing for the team to act on.",
}

# Header of the first page (titles, authors, affiliations). Affiliations sit
# alongside the author list, so the rest of the page is distractor text.
FIRST_PAGE_HEADER_CHARS = 2500


@lru_cache(maxsize=1)
def client():
    from typesafe_sdk import TypeSafeClient
    return TypeSafeClient(model=JEV_MODEL)


def enabled() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY"))


# --------------------------------------------------------------------------
# Profile rubric: pull the judgment-relevant parts out of the LLM prompt.
# --------------------------------------------------------------------------

_SKIP_FIELD = re.compile(
    r"^- (secondary_topics|key_findings|actionable|potential_impact|arxiv_id|"
    r"Return ONLY|Do NOT|Output must)", re.I)


def _semantic_sentences(line: str) -> list[str]:
    """Keep the meaning of a guidance line, dropping the LLM's numeric scale
    and output-field instructions (Jev reads these literally; its own levels
    define the scale)."""
    line = re.sub(r"^relevance_score guidance:\s*", "", line)
    if line.startswith("relevance_score:"):
        line = re.sub(r"^relevance_score:\s*1-10 where[^.]*\.\s*", "", line)
    out = []
    for sent in re.split(r"(?<=\.)\s+", line):
        sent = re.sub(r',?\s*set primary_topic to "([^"]+)"', r", treat it as \1", sent)
        if not sent or re.search(r"\b\d+-\d+\b|^\d+ means|primary_topic|relevance_score", sent):
            continue
        out.append(sent)
    return out


def profile_rubric(profile: dict) -> dict:
    """Team focus sentence, relevance guidance lines and topic taxonomy, taken
    from the profile's classification system prompt."""
    focus = profile["topic"]["focus"]
    system = profile["prompts"]["classification"].get("system", "").replace(
        "{{topic_focus}}", focus)
    lines = [l.rstrip() for l in system.splitlines()]

    team = next((l.strip() for l in lines if l.strip().startswith("Your task is")), "")
    team = re.sub(r"^Your task is to evaluate this paper's significance for ", "", team)
    team = team.rstrip(".") or f"a team working on {focus}"

    guidance, taxonomy = [], []
    in_rules = in_taxonomy = False
    for l in lines:
        s = l.strip()
        if s.startswith("IMPORTANT OUTPUT RULES"):
            in_rules = True
            continue
        if in_rules:
            if not s:
                in_rules = False
            continue
        # A heading ending in ':' that names categories opens a taxonomy block.
        if s.endswith(":") and re.search(r"categor|classification|catch-all", s, re.I):
            in_taxonomy = True
            continue
        if in_taxonomy:
            if s.startswith("- ") and not re.match(r"- \w+(_\w+)+:", s):
                taxonomy.append(s[2:].strip())
                continue
            if s:
                in_taxonomy = False
        if s.startswith("- ") and _SKIP_FIELD.match(s):
            continue
        if s.startswith("- primary_topic"):
            continue
        if (s.startswith("- ") or s.startswith("NON-NEGOTIABLE")) and re.search(
                r"relevan|priorit|strategic|memory-system|storage|not related|"
                r"implication|NON-NEGOTIABLE|lower", s, re.I):
            guidance.extend(_semantic_sentences(s.lstrip("- ").strip()))

    taxonomy = [re.sub(r"\s*\(specify which\)|specify \w+: ", "", t) for t in taxonomy]
    taxonomy = list(dict.fromkeys(taxonomy))
    if not any(t.lower().startswith("not related") for t in taxonomy):
        taxonomy.append(f"Not related to {focus}")
    return {"focus": focus, "team": team, "guidance": guidance, "taxonomy": taxonomy}


def paper_questions(rubric: dict) -> dict:
    from typesafe_sdk import Choice, Score

    team = {"team": rubric["team"], "focus_area": rubric["focus"]}
    if rubric["guidance"]:
        team["relevance_guidance"] = rubric["guidance"]
    return {
        "relevance": Score(
            instructions={
                **team,
                "question": "How relevant is `paper` to the work of `team`, "
                            "following `relevance_guidance`?",
            },
            criteria=RELEVANCE_LEVELS,
        ),
        "primary_topic": Choice(
            instructions={
                "focus_area": rubric["focus"],
                "question": "Which category best describes the main subject of `paper`? "
                            "Choose the 'Not related' category if `paper` is not "
                            "substantially about `focus_area`.",
            },
            criteria={t: None for t in rubric["taxonomy"]},
        ),
        "potential_impact": Choice(
            instructions={
                "focus_area": rubric["focus"],
                "question": "What is the potential impact of `paper` on `focus_area`?",
            },
            criteria=IMPACT_OPTIONS,
        ),
        "actionable": Choice(
            instructions={
                **team,
                "question": "Is `paper` actionable for `team`?",
            },
            criteria=ACTIONABLE_OPTIONS,
        ),
    }


def credibility_question(focus: str):
    from typesafe_sdk import Score
    return Score(
        instructions={
            "focus_area": focus,
            "question": "How credible are the institutions the authors of `paper` are "
                        "affiliated with, as sources of `focus_area` research? Judge the "
                        "most reputable affiliated institution.",
        },
        criteria=CREDIBILITY_LEVELS,
    )


# --------------------------------------------------------------------------
# Calls. Each returns the pipeline field names plus the raw answer for audit.
# --------------------------------------------------------------------------

def relevance_to_10(expectation: float) -> int:
    """Map a 0..4 relevance expectation onto the LLM's 1-10 scale (band midpoints)."""
    return max(1, min(10, round(1.5 + expectation * 2)))


def credibility_to_10(expectation: float) -> int:
    lo = int(expectation)
    hi = min(lo + 1, len(_CRED_BANDS) - 1)
    frac = expectation - lo
    return max(1, min(10, round(_CRED_BANDS[lo] * (1 - frac) + _CRED_BANDS[hi] * frac)))


def judge_paper(paper: dict, rubric: dict) -> dict:
    """Relevance, primary topic, impact and actionable from title + abstract."""
    state = {"paper": {"title": paper["title"], "abstract": paper["summary"]}}
    t0 = time.perf_counter()
    r = client().system_one(state, paper_questions(rubric))
    latency = time.perf_counter() - t0
    rel = r.scores["relevance"]
    topic = r.choices["primary_topic"]
    return {
        "relevance_score": relevance_to_10(rel.score),
        "relevance_level": rel.score,
        "relevance_confidence": rel.confidence,
        "primary_topic": topic.choice,
        "primary_topic_confidence": topic.confidence,
        "primary_topic_probabilities": dict(topic.probabilities),
        "potential_impact": r.choices["potential_impact"].choice,
        "actionable": r.choices["actionable"].choice,
        "latency_s": latency,
        "input_tokens": r.usage.input_tokens,
    }


def judge_credibility(focus: str, *, affiliations: list[str] | None = None,
                      first_page_text: str | None = None) -> dict:
    """Credibility tier from an affiliation list (preferred) or the first-page
    header. Exactly one source should be given."""
    if affiliations is not None:
        state = {"paper": {"author_affiliations": affiliations or ["None stated"]}}
    else:
        state = {"paper": {"first_page_header": (first_page_text or "")[:FIRST_PAGE_HEADER_CHARS]}}
    t0 = time.perf_counter()
    r = client().system_one(state, {"credibility": credibility_question(focus)})
    latency = time.perf_counter() - t0
    cred = r.scores["credibility"]
    return {
        "credibility_tier": credibility_to_10(cred.score),
        "credibility_level": cred.score,
        "credibility_confidence": cred.confidence,
        "latency_s": latency,
        "input_tokens": r.usage.input_tokens,
    }


# --------------------------------------------------------------------------
# News articles: per-article relevance, ranked in code (media_selection).
# --------------------------------------------------------------------------

ARTICLE_LEVELS = [
    "Unrelated: the article is not about the digest's focus area.",
    "Passing mention: the focus area appears only as brief context for a story "
    "about something else.",
    "Partly relevant: a meaningful part of the article concerns the focus area.",
    "Central: the article is mainly about the focus area.",
]
ARTICLE_CONTENT_CHARS = 3000


def judge_article(article: dict, focus: str, guidance: str) -> dict:
    """Relevance of one news article to a digest; code ranks and takes the top N."""
    from typesafe_sdk import Score

    instructions = {"digest_focus": focus,
                    "question": "How relevant is `article` to a `digest_focus` digest"
                                + (", following `selection_guidance`?" if guidance else "?")}
    if guidance:
        instructions["selection_guidance"] = guidance
    state = {"article": {"source": article.get("source", ""),
                         "title": article.get("title", ""),
                         "content": (article.get("content") or "")[:ARTICLE_CONTENT_CHARS]}}
    t0 = time.perf_counter()
    r = client().system_one(state, {"relevance": Score(instructions=instructions,
                                                       criteria=ARTICLE_LEVELS)})
    ans = r.scores["relevance"]
    return {"relevance_level": ans.score, "relevance_confidence": ans.confidence,
            "latency_s": time.perf_counter() - t0, "input_tokens": r.usage.input_tokens}


def select_articles(articles: list[dict], focus: str, guidance: str, n: int = 10,
                    workers: int = 16) -> list[dict]:
    """Jev counterpart of media._filter_articles: score each article, keep the top n."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(lambda a: judge_article(a, focus, guidance), articles))
    ranked = sorted(zip(articles, scores), key=lambda x: -x[1]["relevance_level"])
    return [dict(a, jev_relevance=s["relevance_level"]) for a, s in ranked[:n]]


# --------------------------------------------------------------------------
# Pre-screen for the weekly/daily CLI run (opt-in: run.py --jev-prescreen).
# --------------------------------------------------------------------------

# Relevance-level cutoffs below which a paper skips first-page extraction,
# affiliation and classification. Chosen so that on two disjoint samples of the
# cached LLM baseline (scripts/bench_jev.py, seeds 42 and 43) >= 99% of papers
# the LLM passed and 100% of papers featured in production reports were kept.
# Profiles without a validated cutoff are not pre-screened.
PRESCREEN_CUTOFFS = {
    "memory-innovation": 0.15,
    "cxl-research": 0.05,
    "storage-innovation": 0.05,
}


def prescreen(papers: list[dict], profile: dict, workers: int = 16) -> tuple[list[dict], list[dict]]:
    """Split papers into (kept, screened). Screened papers are clearly below the
    profile's relevance gate; kept papers go through the LLM stages unchanged.
    Any Jev failure keeps the paper (fails open to the existing path)."""
    from concurrent.futures import ThreadPoolExecutor

    cutoff = PRESCREEN_CUTOFFS.get(profile["profile_id"])
    if cutoff is None or not papers:
        return papers, []
    rubric = profile_rubric(profile)

    def one(p):
        try:
            return judge_paper(p, rubric)
        except Exception as e:  # noqa: BLE001 — fail open
            print(f"  WARNING: Jev pre-screen failed for {p['id']} — {e}. Keeping.")
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        judgments = list(pool.map(one, papers))
    kept, screened = [], []
    for p, j in zip(papers, judgments):
        if j is not None and j["relevance_level"] < cutoff:
            p["jev_prescreen"] = {k: j[k] for k in ("relevance_level", "relevance_confidence",
                                                    "primary_topic")}
            screened.append(p)
        else:
            kept.append(p)
    return kept, screened

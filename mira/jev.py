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

# Decision-model backend. All speak TypeSafe's /v1/systemone API, so the same
# SDK and questions work against each; only the server, option limit, price
# and calibrated cutoffs differ.
#   typesafe  Jev, TypeSafe's hosted model (needs TYPESAFE_API_KEY)
#   nimble    Bespoke-Nimble-9B served locally by Ollama >= 0.35
#   kev       Kev (jaredpalmer/kev) served locally by `python -m kev.serve`
BACKENDS = {
    "typesafe": {"base_url": None, "model": "jev-latest", "max_options": 255, "price_per_mtok": 0.042},
    "nimble": {"base_url": "http://127.0.0.1:11435", "model": "nimble", "max_options": 26, "price_per_mtok": 0.0},
    "kev": {"base_url": "http://127.0.0.1:8009", "model": "kev-latest", "max_options": 255, "price_per_mtok": 0.0},
}
BACKEND = os.environ.get("JEV_BACKEND", "typesafe")
if BACKEND not in BACKENDS:
    raise ValueError(f"JEV_BACKEND must be one of {sorted(BACKENDS)}")
_B = BACKENDS[BACKEND]
JEV_MODEL = os.environ.get("JEV_MODEL", _B["model"])
JEV_BASE_URL = os.environ.get("JEV_BASE_URL", _B["base_url"] or "") or None

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


def set_backend(name: str) -> None:
    """Switch the decision-model backend for this process (a flow run)."""
    global BACKEND, _B, JEV_MODEL, JEV_BASE_URL
    if name not in BACKENDS:
        raise ValueError(f"jev_backend must be one of {sorted(BACKENDS)}")
    BACKEND, _B = name, BACKENDS[name]
    JEV_MODEL = _B["model"]
    JEV_BASE_URL = _B["base_url"]
    client.cache_clear()


@lru_cache(maxsize=1)
def client():
    from typesafe_sdk import TypeSafeClient
    if JEV_BASE_URL:  # local server: no TypeSafe account involved
        return TypeSafeClient(api_key="local", base_url=JEV_BASE_URL, model=JEV_MODEL)
    return TypeSafeClient(model=JEV_MODEL)


def _record(stage: str, response, seconds: float) -> None:
    from mira import usage
    usage.record_jev(stage, getattr(getattr(response, "usage", None), "input_tokens", 0) or 0, seconds)


def enabled() -> bool:
    return bool(JEV_BASE_URL) or bool(os.environ.get("TYPESAFE_API_KEY"))


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
    questions = paper_questions(rubric)
    if len(rubric["taxonomy"]) > _B["max_options"]:
        del questions["primary_topic"]  # e.g. Ollama caps choices at 26 options
    r = client().system_one(state, questions)
    latency = time.perf_counter() - t0
    _record("paper relevance", r, latency)
    rel = r.scores["relevance"]
    topic = r.choices.get("primary_topic")
    return {
        "relevance_score": relevance_to_10(rel.score),
        "relevance_level": rel.score,
        "relevance_confidence": rel.confidence,
        "primary_topic": topic.choice if topic else "",
        "primary_topic_confidence": topic.confidence if topic else None,
        "primary_topic_probabilities": dict(topic.probabilities) if topic else {},
        "potential_impact": r.choices["potential_impact"].choice,
        "actionable": r.choices["actionable"].choice,
        "latency_s": latency,
        "input_tokens": r.usage.input_tokens,
    }


def judge_relevance(paper: dict, rubric: dict) -> dict:
    """Only the relevance question: all that prescreen/gate need. Questions are
    answered independently, so the relevance answer is the same as in
    judge_paper; local backends (one request at a time) take half the time."""
    from typesafe_sdk import Score  # noqa: F401 — paper_questions builds it
    state = {"paper": {"title": paper["title"], "abstract": paper["summary"]}}
    t0 = time.perf_counter()
    r = client().system_one(state, {"relevance": paper_questions(rubric)["relevance"]})
    latency = time.perf_counter() - t0
    _record("paper relevance", r, latency)
    rel = r.scores["relevance"]
    return {"relevance_score": relevance_to_10(rel.score), "relevance_level": rel.score,
            "relevance_confidence": rel.confidence, "primary_topic": "",
            "primary_topic_confidence": None, "primary_topic_probabilities": {},
            "potential_impact": "", "actionable": "", "latency_s": latency,
            "input_tokens": r.usage.input_tokens}


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
    _record("credibility", r, latency)
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
    _record("news relevance", r, time.perf_counter() - t0)
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
# Jev relevance-level cutoffs per profile, calibrated against the cached LLM
# baseline on two disjoint samples (scripts/bench_jev.py, seeds 42 and 43).
#   prescreen  papers below it skip the LLM stages; kept >= 99% of LLM-passing
#              and 100% of report-featured papers on both samples
#   gate       keeps ~95% of LLM-passing papers (the LLM still re-checks what
#              passes); memory: 98.8% of featured kept, 73% of calls saved
#   decide     Jev's decision is final (replace mode, live paper alerts):
#              best F1 against the LLM's pass/fail on the random (real-traffic)
#              papers; memory: passes 17.3% like the LLM, 79% precision/recall
#   priority   equivalent of LLM relevance >= 7
#   cred_gate  credibility level (first-page header) equivalent of the
#              profile's credibility threshold
CUTOFFS = {
    "memory-innovation":     {"prescreen": 0.15, "gate": 0.35, "decide": 0.57, "priority": 0.81, "cred_gate": 0.02},
    "cxl-research":          {"prescreen": 0.05, "gate": 0.26, "decide": 0.67, "priority": 1.09, "cred_gate": 0.16},
    "storage-innovation":    {"prescreen": 0.05, "gate": 0.11, "decide": 0.60, "priority": 1.32, "cred_gate": 0.22},
    "optical-interconnects": {"prescreen": None, "gate": 0.07, "decide": 0.31, "priority": 1.68, "cred_gate": 0.08},
}
# Cutoffs for local backends, calibrated the same way on the same samples
# (scripts/bench_jev.py with JEV_BACKEND set). A backend without an entry for
# a profile is not used for that profile.
LOCAL_CUTOFFS: dict = {"nimble": {}, "kev": {}}


def cutoffs_for(profile_id: str) -> dict:
    table = CUTOFFS if BACKEND == "typesafe" else LOCAL_CUTOFFS.get(BACKEND, {})
    return table.get(profile_id) or {}


PRESCREEN_CUTOFFS = {k: v["prescreen"] for k, v in CUTOFFS.items() if v["prescreen"] is not None}
JEV_LEVELS = ("off", "prescreen", "gate", "replace")


def _judge_all(papers: list[dict], rubric: dict, workers: int) -> list:
    from concurrent.futures import ThreadPoolExecutor

    # Local servers answer one request at a time: ask only what screening needs
    # and don't queue more than a couple of requests.
    judge = judge_paper if BACKEND == "typesafe" else judge_relevance
    if BACKEND != "typesafe":
        workers = min(workers, 2)

    def one(p):
        try:
            return judge(p, rubric)
        except Exception as e:  # noqa: BLE001 — fail open
            print(f"  WARNING: Jev judgment failed for {p['id']} — {e}. Keeping.")
            return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, papers))


def prescreen(papers: list[dict], profile: dict, workers: int = 16,
              level: str = "prescreen") -> tuple[list[dict], list[dict]]:
    """Split papers into (kept, screened). level="prescreen" uses the safe
    cutoff (only clearly irrelevant papers are screened); level="gate" uses the
    calibrated relevance gate, so Jev decides relevance and the LLM stages run
    only on what Jev passes. Kept papers go through the LLM stages unchanged.
    Any Jev failure keeps the paper (fails open to the existing path)."""
    cut = cutoffs_for(profile["profile_id"])
    cutoff = cut.get("prescreen") if level == "prescreen" else cut.get("gate")
    if cutoff is None or not papers:
        return papers, []
    judgments = _judge_all(papers, profile_rubric(profile), workers)
    kept, screened = [], []
    for p, j in zip(papers, judgments):
        if j is not None and j["relevance_level"] < cutoff:
            p["jev_prescreen"] = {k: j[k] for k in ("relevance_level", "relevance_confidence",
                                                    "primary_topic")}
            screened.append(p)
        else:
            kept.append(p)
    return kept, screened


def replace_classification(papers: list[dict], profile: dict, workers: int = 16) -> list[dict]:
    """Jev instead of the per-paper LLM stages (affiliation + classification).

    Relevance, topic, impact and actionable come from one Jev call per paper;
    credibility from the first-page header, only for papers past the relevance
    gate (call after extract_first_pages on those). Scores are written on the
    LLM's 1-10 scale, snapped so the profile's own thresholds reproduce Jev's
    calibrated gates. Not available from Jev: key findings, affiliation lists,
    reasoning text (left empty)."""
    cut = cutoffs_for(profile["profile_id"])
    th = profile["thresholds"]
    rmin, cmin = th["relevance_score_min"], th["credibility_tier_min"]
    judgments = _judge_all(papers, profile_rubric(profile), workers)
    out = []
    for p, j in zip(papers, judgments):
        if j is None:
            continue  # like a failed LLM stage: the completeness gate drops it
        mapped = j["relevance_score"]
        passed = j["relevance_level"] >= cut["decide"]
        p.update({
            "relevance_score": max(mapped, rmin) if passed else min(mapped, rmin - 1),
            "primary_topic": j["primary_topic"], "secondary_topics": [],
            "potential_impact": j["potential_impact"], "actionable": j["actionable"],
            "key_findings": "", "affiliations": [], "author_affiliations": {},
            "credibility_reasoning": "", "jev": {"relevance_level": j["relevance_level"]},
        })
        out.append(p)
    return out


def judge_credibility_for(papers: list[dict], profile: dict, workers: int = 16) -> None:
    """Credibility tier from the first-page header, snapped to the profile's
    threshold at Jev's calibrated credibility gate (replace mode)."""
    from concurrent.futures import ThreadPoolExecutor

    cut = cutoffs_for(profile["profile_id"])
    cmin = profile["thresholds"]["credibility_tier_min"]
    focus = profile["topic"]["focus"]

    def one(p):
        try:
            return judge_credibility(focus, first_page_text=p.get("first_page_text", ""))
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: Jev credibility failed for {p['id']} — {e}")
            return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        creds = list(pool.map(one, papers))
    for p, c in zip(papers, creds):
        if c is None:
            p["credibility_tier"] = cmin  # no evidence either way: don't block on credibility
            continue
        ok = c["credibility_level"] >= cut["cred_gate"]
        p["credibility_tier"] = max(c["credibility_tier"], cmin) if ok else min(c["credibility_tier"], cmin - 1)

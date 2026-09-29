"""LLM feasibility judge (spec §3 Stage 4).

The future pipeline stages require actually running the experiments, so
tractability affects ranking. Classification, not synthesis — a small model
suffices. Failures degrade to tier 'unrated' (neutral multiplier, spec §5).
"""
from __future__ import annotations

from mira.config import llm_call, shared_llm_model
from mira.hypothesis.gaps import GapCandidate, gap_facts

# Resolved from the shared `llm_models` block so the judge follows the same
# model policy as every other stage; the literal is the shipped fallback.
FEASIBILITY_MODEL = shared_llm_model("feasibility", "anthropic/claude-haiku-4-5")

TIER_MULTIPLIERS = {"T1": 1.0, "T2": 0.9, "T3": 0.6, "T4": 0.3, "unrated": 1.0}

FEASIBILITY_SYSTEM = (
    "You are a research-feasibility assessor for a small team with no fab "
    "access and no proprietary-hardware partnerships. Classify what validating "
    "a hypothesis in the given gap would MINIMALLY require. Reply with exactly "
    "two lines:\n"
    "TIER: <T1|T2|T3|T4>\n"
    "<one-sentence justification>\n"
    "T1 = simulation, modeling, or public-data analysis. "
    "T2 = commodity-hardware benchmarking. "
    "T3 = prototype hardware or restricted data. "
    "T4 = fab access or a proprietary process."
)


def build_feasibility_prompt(cand: GapCandidate) -> str:
    return (
        f"{gap_facts(cand)}\n\n"
        f"What would validating a hypothesis bridging '{cand.topic_a}' and "
        f"'{cand.topic_c}' minimally require?"
    )


def parse_tier(reply: str) -> tuple[str, str]:
    first, _, rest = reply.strip().partition("\n")
    head = first.strip().upper()
    if head.startswith("TIER:"):
        word = head.removeprefix("TIER:").strip()
        if word in ("T1", "T2", "T3", "T4"):
            return word, rest.strip()
    return "unrated", reply.strip()


def judge_feasibility(client, cand: GapCandidate) -> tuple[str, str]:
    try:
        reply = llm_call(client, FEASIBILITY_MODEL, FEASIBILITY_SYSTEM,
                         build_feasibility_prompt(cand), temperature=0.0)
    except RuntimeError as exc:
        return "unrated", f"judge unavailable ({exc})"
    return parse_tier(reply)

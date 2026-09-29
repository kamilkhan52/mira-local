"""LLM synthesis of hypotheses from verified gaps.

The algorithm found the gap; the LLM makes the creative leap — grounded in
LightRAG-retrieved evidence for both sides (spec §1, §4). Retrieval uses
`only_need_context` so LightRAG returns its retrieved chunks/entities without
generating an answer itself.
"""
from __future__ import annotations

import os

import requests

from mira.config import llm_call
from .gaps import GapCandidate, gap_facts

SYNTHESIS_MODEL = "anthropic/claude-sonnet-4.6"
_DEFAULT_BASE_URL = "http://localhost:9621"

SYNTHESIS_SYSTEM = (
    "You are a memory-technology research strategist. You draft precise, "
    "falsifiable research hypotheses grounded ONLY in the evidence provided. "
    "Cite papers by title and arXiv URL exactly as they appear in the evidence. "
    "Never invent citations."
)

EXHAUSTIVE_SYNTHESIS_SYSTEM = (
    "You are MIRA, a cross-domain research strategist. Draft precise, "
    "falsifiable hypotheses grounded only in the supplied exhaustive evidence. "
    "Preserve numerical results and contradictions. Cite only supplied paper "
    "titles and paths. Return JSON only with one hypothesis for every requested "
    "candidate pair, using the required four markdown sections."
)

CRITIC_SYSTEM = (
    "You are a rigorous but fair reviewer of a proposed research hypothesis. "
    "Judge it ONLY against the evidence provided. Your reply MUST start with "
    "exactly one line: 'VERDICT: ACCEPT', 'VERDICT: STRENGTHEN', or "
    "'VERDICT: KILL'.\n"
    "- ACCEPT: every claim is consistent with the evidence and every citation "
    "appears in it. Follow with one sentence saying so.\n"
    "- STRENGTHEN: the core idea survives but specific claims overreach the "
    "evidence or citations are imprecise. Follow with ONLY the full revised "
    "hypothesis text, starting directly at '**Claim**' with the same section "
    "structure — no analysis, preamble, or extra headers.\n"
    "- KILL: reserve for decisive defects only — a claim the provided evidence "
    "contradicts, a citation that does not appear in the evidence, or external "
    "work in the evidence showing the gap is already occupied. The hypothesis "
    "is speculative by design: general skepticism, missing experiments, or "
    "'needs more validation' are NOT kill reasons. Follow with the decisive "
    "objection, quoting the contradicting evidence."
)


def retrieve_context(query: str, base_url: str | None = None, top_k: int = 10) -> str | None:
    resolved = base_url or os.environ.get("LIGHTRAG_BASE_URL") or _DEFAULT_BASE_URL
    # The combined instance runs with LIGHTRAG_API_KEY set and answers an
    # unauthenticated /query with {"detail": "API Key required"}. That surfaces
    # here only as a RequestException, so a missing header degrades grounding to
    # graph metadata SILENTLY rather than failing the run. Instances without a
    # key configured keep working: the header is simply omitted.
    api_key = os.environ.get("LIGHTRAG_API_KEY", "")
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        resp = requests.post(
            f"{resolved}/query",
            json={"query": query, "mode": "hybrid",
                  "only_need_context": True, "top_k": top_k},
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") if isinstance(data, dict) else None
    except requests.RequestException:
        return None


def _evidence_block(label: str, context: str | None) -> str:
    if context:
        return f"### Evidence for {label}\n{context}"
    return (
        f"### Evidence for {label}\n"
        "(retrieval unavailable — ground statements about this side in the gap "
        "facts only, and say so explicitly)"
    )


def build_synthesis_prompt(
    cand: GapCandidate, context_a: str | None, context_b: str | None
) -> str:
    novelty = (
        f"External literature check: {cand.novelty_hits} recent papers near this "
        f"pair ({cand.novelty_label})."
        if cand.novelty_hits is not None
        else "External literature check: unverified (venue corpus could not verify this pair)."
    )
    if cand.external_titles:
        novelty += " Closest external work: " + "; ".join(cand.external_titles)
    return f"""## Verified gap (structural facts about the corpus)
{gap_facts(cand)}

{novelty}

{_evidence_block(cand.topic_a, context_a)}

{_evidence_block(cand.topic_c, context_b)}

## Task
Draft ONE research hypothesis bridging '{cand.topic_a}' and '{cand.topic_c}'.
Output exactly these four markdown sections:

**Claim** — one falsifiable sentence.
**Supporting evidence** — bullet points, each citing a specific paper (title + arXiv URL) from the evidence above.
**Why plausibly unexplored** — grounded in the gap facts and the external literature check.
**Suggested validation** — one concrete experiment or analysis direction.
"""


def synthesize(client, cand: GapCandidate, context_a: str | None, context_b: str | None) -> str:
    return llm_call(client, SYNTHESIS_MODEL, SYNTHESIS_SYSTEM,
                    build_synthesis_prompt(cand, context_a, context_b))


def critique(client, cand: GapCandidate, hypothesis_text: str,
             context_a: str | None, context_b: str | None) -> str:
    user = f"""## Hypothesis under review
{hypothesis_text}

## Verified gap facts
{gap_facts(cand)}

{_evidence_block(cand.topic_a, context_a)}

{_evidence_block(cand.topic_c, context_b)}

Check every claim and citation against the evidence above — unsupported claims,
miscitations, contradicting evidence. Then deliver your verdict as instructed."""
    return llm_call(client, SYNTHESIS_MODEL, CRITIC_SYSTEM, user)


def apply_critic(client, cand: GapCandidate, text: str,
                 context_a: str | None, context_b: str | None) -> tuple[str, str]:
    """Critique a synthesized hypothesis and apply the verdict.

    Returns (verdict, text): 'strengthen' returns the revised text; 'kill'
    labels the candidate and records the objection (text unchanged, caller
    decides to drop); 'unknown' leaves everything as-is. Raises RuntimeError
    when the critic LLM call fails.
    """
    verdict, body = parse_critic_verdict(
        critique(client, cand, text, context_a, context_b))
    if verdict == "strengthen" and body:
        # Defensive: models sometimes prefix reviewer analysis despite the
        # prompt; the dossier must carry only the revised hypothesis.
        idx = body.find("**Claim**")
        return verdict, body[idx:] if idx > 0 else body
    if verdict == "kill":
        cand.novelty_label += " · killed by critic"
        cand.critic_objection = body
    return verdict, text


def parse_critic_verdict(reply: str) -> tuple[str, str]:
    first, _, rest = reply.strip().partition("\n")
    verdict = first.strip().upper()
    if verdict.startswith("VERDICT:"):
        word = verdict.removeprefix("VERDICT:").strip()
        if word in ("ACCEPT", "STRENGTHEN", "KILL"):
            return word.lower(), rest.strip()
    return "unknown", reply.strip()

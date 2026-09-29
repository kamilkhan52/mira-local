"""Hypothesis generation grounded in one exhaustive evidence compilation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from datetime import date
from itertools import combinations
from typing import Any, Callable, Literal

import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mira.exhaustive.compiler import (
    MAX_SYNTHESIS_ATTEMPTS,
    CitationRecord,
    CompiledEvidence,
    ModelCall,
    SYNTHESIS_MODEL,
    _retry_prompt,
    is_provider_failure,
)
from mira.exhaustive.costs import CostLedger
from mira.exhaustive.engine import PreparedResearch
from mira.exhaustive.types import DOMAIN_NAMES

from .corpus import Corpus, match_anchor_topics
from .dossier import render_dossier
from .gaps import (
    GapCandidate,
    gap_facts,
    is_alias_pair,
    mine_gaps,
    score_candidates,
)
from .synthesis import EXHAUSTIVE_SYNTHESIS_SYSTEM


@dataclass(frozen=True, slots=True)
class ExhaustiveHypothesisOutput:
    markdown: str
    candidates_evaluated: int
    hypotheses_generated: int
    candidate_pairs: tuple[tuple[str, str], ...]
    citations: tuple[CitationRecord, ...]


class _HypothesisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic_a: str
    topic_c: str
    text: str


class _SynthesisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypotheses: list[_HypothesisPayload]


class _ReviewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic_a: str
    topic_c: str
    verdict: Literal["accept", "strengthen", "kill"]
    text: str
    objections: list[str] = Field(default_factory=list)


class _CriticPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviews: list[_ReviewPayload]


CRITIC_SYSTEM = (
    "You are a rigorous but fair reviewer of proposed research hypotheses. "
    "Judge each draft ONLY against the supplied exhaustive evidence. Return "
    "ONLY a JSON object with one field, \"reviews\": a list with exactly one "
    "object per supplied candidate, each having exactly these fields:\n"
    '  "topic_a": str — copied verbatim from the candidate.\n'
    '  "topic_c": str — copied verbatim from the candidate.\n'
    '  "verdict": one of "accept", "strengthen", "kill".\n'
    '  "text": str — the hypothesis to publish, in the same four markdown '
    "sections. For \"strengthen\" return your revised version; for "
    "\"accept\" and \"kill\" return the draft unchanged.\n"
    '  "objections": list[str] — every unsupported claim, miscitation, or '
    "contradicting result you found. Required and non-empty for "
    "\"kill\"; may be empty for \"accept\"."
)


@dataclass(frozen=True, slots=True)
class HypothesisReview:
    verdict: str
    text: str
    objections: tuple[str, ...]


def _record_payload(record: Any) -> dict:
    return {
        "claim": record.claim,
        "measurements": list(record.measurements),
        "mechanisms": list(record.mechanisms),
        "assumptions": list(record.assumptions),
        "limitations": list(record.limitations),
        "contradictions": list(record.contradictions),
        "citations": [{
            "chunk_id": citation.chunk_id,
            "source_chunk_id": citation.source_chunk_id,
            "file_path": citation.file_path,
            "title": citation.title,
            "domains": list(citation.domains),
        } for citation in record.citations],
        "relevance": record.relevance,
        "contributing_chunk_ids": list(record.contributing_chunk_ids),
    }


class ExhaustiveHypothesisRunner:
    def __init__(
        self,
        model_call: ModelCall,
        ledger: CostLedger,
        *,
        model: str = SYNTHESIS_MODEL,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model_call = model_call
        self.ledger = ledger
        self.model = model
        self.sleep = sleep

    def __call__(
        self,
        prepared: PreparedResearch,
        compiled: CompiledEvidence,
        request: dict,
        emit: Callable[[dict], None],
    ) -> ExhaustiveHypothesisOutput:
        if not prepared.bundle.exhaustive:
            raise ValueError(
                "hypothesis generation requires exhaustive evidence"
            )
        if set(prepared.snapshots) != set(DOMAIN_NAMES):
            raise ValueError(
                "hypothesis generation requires all three domain snapshots"
            )
        profiles = tuple(request.get("profiles") or ())
        topics = tuple(request.get("topics") or ())
        if not profiles or not topics:
            raise ValueError("topics and profiles are required")

        graph = nx.compose_all([
            prepared.snapshots[domain].graph for domain in DOMAIN_NAMES
        ])
        corpus = Corpus.build(graph, profiles)
        if not corpus.papers:
            raise ValueError(
                "selected profiles contain no papers in the domain graphs"
            )
        anchors = list(dict.fromkeys(
            anchor
            for topic in topics
            for anchor in match_anchor_topics(topic, corpus.topic_papers)
        ))
        if not anchors:
            raise ValueError("selected topics do not match graph topics")
        expanded = list(dict.fromkeys(
            [*anchors]
            + sorted({
                related
                for anchor in anchors
                for related in corpus.related_topics(anchor)
                if related in corpus.topic_papers
            })
        ))
        emit({
            "name": "hypothesis_candidates_started",
            "anchors": anchors,
        })
        explicit_rejections = []
        for left, right in combinations(anchors, 2):
            if is_alias_pair(left, right):
                explicit_rejections.append(
                    f"{left} × {right}: rejected as topic aliases"
                )
            elif corpus.topic_papers[left] & corpus.topic_papers[right]:
                explicit_rejections.append(
                    f"{left} × {right}: already connected by a paper"
                )
        candidates = mine_gaps(
            corpus,
            expanded,
            explicit_anchor_topics=anchors,
        )
        citation_domains: dict[str, set[str]] = {}
        for citation in compiled.citations:
            if citation.title:
                citation_domains.setdefault(citation.title, set()).update(
                    citation.domains
                )
        evidence_papers = set(citation_domains)
        supported = []
        for candidate in candidates:
            left_papers = (
                corpus.topic_papers[candidate.topic_a] & evidence_papers
            )
            right_papers = (
                corpus.topic_papers[candidate.topic_c] & evidence_papers
            )
            candidate.domains_a = tuple(
                domain
                for domain in DOMAIN_NAMES
                if any(
                    domain in citation_domains[paper]
                    for paper in left_papers
                )
            )
            candidate.domains_c = tuple(
                domain
                for domain in DOMAIN_NAMES
                if any(
                    domain in citation_domains[paper]
                    for paper in right_papers
                )
            )
            if (
                left_papers
                and right_papers
                and candidate.domains_a
                and candidate.domains_c
            ):
                supported.append(candidate)
            elif candidate.explicit_anchor_pair:
                explicit_rejections.append(
                    f"{candidate.topic_a} × {candidate.topic_c}: "
                    "insufficient compiled evidence for both sides"
                )
        candidates = supported
        vector_index = self._vector_index(prepared)
        semantic = {
            (candidate.topic_a, candidate.topic_c):
                self._side_similarity(
                    vector_index,
                    corpus.topic_papers.get(candidate.topic_a, ()),
                    corpus.topic_papers.get(candidate.topic_c, ()),
                )
            for candidate in candidates
        }
        ranked = score_candidates(candidates, semantic)
        for candidate in ranked:
            candidate.novelty_label = "unverified"
        emit({
            "name": "hypothesis_candidates_completed",
            "candidates_evaluated": len(ranked),
        })

        requested_count = int(request.get("max_hypotheses", 5))
        if requested_count <= 0:
            raise ValueError("max_hypotheses must be positive")
        selected = ranked[:requested_count]
        hypotheses = self._synthesize(
            selected, compiled, "hypothesis_synthesis", emit
        )
        reviews: dict[tuple[str, str], HypothesisReview] = {}
        if request.get("critic") and hypotheses:
            reviews = self._critic(
                selected, hypotheses, compiled, emit
            )
        for candidate in selected:
            review = reviews.get((candidate.topic_a, candidate.topic_c))
            if review is not None and review.verdict == "kill":
                candidate.novelty_label += " · killed by critic"
                candidate.critic_objection = "; ".join(review.objections)

        rendered = []
        for candidate in selected:
            pair = (candidate.topic_a, candidate.topic_c)
            review = reviews.get(pair)
            text = review.text if review is not None else hypotheses.get(
                pair, ""
            )
            if not text:
                continue
            rendered.append({
                "candidate": candidate,
                "text": text,
                "critic_verdict": review.verdict if review else None,
                "critic_body": (
                    "\n".join(f"- {item}" for item in review.objections)
                    if review and review.objections else None
                ),
            })
        coverage = " · ".join(
            f"{domain}: {prepared.bundle.nodes_scanned[domain]} nodes / "
            f"{prepared.bundle.edges_scanned[domain]} edges"
            for domain in DOMAIN_NAMES
        )
        markdown = render_dossier(
            {
                "topic": ", ".join(topics),
                "profiles": profiles,
                "run_date": date.today().isoformat(),
                "graph_stats": coverage,
                "flags": "--exhaustive"
                + (" --critic" if request.get("critic") else ""),
                "anchors": anchors,
                "degraded": (
                    ["external novelty check not run in queued exhaustive mode"]
                    if not request.get("no_external")
                    else ["--no-external — novelty unverified"]
                ),
            },
            rendered,
            ranked,
        )
        if explicit_rejections:
            markdown += (
                "\n\n## Explicit pair evaluations\n\n"
                + "\n".join(
                    f"- {reason}" for reason in explicit_rejections
                )
                + "\n"
            )
        return ExhaustiveHypothesisOutput(
            markdown=markdown,
            candidates_evaluated=len(ranked),
            hypotheses_generated=len(rendered),
            candidate_pairs=tuple(
                (candidate.topic_a, candidate.topic_c)
                for candidate in ranked
            ),
            citations=compiled.citations,
        )

    def _synthesize(
        self,
        selected: list[GapCandidate],
        compiled: CompiledEvidence,
        stage: str,
        emit: Callable[[dict], None],
    ) -> dict[tuple[str, str], str]:
        if not selected:
            return {}
        emit({
            "name": f"{stage}_started",
            "candidates": len(selected),
        })
        user = json.dumps({
            "question": compiled.query,
            "candidates": [{
                "topic_a": candidate.topic_a,
                "topic_c": candidate.topic_c,
                "gap_facts": gap_facts(candidate),
                "combined_score": candidate.combined_score,
                "domains_a": list(candidate.domains_a),
                "domains_c": list(candidate.domains_c),
                "cross_domain": bool(
                    set(candidate.domains_a) - set(candidate.domains_c)
                    or set(candidate.domains_c) - set(candidate.domains_a)
                ),
            } for candidate in selected],
            "evidence": [
                _record_payload(record) for record in compiled.records
            ],
            "required_sections": [
                "Claim",
                "Supporting evidence",
                "Why plausibly unexplored",
                "Suggested validation",
            ],
        }, ensure_ascii=False)
        expected = {
            (candidate.topic_a, candidate.topic_c) for candidate in selected
        }

        def parse(text: str) -> dict[tuple[str, str], str]:
            payload = _SynthesisPayload.model_validate_json(text)
            found = {
                (item.topic_a, item.topic_c)
                for item in payload.hypotheses
            }
            if found != expected or len(found) != len(payload.hypotheses):
                raise ValueError(
                    "synthesis did not return every requested pair once"
                )
            return {
                (item.topic_a, item.topic_c): item.text.strip()
                for item in payload.hypotheses
                if item.text.strip()
            }

        hypotheses = self._validated_call(
            system=EXHAUSTIVE_SYNTHESIS_SYSTEM,
            user=user,
            stage=stage,
            parse=parse,
            emit=emit,
        )
        emit({"name": f"{stage}_completed"})
        return hypotheses

    def _validated_call(
        self,
        *,
        system: str,
        user: str,
        stage: str,
        parse: Callable[[str], Any],
        emit: Callable[[dict], None],
    ) -> Any:
        """Call the synthesis model until its JSON validates.

        A retry states what was wrong with the previous attempt -- an
        identical request is pure spend -- and a provider rejection (auth,
        quota, rate limit) is raised at once instead of tried twice more.
        """
        last_error: Exception | None = None
        for attempt in range(MAX_SYNTHESIS_ATTEMPTS):
            try:
                response = self.model_call(
                    model=self.model,
                    system=system,
                    user=user if last_error is None else _retry_prompt(
                        user, last_error
                    ),
                    stage=stage,
                    max_output_tokens=12_000,
                )
                self.ledger.record_usage(replace(
                    response.usage, stage=stage, model=self.model
                ))
                cost = self.ledger.summary()
                emit({
                    "name": "cost_actual",
                    "actual_cost_usd": cost.actual_cost_usd,
                    "cost_status": cost.cost_status,
                    "by_stage": dict(cost.by_stage),
                })
                return parse(response.text)
            except Exception as exc:
                if is_provider_failure(exc):
                    raise RuntimeError(
                        f"{stage} failed: provider rejected the call: {exc}"
                    ) from exc
                last_error = exc
                if attempt + 1 < MAX_SYNTHESIS_ATTEMPTS:
                    self.sleep(2.0 ** attempt)
        raise RuntimeError(
            f"{stage} failed after {MAX_SYNTHESIS_ATTEMPTS} attempts: "
            f"{last_error}"
        )

    def _critic(
        self,
        selected: list[GapCandidate],
        hypotheses: dict[tuple[str, str], str],
        compiled: CompiledEvidence,
        emit: Callable[[dict], None],
    ) -> dict[tuple[str, str], HypothesisReview]:
        """Review every drafted hypothesis on the configured synthesis model.

        The verdict and the objections come from the model. Reporting a fixed
        "reviewed" label would tell a reader a critic ran without telling them
        what it found -- which is the only part worth paying for.
        """
        selected = [
            candidate
            for candidate in selected
            if (candidate.topic_a, candidate.topic_c) in hypotheses
        ]
        if not selected:
            return {}
        stage = "hypothesis_critic"
        emit({"name": f"{stage}_started", "candidates": len(selected)})
        user = json.dumps({
            "question": compiled.query,
            "candidates": [{
                "topic_a": candidate.topic_a,
                "topic_c": candidate.topic_c,
                "gap_facts": gap_facts(candidate),
                "draft": hypotheses[
                    (candidate.topic_a, candidate.topic_c)
                ],
            } for candidate in selected],
            "evidence": [
                _record_payload(record) for record in compiled.records
            ],
            "required_sections": [
                "Claim",
                "Supporting evidence",
                "Why plausibly unexplored",
                "Suggested validation",
            ],
        }, ensure_ascii=False)
        expected = {
            (candidate.topic_a, candidate.topic_c) for candidate in selected
        }

        def parse(text: str) -> dict[tuple[str, str], HypothesisReview]:
            payload = _CriticPayload.model_validate_json(text)
            found = {
                (item.topic_a, item.topic_c) for item in payload.reviews
            }
            if found != expected or len(found) != len(payload.reviews):
                raise ValueError(
                    "critic did not review every drafted pair once"
                )
            missing = [
                item for item in payload.reviews
                if item.verdict == "kill" and not any(
                    objection.strip() for objection in item.objections
                )
            ]
            if missing:
                raise ValueError("a killed hypothesis carried no objection")
            return {
                (item.topic_a, item.topic_c): HypothesisReview(
                    verdict=item.verdict,
                    text=item.text.strip() or hypotheses[
                        (item.topic_a, item.topic_c)
                    ],
                    objections=tuple(
                        objection.strip()
                        for objection in item.objections
                        if objection.strip()
                    ),
                )
                for item in payload.reviews
            }

        reviews = self._validated_call(
            system=CRITIC_SYSTEM,
            user=user,
            stage=stage,
            parse=parse,
            emit=emit,
        )
        emit({"name": f"{stage}_completed"})
        return reviews

    def _side_similarity(
        self,
        vector_index: dict[str, list[np.ndarray]],
        papers_a: Any,
        papers_b: Any,
    ) -> float:
        vectors_a = [
            vector
            for paper in papers_a
            for vector in vector_index.get(paper, ())
        ]
        vectors_b = [
            vector
            for paper in papers_b
            for vector in vector_index.get(paper, ())
        ]
        if not vectors_a or not vectors_b:
            return 0.0
        mean_a = np.mean(vectors_a, axis=0)
        mean_b = np.mean(vectors_b, axis=0)
        denominator = float(np.linalg.norm(mean_a) * np.linalg.norm(mean_b))
        if denominator == 0:
            return 0.0
        return max(0.0, min(1.0, float(mean_a @ mean_b) / denominator))

    def _vector_index(
        self,
        prepared: PreparedResearch,
    ) -> dict[str, list[np.ndarray]]:
        vectors: dict[str, list[np.ndarray]] = {}
        for domain in DOMAIN_NAMES:
            snapshot = prepared.snapshots[domain]
            for index, name in enumerate(snapshot.entity_names):
                vectors.setdefault(name, []).append(
                    snapshot.entity_matrix[index]
                )
        return vectors

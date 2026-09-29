"""Shared orchestration for exhaustive chat and hypothesis research."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable

from .compiler import (
    MAX_SYNTHESIS_ATTEMPTS,
    CitationRecord,
    CompiledEvidence,
    EvidenceCompiler,
)
from .costs import CostEstimate, CostEstimator, CostLedger, StageEstimate
from .evidence import EvidenceCollector
from .scoring import QueryScorer
from .snapshots import GraphSnapshotStore
from .types import (
    CostSummary,
    DOMAIN_NAMES,
    ProgressEvent,
    ResearchCoverageSummary,
    ResearchEvidenceBundle,
    summarize_coverage,
)


class ExhaustiveResearchError(RuntimeError):
    """A request could not prove complete exhaustive coverage."""


class ExhaustiveCostLimitError(ExhaustiveResearchError):
    """The preflight estimate exceeds the configured spend ceiling."""


@dataclass(frozen=True, slots=True)
class PreparedResearch:
    query: str
    history: tuple[str, ...]
    bundle: ResearchEvidenceBundle
    estimated_cost: CostEstimate
    snapshots: Any


@dataclass(frozen=True, slots=True)
class ChatResearchResult:
    markdown: str
    citations: tuple[CitationRecord, ...]
    contributing_chunk_ids: tuple[str, ...]
    coverage: ResearchCoverageSummary
    estimated_cost: CostEstimate
    cost: CostSummary


@dataclass(frozen=True, slots=True)
class HypothesisResearchResult:
    result: Any
    # The deduplicated citation list the UI renders -- not the compiled
    # corpus. The full evidence is multiple megabytes and would be copied
    # through the job pipe into every job record and every GET response.
    citations: tuple[CitationRecord, ...]
    coverage: ResearchCoverageSummary
    estimated_cost: CostEstimate
    cost: CostSummary


class _ProgressPublisher:
    def __init__(self, callback: Callable[[ProgressEvent], None]):
        self.callback = callback
        self.sequence = 0

    def emit(self, event: str | dict[str, Any], **data: Any) -> None:
        if isinstance(event, dict):
            payload = dict(event)
            name = str(payload.pop("name"))
            data = payload
        else:
            name = event
        self.sequence += 1
        self.callback(ProgressEvent(
            sequence=self.sequence,
            name=name,
            data=MappingProxyType(dict(data)),
        ))


class ExhaustiveResearchEngine:
    def __init__(
        self,
        *,
        store: GraphSnapshotStore,
        scorer: QueryScorer,
        collector: EvidenceCollector,
        compiler: EvidenceCompiler,
        estimator: CostEstimator,
        ledger: CostLedger,
        hypothesis_runner: Callable[
            [
                PreparedResearch,
                CompiledEvidence,
                Any,
                Callable[[dict], None],
            ],
            Any,
        ] | None = None,
        cost_fetcher: Callable[[str], dict] | None = None,
        max_estimated_cost_usd: float = 25.0,
    ):
        if max_estimated_cost_usd <= 0:
            raise ValueError("max_estimated_cost_usd must be positive")
        self.store = store
        self.scorer = scorer
        self.collector = collector
        self.compiler = compiler
        self.estimator = estimator
        self.ledger = ledger
        self.hypothesis_runner = hypothesis_runner
        self.cost_fetcher = cost_fetcher
        self.max_estimated_cost_usd = max_estimated_cost_usd

    def prepare(
        self,
        query: str,
        history: Iterable[str] = (),
        emit: Callable[[ProgressEvent], None] = lambda _event: None,
    ) -> PreparedResearch:
        publisher = _ProgressPublisher(emit)
        return self._prepare(query, tuple(history), publisher)

    def run_chat(
        self,
        research: str | PreparedResearch,
        history: Iterable[str] = (),
        emit: Callable[[ProgressEvent], None] = lambda _event: None,
    ) -> ChatResearchResult:
        publisher = _ProgressPublisher(emit)
        usage_checkpoint = self.ledger.checkpoint()
        try:
            prepared = (
                research
                if isinstance(research, PreparedResearch)
                else self._prepare(research, tuple(history), publisher)
            )
            self._enforce_cost_ceiling(prepared, publisher)
            compiled = self.compiler.compile(
                prepared.bundle, publisher.emit
            )
            answer = self.compiler.synthesize_chat(
                compiled,
                publisher.emit,
                history=prepared.history,
            )
            cost = self._finalize_and_emit_cost(
                usage_checkpoint, publisher
            )
            return ChatResearchResult(
                markdown=answer.markdown,
                citations=answer.citations,
                contributing_chunk_ids=answer.contributing_chunk_ids,
                coverage=summarize_coverage(prepared.bundle),
                estimated_cost=prepared.estimated_cost,
                cost=cost,
            )
        except ExhaustiveResearchError:
            self._finalize_failure_cost(usage_checkpoint, publisher)
            raise
        except Exception as exc:
            self._finalize_failure_cost(usage_checkpoint, publisher)
            raise ExhaustiveResearchError(
                f"exhaustive chat failed: {exc}"
            ) from exc

    def run_hypotheses(
        self,
        research: str | PreparedResearch,
        request: Any,
        emit: Callable[[ProgressEvent], None] = lambda _event: None,
    ) -> HypothesisResearchResult:
        publisher = _ProgressPublisher(emit)
        usage_checkpoint = self.ledger.checkpoint()
        try:
            prepared = (
                research
                if isinstance(research, PreparedResearch)
                else self._prepare(
                    research,
                    (),
                    publisher,
                    estimate_kind="hypotheses",
                    estimate_request=request,
                )
            )
            self._enforce_cost_ceiling(prepared, publisher)
            compiled = self.compiler.compile(
                prepared.bundle, publisher.emit
            )
            if self.hypothesis_runner is None:
                raise ExhaustiveResearchError(
                    "no exhaustive hypothesis runner is configured"
                )
            result = self.hypothesis_runner(
                prepared, compiled, request, publisher.emit
            )
            cost = self._finalize_and_emit_cost(
                usage_checkpoint, publisher
            )
            return HypothesisResearchResult(
                result=result,
                citations=compiled.citations,
                coverage=summarize_coverage(prepared.bundle),
                estimated_cost=prepared.estimated_cost,
                cost=cost,
            )
        except ExhaustiveResearchError:
            self._finalize_failure_cost(usage_checkpoint, publisher)
            raise
        except Exception as exc:
            self._finalize_failure_cost(usage_checkpoint, publisher)
            raise ExhaustiveResearchError(
                f"exhaustive hypothesis generation failed: {exc}"
            ) from exc

    def _enforce_cost_ceiling(
        self,
        prepared: PreparedResearch,
        publisher: _ProgressPublisher,
    ) -> None:
        """Refuse a run whose estimate exceeds the ceiling, before spending.

        `prepare` is model-free apart from the query embedding, so this runs
        with the estimate known and every billable stage still ahead.
        """
        estimate = prepared.estimated_cost.total_with_reserve_usd
        if estimate <= self.max_estimated_cost_usd:
            return
        message = (
            f"estimated cost ${estimate:,.2f} exceeds the configured limit "
            f"${self.max_estimated_cost_usd:,.2f} "
            "(MIRA_MAX_ESTIMATED_COST_USD); no model call was made"
        )
        publisher.emit(
            "cost_limit_exceeded",
            estimated_cost_usd=estimate,
            max_estimated_cost_usd=self.max_estimated_cost_usd,
        )
        raise ExhaustiveCostLimitError(message)

    def _prepare(
        self,
        query: str,
        history: tuple[str, ...],
        publisher: _ProgressPublisher,
        *,
        estimate_kind: str = "chat",
        estimate_request: Any = None,
    ) -> PreparedResearch:
        try:
            publisher.emit("snapshot_pin_started")
            snapshots = self.store.pin_all()
            publisher.emit("snapshot_pin_completed")
            scoring = self.scorer.score_all(
                self._search_statement(query, history),
                snapshots,
            )
            for domain in DOMAIN_NAMES:
                publisher.emit(
                    f"domain_scanned:{domain}",
                    domain=domain,
                    nodes=scoring.nodes_by_domain[domain],
                    edges=scoring.edges_by_domain[domain],
                )
            bundle = self.collector.collect(query, scoring, snapshots)
            publisher.emit(
                "evidence_collected",
                papers=len(bundle.paper_names),
                chunks=len(bundle.chunks),
            )
            incurred = self.ledger.summary()
            if incurred.by_stage:
                publisher.emit(
                    "cost_actual",
                    actual_cost_usd=incurred.actual_cost_usd,
                    cost_status=incurred.cost_status,
                    by_stage=dict(incurred.by_stage),
                )
            estimate = self._estimate(
                bundle,
                kind=estimate_kind,
                request=estimate_request,
                history=history,
            )
            publisher.emit(
                "cost_estimated",
                subtotal_usd=estimate.subtotal_usd,
                total_with_reserve_usd=estimate.total_with_reserve_usd,
                by_stage=dict(estimate.by_stage),
                caps=dict(estimate.caps),
            )
            return PreparedResearch(
                query=query,
                history=history,
                bundle=bundle,
                estimated_cost=estimate,
                snapshots=snapshots,
            )
        except Exception as exc:
            raise ExhaustiveResearchError(
                f"exhaustive research prepare failed: {exc}"
            ) from exc

    def _estimate(
        self,
        bundle: ResearchEvidenceBundle,
        *,
        kind: str = "chat",
        request: Any = None,
        history: tuple[str, ...] = (),
    ) -> CostEstimate:
        chunk_tokens = sum(
            self.compiler.token_counter(chunk.content)
            for chunk in bundle.chunks
        )
        batch_count = self.compiler.batch_count(bundle.chunks)
        map_input = (
            chunk_tokens
            + batch_count * self.compiler.prompt_overhead_tokens
        )
        # Each map call permits 16k output tokens. Budget against that ceiling
        # so the preflight figure covers any valid provider response.
        #
        # This is an upper bound for the paths modelled here, not a guarantee:
        # a run can still exceed it -- provider prices move between the catalog
        # fetch and the call, and reasoning tokens are billed but not modelled.
        map_output = max(
            256 * len(bundle.chunks),
            int(map_input * 0.15),
            batch_count * 16_000,
        )
        # The map stage is the dominant cost and every batch may be retried up
        # to the validation ceiling, so budget the retries rather than one
        # attempt -- otherwise the estimate understates it threefold.
        map_attempts = self.compiler.max_validation_attempts
        needs_reduction = map_output > self.compiler.compiled_target_tokens
        # Each successful reduction round must strictly lower token volume.
        # Budget the configured round ceiling at the full map-output volume
        # for both prompt and completion, including per-group prompt overhead,
        # so preflight remains an upper bound.
        reduction_payload_capacity = max(
            1,
            self.compiler.batch_tokens
            - self.compiler.prompt_overhead_tokens,
        )
        reduction_groups = max(
            1,
            (
                map_output + reduction_payload_capacity - 1
            ) // reduction_payload_capacity,
        )
        reduction_round_input = (
            map_output
            + reduction_groups * self.compiler.prompt_overhead_tokens
        )
        reduction_input = (
            reduction_round_input
            * self.compiler.max_reduction_rounds
            * self.compiler.max_validation_attempts
            if needs_reduction else 0
        )
        reduction_output = (
            map_output
            * self.compiler.max_reduction_rounds
            * self.compiler.max_validation_attempts
            if needs_reduction else 0
        )
        synthesis_input = min(
            map_output, self.compiler.compiled_target_tokens
        )
        stages = [
            StageEstimate(
                "query_embedding",
                "openai/text-embedding-3-small",
                self.compiler.token_counter(
                    self._search_statement(bundle.query, history)
                ),
                0,
            ),
            StageEstimate(
                "evidence_map",
                self.compiler.map_model,
                map_input * map_attempts,
                map_output * map_attempts,
            ),
        ]
        if reduction_input:
            stages.append(StageEstimate(
                "evidence_reduce",
                self.compiler.map_model,
                reduction_input,
                reduction_output,
            ))
        if kind == "hypotheses":
            request = request or {}
            hypotheses = max(1, int(request.get("max_hypotheses", 5)))
            hypothesis_output = 12_000
            # Both synthesis stages retry a schema failure up to their own
            # ceiling, which is the same retry budget the map stage gets.
            stages.append(StageEstimate(
                "hypothesis_synthesis",
                self.compiler.synthesis_model,
                (synthesis_input + hypotheses * 256) * MAX_SYNTHESIS_ATTEMPTS,
                hypothesis_output * MAX_SYNTHESIS_ATTEMPTS,
            ))
            if request.get("critic"):
                stages.append(StageEstimate(
                    "hypothesis_critic",
                    self.compiler.synthesis_model,
                    (synthesis_input + hypothesis_output)
                    * MAX_SYNTHESIS_ATTEMPTS,
                    hypothesis_output * MAX_SYNTHESIS_ATTEMPTS,
                ))
        else:
            history_tokens = sum(
                self.compiler.token_counter(turn) for turn in history
            )
            stages.append(StageEstimate(
                "synthesis",
                self.compiler.synthesis_model,
                synthesis_input + history_tokens,
                8_000,
            ))
        return self.estimator.estimate(stages, caps={
            "max_papers": getattr(self.collector, "max_papers", 0),
            "max_chunks_per_paper": getattr(
                self.collector, "max_chunks_per_paper", 0
            ),
        })

    def _finalize_cost(self, usage_checkpoint: int) -> CostSummary:
        self.ledger.reconcile(
            self.cost_fetcher or (lambda _generation_id: {}),
            since=usage_checkpoint,
            fallback_unresolved=True,
        )
        return self.ledger.summary(usage_checkpoint)

    def _finalize_and_emit_cost(
        self,
        usage_checkpoint: int,
        publisher: _ProgressPublisher,
    ) -> CostSummary:
        cost = self._finalize_cost(usage_checkpoint)
        publisher.emit(
            "cost_actual",
            actual_cost_usd=cost.actual_cost_usd,
            cost_status=cost.cost_status,
            by_stage=dict(cost.by_stage),
        )
        return cost

    def _finalize_failure_cost(
        self,
        usage_checkpoint: int,
        publisher: _ProgressPublisher,
    ) -> None:
        try:
            self._finalize_and_emit_cost(usage_checkpoint, publisher)
        except Exception:
            # Cost reconciliation must never replace the research failure that
            # callers need to diagnose.
            return

    @staticmethod
    def _search_statement(
        query: str,
        history: tuple[str, ...],
    ) -> str:
        if not history:
            return query
        return "\n".join((*history, f"user: {query}"))

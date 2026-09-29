import json

import numpy as np
import pytest

from mira.exhaustive.compiler import EvidenceCompiler, ModelResponse
from mira.exhaustive.costs import CostEstimator, CostLedger, PricingCatalog
from mira.exhaustive.compiler import MAX_SYNTHESIS_ATTEMPTS
from mira.exhaustive.engine import (
    ExhaustiveCostLimitError,
    ExhaustiveResearchEngine,
    ExhaustiveResearchError,
    PreparedResearch,
)
from mira.exhaustive.evidence import EvidenceCollector
from mira.exhaustive.scoring import QueryScorer
from mira.exhaustive.snapshots import GraphSnapshotStore
from mira.exhaustive.types import DOMAIN_NAMES, UsageRecord
from tests.test_exhaustive_compiler import prompt_payload
from tests.test_exhaustive_snapshots import write_snapshot


class RecordingModel:
    def __init__(self, *, total_cost=0.001, generation_id=None):
        self.calls = []
        self.users = []
        self.total_cost = total_cost
        self.generation_id = generation_id

    def __call__(
        self,
        *,
        model,
        system,
        user,
        stage,
        max_output_tokens,
    ):
        self.calls.append(stage)
        self.users.append(user)
        if stage in {"evidence_map", "evidence_reduce"}:
            payload = prompt_payload(user)
            chunk_ids = (
                [item["chunk_id"] for item in payload["chunks"]]
                if "chunks" in payload
                else payload["expected_chunk_ids"]
            )
            chunk_metadata = {
                item["chunk_id"]: item
                for item in payload.get("chunks", ())
            }
            text = json.dumps({
                "processed_chunk_ids": chunk_ids,
                "records": [{
                    "claim": f"Evidence from {chunk_id}",
                    "measurements": [],
                    "mechanisms": [],
                    "assumptions": [],
                    "limitations": [],
                    "contradictions": [],
                    "citations": [{
                        "chunk_id": chunk_id,
                        "source_chunk_id": (
                            chunk_metadata[chunk_id]["provenance"][0][
                                "source_chunk_id"
                            ]
                            if chunk_id in chunk_metadata
                            else payload["records"][0]["citations"][0][
                                "source_chunk_id"
                            ]
                        ),
                        "file_path": (
                            chunk_metadata[chunk_id]["provenance"][0][
                                "file_path"
                            ]
                            if chunk_id in chunk_metadata
                            else payload["records"][0]["citations"][0][
                                "file_path"
                            ]
                        ),
                        "title": (
                            chunk_metadata[chunk_id]["provenance"][0][
                                "paper_name"
                            ]
                            if chunk_id in chunk_metadata
                            else payload["records"][0]["citations"][0][
                                "title"
                            ]
                        ),
                        "domains": (
                            [
                                chunk_metadata[chunk_id]["provenance"][0][
                                    "domain"
                                ]
                            ]
                            if chunk_id in chunk_metadata
                            else payload["records"][0]["citations"][0][
                                "domains"
                            ]
                        ),
                    }],
                    "relevance": "direct",
                    "contributing_chunk_ids": [chunk_id],
                } for chunk_id in chunk_ids],
            })
        else:
            text = "# Exhaustive answer\n\nGrounded result."
        return ModelResponse(
            text=text,
            usage=UsageRecord(
                stage=stage,
                model=model,
                input_tokens=100,
                output_tokens=20,
                total_cost=self.total_cost,
                generation_id=self.generation_id,
            ),
        )


class CountingStore(GraphSnapshotStore):
    def __init__(self, directories):
        super().__init__(directories)
        self.pin_calls = 0

    def pin_all(self):
        self.pin_calls += 1
        return super().pin_all()


def _engine(tmp_path, model=None):
    directories = {domain: tmp_path / domain for domain in DOMAIN_NAMES}
    for domain, directory in directories.items():
        write_snapshot(directory, domain)
    model = model or RecordingModel()
    ledger = CostLedger()
    compiler = EvidenceCompiler(
        model,
        ledger,
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
    )
    store = CountingStore(directories)
    engine = ExhaustiveResearchEngine(
        store=store,
        scorer=QueryScorer(
            lambda _query: np.asarray([1.0, 0.0], dtype=np.float32),
            node_threshold=0.0,
            edge_threshold=0.0,
        ),
        collector=EvidenceCollector(),
        compiler=compiler,
        estimator=CostEstimator(PricingCatalog.fallback()),
        ledger=ledger,
    )
    return engine, model, store


def test_engine_scans_all_domains_before_estimate_and_inference(tmp_path):
    engine, model, _store = _engine(tmp_path)
    events = []

    answer = engine.run_chat("question", (), events.append)

    names = [event.name for event in events]
    assert names.index("domain_scanned:storage") < names.index("cost_estimated")
    assert names.index("cost_estimated") < names.index(
        "evidence_batch_started"
    )
    assert answer.coverage.exhaustive is True
    assert set(answer.coverage.nodes_scanned) == set(DOMAIN_NAMES)
    assert answer.estimated_cost.total_with_reserve_usd > 0
    assert answer.cost.actual_cost_usd == pytest.approx(0.002)
    assert model.calls == ["evidence_map", "synthesis"]
    assert [event.sequence for event in events] == list(
        range(1, len(events) + 1)
    )
    assert all(json.dumps(event.to_dict()) for event in events)


def test_prepare_is_model_free_and_can_be_reused_without_rescanning(tmp_path):
    engine, model, store = _engine(tmp_path)
    events = []

    prepared = engine.prepare("question", ("prior context",), events.append)

    assert isinstance(prepared, PreparedResearch)
    assert model.calls == []
    assert store.pin_calls == 1
    first = engine.run_chat(prepared, (), events.append)
    assert store.pin_calls == 1
    assert first.coverage is not prepared.bundle
    assert first.coverage.exhaustive is True
    assert not hasattr(first.coverage, "chunks")
    assert not hasattr(first.coverage, "selected_regions")
    assert first.coverage.chunk_count == len(prepared.bundle.chunks)
    assert first.coverage.paper_count == len(prepared.bundle.paper_names)
    assert all(
        summary.file_count > 0
        for summary in first.coverage.snapshot_fingerprints.values()
    )


def test_engine_forwards_prepared_history_to_synthesis(tmp_path):
    engine, model, _store = _engine(tmp_path)
    search_inputs = []
    engine.scorer.embed_query = lambda query: (
        search_inputs.append(query)
        or np.asarray([1.0, 0.0], dtype=np.float32)
    )
    prepared = engine.prepare(
        "follow-up question",
        ("user: first question", "assistant: first answer"),
    )

    engine.run_chat(prepared)

    synthesis_payload = json.loads(model.users[-1])
    assert synthesis_payload["conversation_history"] == [
        "user: first question",
        "assistant: first answer",
    ]
    assert search_inputs == [
        "user: first question\n"
        "assistant: first answer\n"
        "user: follow-up question"
    ]


def test_engine_reconciles_pending_provider_cost_before_returning(tmp_path):
    model = RecordingModel(
        total_cost=None,
        generation_id="gen-pending",
    )
    engine, _model, _store = _engine(tmp_path, model)
    engine.cost_fetcher = lambda generation_id: {
        "data": {
            "id": generation_id,
            "total_cost": 0.0042,
            "tokens_prompt": 100,
            "tokens_completion": 20,
        }
    }

    answer = engine.run_chat("question")

    assert answer.cost.cost_status == "actual"
    assert answer.cost.actual_cost_usd == pytest.approx(0.0084)


def test_reused_engine_reports_cost_for_each_call_not_lifetime_total(tmp_path):
    engine, _model, _store = _engine(tmp_path)

    first = engine.run_chat("first", (), lambda _event: None)
    second = engine.run_chat("second", (), lambda _event: None)

    assert first.cost.actual_cost_usd == pytest.approx(0.002)
    assert second.cost.actual_cost_usd == pytest.approx(0.002)


def test_hypothesis_estimate_prices_shared_synthesis_and_optional_critic(
    tmp_path,
):
    engine, _model, _store = _engine(tmp_path)
    prepared = engine.prepare("question")

    estimate = engine._estimate(
        prepared.bundle,
        kind="hypotheses",
        request={"max_hypotheses": 5, "critic": True},
    )

    assert "synthesis" not in estimate.by_stage
    assert "hypothesis_synthesis" in estimate.by_stage
    assert "hypothesis_critic" in estimate.by_stage


def test_reduction_estimate_budgets_full_output_at_each_possible_level(
    tmp_path,
):
    engine, _model, _store = _engine(tmp_path)
    prepared = engine.prepare("question")
    engine.compiler.compiled_target_tokens = 1

    estimate = engine._estimate(prepared.bundle)

    assert "evidence_reduce" in estimate.by_stage
    map_output = (
        engine.compiler.batch_count(prepared.bundle.chunks) * 16_000
    )
    reduction_tokens = (
        map_output
        * engine.compiler.max_reduction_rounds
        * engine.compiler.max_validation_attempts
    )
    expected = PricingCatalog.fallback().price(
        engine.compiler.map_model
    ).calculate(reduction_tokens, reduction_tokens)
    assert estimate.by_stage["evidence_reduce"] == pytest.approx(
        expected
    )


def test_one_domain_failure_prevents_any_model_call(tmp_path):
    engine, model, _store = _engine(tmp_path)
    missing = tmp_path / "optical" / "vdb_entities.json"
    missing.unlink()

    with pytest.raises(ExhaustiveResearchError, match="prepare"):
        engine.run_chat("question", (), lambda _event: None)

    assert model.calls == []


def test_run_aborts_before_any_model_call_when_the_estimate_exceeds_the_limit(
    tmp_path,
):
    engine, model, _store = _engine(tmp_path)
    engine.max_estimated_cost_usd = 1e-9
    events = []

    with pytest.raises(ExhaustiveCostLimitError) as exc:
        engine.run_chat("question", (), events.append)

    assert "exceeds the configured limit" in str(exc.value)
    assert "$0.00" in str(exc.value)
    assert "MIRA_MAX_ESTIMATED_COST_USD" in str(exc.value)
    assert model.calls == []
    assert "cost_limit_exceeded" in [event.name for event in events]


def test_hypothesis_run_is_gated_by_the_same_ceiling(tmp_path):
    engine, model, _store = _engine(tmp_path)
    engine.hypothesis_runner = lambda *_args: None
    engine.max_estimated_cost_usd = 1e-9

    with pytest.raises(ExhaustiveCostLimitError):
        engine.run_hypotheses("question", {"max_hypotheses": 1})

    assert model.calls == []


def test_an_affordable_run_is_not_gated(tmp_path):
    engine, model, _store = _engine(tmp_path)

    engine.run_chat("question", (), lambda _event: None)

    assert model.calls == ["evidence_map", "synthesis"]


def test_map_estimate_budgets_every_validation_attempt(tmp_path):
    """The map stage dominates the bill and retries at the same volume."""
    engine, _model, _store = _engine(tmp_path)
    prepared = engine.prepare("question")

    estimate = engine._estimate(prepared.bundle)

    attempts = engine.compiler.max_validation_attempts
    assert attempts == 3
    chunk_tokens = sum(
        engine.compiler.token_counter(chunk.content)
        for chunk in prepared.bundle.chunks
    )
    batch_count = engine.compiler.batch_count(prepared.bundle.chunks)
    map_input = chunk_tokens + batch_count * (
        engine.compiler.prompt_overhead_tokens
    )
    map_output = max(
        256 * len(prepared.bundle.chunks),
        int(map_input * 0.15),
        batch_count * 16_000,
    )
    expected = PricingCatalog.fallback().price(
        engine.compiler.map_model
    ).calculate(map_input * attempts, map_output * attempts)
    assert estimate.by_stage["evidence_map"] == pytest.approx(expected)


def test_hypothesis_stages_budget_their_own_retry_ceiling(tmp_path):
    engine, _model, _store = _engine(tmp_path)
    prepared = engine.prepare("question")

    estimate = engine._estimate(
        prepared.bundle,
        kind="hypotheses",
        request={"max_hypotheses": 5, "critic": True},
    )

    batch_count = engine.compiler.batch_count(prepared.bundle.chunks)
    map_output = max(
        256 * len(prepared.bundle.chunks),
        int(sum(
            engine.compiler.token_counter(chunk.content)
            for chunk in prepared.bundle.chunks
        ) * 0.15),
        batch_count * 16_000,
    )
    synthesis_input = min(map_output, engine.compiler.compiled_target_tokens)
    price = PricingCatalog.fallback().price(engine.compiler.synthesis_model)
    assert estimate.by_stage["hypothesis_synthesis"] == pytest.approx(
        price.calculate(
            (synthesis_input + 5 * 256) * MAX_SYNTHESIS_ATTEMPTS,
            12_000 * MAX_SYNTHESIS_ATTEMPTS,
        )
    )
    assert estimate.by_stage["hypothesis_critic"] == pytest.approx(
        price.calculate(
            (synthesis_input + 12_000) * MAX_SYNTHESIS_ATTEMPTS,
            12_000 * MAX_SYNTHESIS_ATTEMPTS,
        )
    )


def test_estimate_reports_the_fan_out_caps_it_assumed(tmp_path):
    engine, _model, _store = _engine(tmp_path)
    engine.collector = EvidenceCollector(
        max_papers=7, max_chunks_per_paper=3
    )
    events = []

    prepared = engine.prepare("question", (), events.append)

    assert dict(prepared.estimated_cost.caps) == {
        "max_papers": 7,
        "max_chunks_per_paper": 3,
    }
    estimated = next(
        event for event in events if event.name == "cost_estimated"
    )
    assert estimated.data["caps"] == {
        "max_papers": 7,
        "max_chunks_per_paper": 3,
    }


def test_hypothesis_result_carries_citations_not_the_evidence_corpus(tmp_path):
    engine, _model, _store = _engine(tmp_path)
    engine.hypothesis_runner = lambda _prepared, _compiled, _request, _emit: (
        "dossier"
    )

    result = engine.run_hypotheses("question", {"max_hypotheses": 1})

    assert not hasattr(result, "evidence")
    assert result.citations
    assert all(
        isinstance(citation.chunk_id, str) for citation in result.citations
    )

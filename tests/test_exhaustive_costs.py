from datetime import datetime, timezone

import pytest

from mira.exhaustive.costs import (
    CostEstimator,
    CostLedger,
    PricingCatalog,
    StageEstimate,
)
from mira.exhaustive.types import UsageRecord


def test_spec_actual_cost_canary():
    ledger = CostLedger()
    ledger.record_usage(UsageRecord(
        stage="evidence_map",
        model="google/gemini-2.5-flash",
        input_tokens=500_000,
        output_tokens=25_000,
        total_cost=0.2125,
        generation_id="gen-map",
    ))
    ledger.record_usage(UsageRecord(
        stage="synthesis",
        model="anthropic/claude-sonnet-4.6",
        input_tokens=30_000,
        output_tokens=3_000,
        total_cost=0.135,
        generation_id="gen-synthesis",
    ))

    summary = ledger.summary()

    assert summary.actual_cost_usd == pytest.approx(0.3475)
    assert dict(summary.by_stage) == {
        "evidence_map": pytest.approx(0.2125),
        "synthesis": pytest.approx(0.135),
    }
    assert summary.cost_status == "actual"


def test_estimate_uses_model_prices_and_adds_ten_percent_reserve():
    estimate = CostEstimator(
        PricingCatalog.fallback(), reserve=0.10
    ).estimate((
        StageEstimate(
            "evidence_map",
            "google/gemini-2.5-flash",
            500_000,
            25_000,
        ),
        StageEstimate(
            "synthesis",
            "anthropic/claude-sonnet-4.6",
            30_000,
            3_000,
        ),
    ))

    assert estimate.subtotal_usd == pytest.approx(0.3475)
    assert estimate.total_with_reserve_usd == pytest.approx(0.38225)
    assert dict(estimate.by_stage) == {
        "evidence_map": pytest.approx(0.2125),
        "synthesis": pytest.approx(0.135),
    }


def test_missing_provider_cost_is_pending_until_reconciled():
    ledger = CostLedger()
    ledger.record_usage(UsageRecord(
        stage="evidence_map",
        model="google/gemini-2.5-flash",
        input_tokens=100,
        output_tokens=20,
        total_cost=None,
        generation_id="gen-pending",
    ))
    assert ledger.summary().cost_status == "cost_pending"
    assert ledger.summary().pending_generation_ids == ("gen-pending",)

    ledger.reconcile(lambda generation_id: {
        "data": {
            "id": generation_id,
            "total_cost": 0.0042,
            "tokens_prompt": 100,
            "tokens_completion": 20,
        }
    })

    assert ledger.summary().cost_status == "actual"
    assert ledger.summary().actual_cost_usd == pytest.approx(0.0042)
    assert ledger.summary().pending_generation_ids == ()


def test_unavailable_provider_cost_falls_back_to_catalog_pricing():
    ledger = CostLedger()
    ledger.record_usage(UsageRecord(
        stage="evidence_map",
        model="google/gemini-2.5-flash",
        input_tokens=1_000,
        output_tokens=100,
        total_cost=None,
        generation_id="gen-unavailable",
    ))

    ledger.reconcile(
        lambda _generation_id: {"data": {"total_cost": None}},
        fallback_unresolved=True,
    )

    summary = ledger.summary()
    assert summary.cost_status == "calculated"
    assert summary.actual_cost_usd == pytest.approx(0.00055)
    assert summary.pending_generation_ids == ()


def test_usage_without_generation_id_is_labeled_calculated():
    ledger = CostLedger()
    ledger.record_usage(UsageRecord(
        stage="synthesis",
        model="anthropic/claude-sonnet-4.6",
        input_tokens=1_000,
        output_tokens=100,
        total_cost=None,
        generation_id=None,
    ))

    summary = ledger.summary()

    assert summary.cost_status == "calculated"
    assert summary.actual_cost_usd == pytest.approx(0.0045)


def test_openrouter_catalog_converts_per_token_prices_to_per_million():
    fetched_at = datetime(2026, 7, 29, tzinfo=timezone.utc)
    catalog = PricingCatalog.from_openrouter({
        "data": [{
            "id": "example/model",
            "pricing": {"prompt": "0.0000003", "completion": "0.0000025"},
        }]
    }, fetched_at=fetched_at)

    price = catalog.price("example/model")

    assert price.input_per_million == pytest.approx(0.30)
    assert price.output_per_million == pytest.approx(2.50)
    assert catalog.fetched_at == fetched_at

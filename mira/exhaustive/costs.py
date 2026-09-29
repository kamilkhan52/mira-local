"""Pre-inference estimates and provider-authoritative cost accounting."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from .types import CostSummary, UsageRecord


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float

    def calculate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million
            + output_tokens * self.output_per_million
        ) / 1_000_000


class PricingCatalog:
    def __init__(
        self,
        prices: Mapping[str, ModelPrice],
        *,
        fetched_at: datetime,
    ):
        self._prices = MappingProxyType(dict(prices))
        self.fetched_at = fetched_at

    @classmethod
    def fallback(cls) -> "PricingCatalog":
        return cls(
            {
                "openai/text-embedding-3-small": ModelPrice(0.02, 0.0),
                "text-embedding-3-small": ModelPrice(0.02, 0.0),
                "google/gemini-2.5-flash": ModelPrice(0.30, 2.50),
                "anthropic/claude-sonnet-4.6": ModelPrice(3.0, 15.0),
                # Placeholder list prices, flagged in the `_llm_models_note`
                # of configs/memory-innovation-profile.json. Live runs fetch
                # the OpenRouter catalog at startup and only reach this table
                # when that fetch fails, which is logged as a WARNING.
                "anthropic/claude-sonnet-5": ModelPrice(3.0, 15.0),
                "openai/gpt-5.6-terra": ModelPrice(0.30, 2.50),
            },
            fetched_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        )

    @classmethod
    def from_openrouter(
        cls,
        payload: dict,
        *,
        fetched_at: datetime | None = None,
    ) -> "PricingCatalog":
        prices = {}
        for record in payload.get("data", []):
            model = record.get("id")
            pricing = record.get("pricing") or {}
            if not model:
                continue
            try:
                input_price = float(pricing["prompt"]) * 1_000_000
                output_price = float(pricing["completion"]) * 1_000_000
            except (KeyError, TypeError, ValueError):
                continue
            if input_price < 0 or output_price < 0:
                continue
            prices[str(model)] = ModelPrice(input_price, output_price)
        if not prices:
            raise ValueError("OpenRouter pricing payload contains no model prices")
        return cls(
            prices,
            fetched_at=fetched_at or datetime.now(timezone.utc),
        )

    def price(self, model: str) -> ModelPrice:
        try:
            return self._prices[model]
        except KeyError:
            raise KeyError(f"no price configured for model {model!r}") from None


@dataclass(frozen=True, slots=True)
class StageEstimate:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class CostEstimate:
    subtotal_usd: float
    total_with_reserve_usd: float
    reserve: float
    by_stage: Mapping[str, float]
    # Fan-out ceilings the estimate was computed under, so a reader can tell a
    # small figure from a capped one.
    caps: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({}))


class CostEstimator:
    def __init__(
        self,
        catalog: PricingCatalog,
        *,
        reserve: float = 0.10,
    ):
        if reserve < 0:
            raise ValueError("reserve must be non-negative")
        self.catalog = catalog
        self.reserve = reserve

    def estimate(
        self,
        stages: Iterable[StageEstimate],
        *,
        caps: Mapping[str, int] | None = None,
    ) -> CostEstimate:
        by_stage: dict[str, float] = {}
        for stage in stages:
            cost = self.catalog.price(stage.model).calculate(
                stage.input_tokens, stage.output_tokens
            )
            by_stage[stage.stage] = by_stage.get(stage.stage, 0.0) + cost
        subtotal = sum(by_stage.values())
        return CostEstimate(
            subtotal_usd=subtotal,
            total_with_reserve_usd=subtotal * (1.0 + self.reserve),
            reserve=self.reserve,
            by_stage=MappingProxyType(by_stage),
            caps=MappingProxyType(dict(caps or {})),
        )


class CostLedger:
    def __init__(self, catalog: PricingCatalog | None = None):
        self.catalog = catalog or PricingCatalog.fallback()
        self._records: list[UsageRecord] = []

    def record_usage(self, record: UsageRecord) -> None:
        if record.input_tokens < 0 or record.output_tokens < 0:
            raise ValueError("usage token counts must be non-negative")
        if record.total_cost is not None and record.total_cost < 0:
            raise ValueError("usage total_cost must be non-negative")
        self._records.append(record)

    def checkpoint(self) -> int:
        return len(self._records)

    def reconcile(
        self,
        fetch_generation: Callable[[str], dict],
        *,
        since: int = 0,
        fallback_unresolved: bool = False,
    ) -> None:
        if since < 0 or since > len(self._records):
            raise ValueError("usage checkpoint is outside the ledger")
        for index in range(since, len(self._records)):
            record = self._records[index]
            if record.total_cost is not None or not record.generation_id:
                continue
            try:
                response = fetch_generation(record.generation_id)
                data = response.get("data", response)
                total_cost = data.get("total_cost")
                if total_cost is None:
                    raise ValueError("provider cost is not available")
                total_cost = float(total_cost)
                if total_cost < 0:
                    raise ValueError("provider cost is negative")
                self._records[index] = replace(
                    record,
                    total_cost=total_cost,
                    input_tokens=int(
                        data.get("tokens_prompt", record.input_tokens)
                    ),
                    output_tokens=int(
                        data.get("tokens_completion", record.output_tokens)
                    ),
                )
            except (KeyError, TypeError, ValueError, OSError):
                if fallback_unresolved:
                    self._records[index] = replace(
                        record,
                        generation_id=None,
                    )
                continue

    def summary(self, since: int = 0) -> CostSummary:
        if since < 0 or since > len(self._records):
            raise ValueError("usage checkpoint is outside the ledger")
        by_stage: dict[str, float] = {}
        pending = []
        calculated = False
        for record in self._records[since:]:
            if record.total_cost is not None:
                cost = record.total_cost
            elif record.generation_id:
                pending.append(record.generation_id)
                continue
            else:
                calculated = True
                cost = self.catalog.price(record.model).calculate(
                    record.input_tokens, record.output_tokens
                )
            by_stage[record.stage] = by_stage.get(record.stage, 0.0) + cost
        status = (
            "cost_pending"
            if pending
            else "calculated"
            if calculated
            else "actual"
        )
        return CostSummary(
            actual_cost_usd=sum(by_stage.values()),
            cost_status=status,
            by_stage=MappingProxyType(by_stage),
            pending_generation_ids=tuple(sorted(pending)),
        )

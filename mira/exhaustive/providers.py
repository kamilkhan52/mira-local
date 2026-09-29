"""OpenRouter-backed provider adapters for exhaustive research workers."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import httpx
import numpy as np
from openai import OpenAI

from chatbot.config import Settings

from .compiler import EvidenceCompiler, ModelResponse, batch_response_schema
from .costs import CostEstimator, CostLedger, PricingCatalog
from .engine import ExhaustiveResearchEngine
from .evidence import EvidenceCollector
from .scoring import QueryScorer
from .snapshots import GraphSnapshotStore
from .types import UsageRecord
from mira.hypothesis.exhaustive import ExhaustiveHypothesisRunner


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
# Stages whose output is parsed as a validated JSON payload.
_JSON_STAGES = frozenset({"evidence_map", "evidence_reduce"})

logger = logging.getLogger(__name__)


def _extra_value(value: Any, name: str, default: Any = None) -> Any:
    direct = getattr(value, name, None)
    if direct is not None:
        return direct
    extra = getattr(value, "model_extra", None) or {}
    return extra.get(name, default)


def _live_catalog(api_key: str) -> PricingCatalog:
    """Price from the live OpenRouter catalog, falling back loudly.

    The static table is a snapshot with hand-entered placeholders for the
    newest models; running on it silently would make every estimate and every
    calculated cost wrong without anyone noticing.
    """
    try:
        response = httpx.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )
        response.raise_for_status()
        return PricingCatalog.from_openrouter(response.json())
    except Exception as exc:
        logger.warning(
            "OpenRouter pricing unavailable (%s); using fallback pricing "
            "from %s. Cost estimates and calculated costs may be wrong.",
            exc,
            PricingCatalog.fallback().fetched_at.date(),
        )
        return PricingCatalog.fallback()


def build_engine(settings: Settings) -> ExhaustiveResearchEngine:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for research")
    catalog = _live_catalog(api_key)
    for model in {
        EMBEDDING_MODEL,
        settings.exhaustive_map_model,
        settings.exhaustive_synthesis_model,
    }:
        try:
            catalog.price(model)
        except KeyError as exc:
            raise RuntimeError(
                f"no price configured for exhaustive model {model!r}"
            ) from exc
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    ledger = CostLedger(catalog)

    def fetch_generation(generation_id: str) -> dict:
        try:
            response = httpx.get(
                f"{OPENROUTER_BASE_URL}/generation",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"id": generation_id},
                timeout=10.0,
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
        except (httpx.HTTPError, ValueError):
            return {}

    def embed_query(query: str) -> np.ndarray:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        usage = response.usage
        ledger.record_usage(UsageRecord(
            stage="query_embedding",
            model=EMBEDDING_MODEL,
            input_tokens=int(getattr(usage, "prompt_tokens", 0)),
            output_tokens=0,
            total_cost=_extra_value(usage, "cost"),
            generation_id=getattr(response, "id", None),
        ))
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    def model_call(
        *,
        model: str,
        system: str,
        user: str,
        stage: str,
        max_output_tokens: int,
    ) -> ModelResponse:
        extra_body: dict[str, Any] = {"usage": {"include": True}}
        if stage in _JSON_STAGES:
            # State the contract to the provider as well as in the prompt: the
            # payload models forbid unknown fields, so an unconstrained
            # response is a retry the schema can prevent.
            extra_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evidence_batch",
                    "schema": batch_response_schema(),
                },
            }
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_output_tokens,
            temperature=0,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"{stage} returned empty content")
        usage = response.usage
        input_tokens = int(
            getattr(usage, "prompt_tokens", 0) if usage else 0
        )
        output_tokens = int(
            getattr(usage, "completion_tokens", 0) if usage else 0
        )
        details = (
            getattr(usage, "completion_tokens_details", None)
            if usage else None
        )
        return ModelResponse(
            text=content,
            usage=UsageRecord(
                stage=stage,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost=(
                    _extra_value(usage, "cost") if usage else None
                ),
                generation_id=getattr(response, "id", None),
                reasoning_tokens=int(
                    getattr(details, "reasoning_tokens", 0) or 0
                ),
            ),
        )

    compiler = EvidenceCompiler(
        model_call,
        ledger,
        batch_tokens=settings.exhaustive_batch_tokens,
        map_model=settings.exhaustive_map_model,
        synthesis_model=settings.exhaustive_synthesis_model,
    )
    hypothesis_runner = ExhaustiveHypothesisRunner(
        model_call,
        ledger,
        model=settings.exhaustive_synthesis_model,
    )
    return ExhaustiveResearchEngine(
        store=GraphSnapshotStore({
            "memory": settings.exhaustive_memory_dir,
            "optical": settings.exhaustive_optical_dir,
            "storage": settings.exhaustive_storage_dir,
        }),
        scorer=QueryScorer(
            embed_query,
            node_threshold=settings.exhaustive_node_threshold,
            edge_threshold=settings.exhaustive_edge_threshold,
        ),
        collector=EvidenceCollector(
            max_papers=settings.exhaustive_max_papers,
            max_chunks_per_paper=settings.exhaustive_max_chunks_per_paper,
        ),
        compiler=compiler,
        estimator=CostEstimator(catalog),
        ledger=ledger,
        hypothesis_runner=hypothesis_runner,
        cost_fetcher=fetch_generation,
        max_estimated_cost_usd=settings.max_estimated_cost_usd,
    )


def run_research_job(
    settings: Settings,
    kind: str,
    request: dict,
    emit: Callable[[Any], None],
) -> Any:
    engine = build_engine(settings)
    if kind == "chat":
        history = tuple(
            f"{turn['role']}: {turn['content']}"
            for turn in request.get("history", ())
        )
        return engine.run_chat(request["query"], history, emit)
    if kind == "hypotheses":
        query = "Research hypotheses connecting: " + ", ".join(
            request["topics"]
        )
        return engine.run_hypotheses(query, request, emit)
    raise ValueError(f"unsupported exhaustive research kind: {kind}")

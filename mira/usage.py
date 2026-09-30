"""Per-run accounting: LLM cost/tokens per stage, Jev calls, and stage wall
times. Prefect runs each flow run in its own process, so a process-global
ledger is per run; call reset() at the start of a run and snapshot() at the end.

Costs are what OpenRouter reports per call (usage accounting); Jev cost is
input tokens x JEV_PRICE_PER_MTOK (TypeSafe bills input tokens only).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager

JEV_PRICE_PER_MTOK = 0.042  # docs.typesafe.ai/models, jev-latest, 2026-09

# Caller function -> readable stage (llm_call infers the stage from its caller).
STAGE_NAMES = {
    "_get_affiliation": "credibility (affiliation)",
    "_classify_paper": "relevance (classification)",
    "_run_selection_agent": "selection",
    "_analyze_paper": "deep analysis",
    "run_media": "news selection + summary",
    "generate_report_text": "report",
    "parse_report_output": "report re-parse",
    "run_trend_analysis": "trend",
    "summarize": "live summaries",
}

_lock = threading.Lock()
_llm: dict = {}
_jev: dict = {}
_stages: dict = {}


def reset() -> None:
    with _lock:
        _llm.clear()
        _jev.clear()
        _stages.clear()


def record_llm(stage: str, model: str, usage, seconds: float) -> None:
    u = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
    details = u.get("completion_tokens_details") or {}
    with _lock:
        row = _llm.setdefault((STAGE_NAMES.get(stage, stage), model), defaultdict(float))
        row["calls"] += 1
        row["prompt_tokens"] += u.get("prompt_tokens") or 0
        row["completion_tokens"] += u.get("completion_tokens") or 0
        row["reasoning_tokens"] += details.get("reasoning_tokens") or 0
        row["cost_usd"] += u.get("cost") or 0.0
        row["seconds"] += seconds


def record_jev(stage: str, input_tokens: int, seconds: float) -> None:
    with _lock:
        row = _jev.setdefault(stage, defaultdict(float))
        row["calls"] += 1
        row["input_tokens"] += input_tokens or 0
        row["cost_usd"] += (input_tokens or 0) * JEV_PRICE_PER_MTOK / 1e6
        row["seconds"] += seconds


@contextmanager
def stage(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        with _lock:
            _stages[name] = _stages.get(name, 0.0) + time.perf_counter() - t0


def snapshot() -> dict:
    with _lock:
        llm = [{"stage": s, "model": m, **{k: round(v, 6) for k, v in r.items()}}
               for (s, m), r in _llm.items()]
        jev = [{"stage": s, **{k: round(v, 6) for k, v in r.items()}} for s, r in _jev.items()]
        stages = {k: round(v, 1) for k, v in _stages.items()}
    return {
        "llm": llm, "jev": jev, "stage_seconds": stages,
        "llm_cost_usd": round(sum(r["cost_usd"] for r in llm), 4),
        "jev_cost_usd": round(sum(r["cost_usd"] for r in jev), 6),
        "llm_calls": int(sum(r["calls"] for r in llm)),
        "jev_calls": int(sum(r["calls"] for r in jev)),
    }

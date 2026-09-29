from __future__ import annotations
from mira.paths import CONFIG_DIR
import argparse
import json
import os
import re
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

STAGE_KEYS = (
    "affiliation", "classification", "selection", "analysis",
    "trend", "report", "media_selection", "media_summary",
)

# Stages that live in the shared `llm_models` block but are not part of the
# n8n pipeline, so `load_config` must not require them. They are read by
# standalone tools (exhaustive research, the feasibility judge) through
# `shared_llm_model`.
OPTIONAL_STAGE_KEYS = (
    "exhaustive_map", "exhaustive_synthesis", "feasibility",
)

_PARSE_FAILURES: dict[str, int] = {}


def shared_llm_models() -> dict[str, str]:
    """Return the shared `llm_models` block, or {} when it cannot be read.

    Unlike `load_config` this resolves no profile, mode, or date window and
    reads no environment: standalone tools need one model name, not a full
    run config, and must not fail because RECIPIENT_EMAIL is unset.
    """
    for config_file in sorted((CONFIG_DIR).glob("*.json")):
        try:
            candidate = json.loads(config_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Some files in configs/ are list-rooted (crawler output: eetimes /
        # semianalysis / trendforce "latest" snapshots — untracked, gitignored,
        # but present in real checkouts). Guard before attribute access.
        if not isinstance(candidate, dict):
            continue
        models = candidate.get("llm_models")
        if isinstance(models, dict):
            return {str(k): str(v) for k, v in models.items()}
    return {}


def shared_llm_model(stage: str, default: str) -> str:
    """Resolve one optional stage model, falling back to a pinned default.

    Unlike `model_for` this never raises: these stages are optional keys, and
    a config that predates them must keep working on the shipped default.
    """
    return shared_llm_models().get(stage) or default


def model_for(config: dict, stage: str) -> str:
    """Resolve the model for one pipeline stage.

    Raises rather than falling back. A missing key used to resolve to an
    arbitrary other stage's model via dict-iteration order, which silently
    routed deep analysis and report generation to the wrong model.
    """
    models = config["llm_models"]
    if stage not in models:
        raise KeyError(
            f"No model configured for stage {stage!r}. Add it to 'llm_models' "
            f"in the profile config. Configured stages: {sorted(models)}"
        )
    return models[stage]


def record_parse_failure(stage: str) -> None:
    """Count a JSON parse failure so a bad model swap is visible, not silent.
    Lock-guarded since classify_papers fans out across 8 worker threads."""
    with _PARSE_FAILURES_LOCK:
        _PARSE_FAILURES[stage] = _PARSE_FAILURES.get(stage, 0) + 1


_PARSE_FAILURES_LOCK = threading.Lock()


def parse_failure_report() -> dict[str, int]:
    return dict(_PARSE_FAILURES)


def reset_parse_failures() -> None:
    _PARSE_FAILURES.clear()


def format_parse_failures(report: dict[str, int]) -> str:
    """Render parse-failure counts for the run summary. Empty string when clean."""
    if not report:
        return ""
    detail = ", ".join(f"{stage}={count}" for stage, count in sorted(report.items()))
    return f"WARNING: JSON parse failures by stage: {detail}"


def load_config(profile: str, mode: str, start_date: str | None = None, end_date: str | None = None) -> dict:
    configs_dir = CONFIG_DIR
    raw = None
    # Sorted, because BU profiles live in two files (the merged n8n config and a
    # standalone copy) and glob order is filesystem-dependent: unsorted, the
    # copy that wins here differs between this machine and the container.
    # Parity between the copies is pinned in tests/test_config.py.
    for config_file in sorted(configs_dir.glob("*.json")):
        try:
            candidate = json.loads(config_file.read_text())
        except OSError:
            continue
        if "profiles" in candidate and any(
            p.get("profile_id") == profile for p in candidate["profiles"]
        ):
            raw = candidate
            break

    if raw is None:
        raise ValueError(f"Profile '{profile}' not found in configs/")

    profile_data = next(
        (p for p in raw["profiles"] if p["profile_id"] == profile), None
    )
    if not profile_data:
        raise ValueError(f"Profile '{profile}' not found in configs/")

    if mode not in profile_data["modes"]:
        raise ValueError(f"Mode '{mode}' not found in profile '{profile}' (available: {list(profile_data['modes'].keys())})")
    mode_cfg = profile_data["modes"][mode]

    missing = [s for s in STAGE_KEYS if s not in raw.get("llm_models", {})]
    if missing:
        raise ValueError(
            f"Profile config is missing a model for stage(s): {', '.join(missing)}. "
            f"Every stage in STAGE_KEYS must be present in 'llm_models'."
        )
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.now()
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = end_dt - timedelta(days=mode_cfg["lookback_days"] - 1)

    recipient_email = os.environ.get("RECIPIENT_EMAIL")
    if not recipient_email:
        raise ValueError("RECIPIENT_EMAIL not set in .env or environment")

    return {
        "profile_id": profile,
        "mode": mode,
        "topic": profile_data["topic"],
        "arxiv": profile_data["arxiv"],
        "thresholds": profile_data["thresholds"],
        "prompts": profile_data["prompts"],
        "media": profile_data.get("media", {}),
        "themes": profile_data.get("themes", []),
        "email_cfg": profile_data["email"][mode],
        "llm_models": raw["llm_models"],
        "mode_cfg": mode_cfg,
        "start_date": start_dt.strftime("%Y%m%d"),
        "end_date": end_dt.strftime("%Y%m%d"),
        "start_date_iso": start_dt.strftime("%Y-%m-%d"),
        "end_date_iso": end_dt.strftime("%Y-%m-%d"),
        "current_date": end_dt.strftime("%Y-%m-%d"),
        "recipient_email": recipient_email,
    }


def iso_date_arg(text: str) -> date:
    """argparse type= for date flags: clean CLI error instead of a traceback."""
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}")


def make_llm_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env or environment")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def llm_call(client: OpenAI, model: str, system: str, user: str, retries: int = 3,
             temperature: float | None = None,
             reasoning_effort: str | None = None) -> str:
    # PR #30 review r2: n8n sends reasoning_effort=high ONLY on the per-paper
    # terra nodes (options.modelKwargs) — the CLI mirrors that for
    # affiliation/classification call sites (passed explicitly), not for
    # media/report/trend/etc. Default None = omit the parameter.
    extra = {} if temperature is None else {"temperature": temperature}
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort  # type: ignore[assignment]
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **extra,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("LLM returned empty content (possible content filter)")
            return content
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(10)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def apply_template(template: str, variables: dict) -> str:
    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        return str(variables.get(key, match.group(0)))
    return re.sub(r"\{\{(\w+)\}\}", replace, template)


def parse_json_response(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())

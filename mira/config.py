from __future__ import annotations
from mira.paths import CONFIG_DIR
import argparse
import json
import os
import re
import threading
import sys
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


# --- run configuration (n8n: Capture Trigger Name .. Set Run Mode) ----------

# The file n8n's "Read Topic Config File" node reads. Trigger routing uses it
# (and only it), exactly like the workflow.
MAIN_CONFIG_NAME = "memory-innovation-profile.json"
_JS_TRUE_VALUES = (True, "true", 1, "1", "yes", "on")


def js_number(value) -> float | int | None:
    """JavaScript Number(value), with NaN mapped to None.

    n8n's Code nodes coerce with Number(), which differs from int()/float():
    null/''/whitespace -> 0, booleans -> 0/1, '0x1f' -> 31, and anything else
    unparsable -> NaN. Integral results come back as int so JSON renders them
    the way JSON.stringify does ("8", not "8.0")."""
    if value is None:
        n = 0.0
    elif isinstance(value, bool):
        n = 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        n = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if text == "":
            n = 0.0
        else:
            try:
                if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", text):
                    n = float(text)
                elif re.fullmatch(r"0[xX][0-9a-fA-F]+", text):
                    n = float(int(text, 16))
                elif re.fullmatch(r"0[bB][01]+", text):
                    n = float(int(text, 2))
                elif re.fullmatch(r"0[oO][0-7]+", text):
                    n = float(int(text, 8))
                elif text in ("Infinity", "+Infinity", "-Infinity"):
                    n = float(text.replace("Infinity", "inf"))
                else:
                    return None
            except ValueError:
                return None
    elif isinstance(value, list):
        # Number([]) = 0, Number([x]) = Number(String(x)), longer arrays = NaN.
        if not value:
            n = 0.0
        elif len(value) == 1 and not isinstance(value[0], (list, dict)):
            return js_number("" if value[0] is None else str(value[0]))
        else:
            return None
    else:
        return None
    if n != n:  # NaN
        return None
    if n in (float("inf"), float("-inf")):
        return n
    return int(n) if n.is_integer() else n


def js_finite_number(value) -> float | int | None:
    """Number(value) when Number.isFinite, else None (n8n's toNumber helpers)."""
    n = js_number(value)
    if n is None or n in (float("inf"), float("-inf")):
        return None
    return n


def js_truthy(value) -> bool:
    """JavaScript truthiness (for n8n's `a || b` fallbacks)."""
    if isinstance(value, float) and value != value:
        return False
    return bool(value) if not isinstance(value, (list, dict)) else True


def js_or(value, fallback):
    """n8n's `value || fallback`."""
    return value if js_truthy(value) else fallback


def parse_bool(value, fallback: bool = False) -> bool:
    """Set Run Mode's parseBool: '', null and undefined give the fallback;
    only true/'true'/1/'1'/'yes'/'on' are true (case-sensitive, like n8n)."""
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return value in _JS_TRUE_VALUES


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _main_config() -> tuple[dict, Path]:
    """The profile config n8n reads (configs/memory-innovation-profile.json),
    falling back to the first configs/*.json that carries a `profiles` list."""
    main = CONFIG_DIR / MAIN_CONFIG_NAME
    raw = _read_json(main) if main.exists() else None
    if isinstance(raw, dict) and isinstance(raw.get("profiles"), list):
        return raw, main
    for config_file in sorted(CONFIG_DIR.glob("*.json")):
        candidate = _read_json(config_file)
        if isinstance(candidate, dict) and isinstance(candidate.get("profiles"), list):
            return candidate, config_file
    raise ValueError(f"No profile config with a 'profiles' list found in {CONFIG_DIR}")


def mode_from_trigger(trigger_name: str | None, profile: dict | None = None) -> str:
    """Set Run Mode: 'daily'/'weekly'/'monthly' substring of the trigger name,
    else the profile's default_mode, else 'weekly'."""
    lowered = (trigger_name or "").lower()
    for mode in ("daily", "weekly", "monthly"):
        if mode in lowered:
            return mode
    return (profile or {}).get("default_mode") or "weekly"


def _route_trigger(raw: dict, trigger_name: str | None) -> tuple[dict, bool, str]:
    """Load Topic Config's profile routing: first profile whose trigger_match
    entry is a (case-insensitive) substring of the trigger name. Otherwise only
    triggers listed in default_trigger_match (exact, trimmed, lowercased) or an
    empty trigger name may fall back to default_profile_id; anything else
    raises, so a mis-named schedule can never mail a digest labeled for the
    wrong topic. (The guarded routing is from the n8n export that carries
    default_trigger_match; the older live export fell back silently.)"""
    profiles = raw["profiles"]
    trigger = (trigger_name or "").strip()
    lowered = trigger.lower()
    for profile in profiles:
        if any(str(m).lower() in lowered for m in profile.get("trigger_match") or []):
            return profile, False, ""
    allowed = [str(t).lower().strip() for t in raw.get("default_trigger_match") or []]
    if lowered and lowered not in allowed:
        known = "; ".join(
            f"{p.get('profile_id')} [{', '.join(p.get('trigger_match') or [])}]" for p in profiles)
        raise ValueError(
            f'No profile matched trigger "{trigger_name}". Refusing to fall back to '
            f'"{raw.get("default_profile_id")}" because that would email a report labeled '
            f"for the wrong topic. Known profiles: {known}. Fix: add a profile whose "
            f"trigger_match covers this trigger, or add \"{trigger_name}\" to "
            f"default_trigger_match to allow the fallback.")
    reason = (f'trigger "{trigger_name}" is allow-listed for the default profile' if lowered
              else "no trigger name available (manual execution); using default profile")
    default = next((p for p in profiles if p.get("profile_id") == raw.get("default_profile_id")),
                   profiles[0])
    return default, True, reason


def resolve_profile(trigger_name: str | None) -> tuple[str, str]:
    """Route an n8n-style trigger/schedule name to (profile_id, mode).

    'CXL Monthly Trigger' -> ('cxl-research', 'monthly'); 'Backfill Trigger'
    (allow-listed) -> (default profile, its default_mode or 'weekly').
    Raises ValueError for a name that matches no profile and is not in
    default_trigger_match."""
    raw, _ = _main_config()
    profile, _, _ = _route_trigger(raw, trigger_name)
    return profile["profile_id"], mode_from_trigger(trigger_name, profile)


def duration_label(days) -> str:
    """Set Run Mode's buildDurationLabel."""
    d = js_number(days) or 0
    if d <= 0:
        return "recent period"
    if d == 1:
        return "last day"
    if d == 7:
        return "last week"
    if d % 30 == 0:
        months = round(d / 30)
        return "last month" if months == 1 else f"last {months} months"
    if d % 7 == 0 and d < 60:
        weeks = round(d / 7)
        return "last week" if weeks == 1 else f"last {weeks} weeks"
    return f"last {d} days"


def title_case(text: str) -> str:
    """Set Run Mode's toTitleCase (first letter of each space-separated word)."""
    return " ".join(w[:1].upper() + w[1:] for w in (text or "").split(" ")).strip()


def _range_label(mode_cfg: dict, lo_key: str, hi_key: str, mode: str, daily: str, other: str) -> str:
    lo = js_or(mode_cfg.get(lo_key), 0)
    hi = js_or(mode_cfg.get(hi_key), 0)
    label = f"{lo}-{hi}"
    return (daily if mode == "daily" else other) if label == "0-0" else label


def _find_profile(profile: str) -> tuple[dict, dict]:
    # Sorted, because BU profiles live in two files (the merged n8n config and a
    # standalone copy) and glob order is filesystem-dependent: unsorted, the
    # copy that wins here differs between this machine and the container.
    # Parity between the copies is pinned in tests/test_config.py.
    for config_file in sorted(CONFIG_DIR.glob("*.json")):
        candidate = _read_json(config_file)
        if not isinstance(candidate, dict) or not isinstance(candidate.get("profiles"), list):
            continue
        match = next((p for p in candidate["profiles"] if p.get("profile_id") == profile), None)
        if match is not None:
            return candidate, match
    raise ValueError(f"Profile '{profile}' not found in configs/")


def load_config(profile: str | None = None, mode: str | None = None,
                start_date: str | None = None, end_date: str | None = None, *,
                trigger_name: str | None = None,
                current_date: str | None = None,
                lookback_days=None,
                max_limit=None,
                test_mode=None,
                llm_cache_bypass=None,
                trend_enabled=None,
                recipient_email: str | None = None,
                now: datetime | None = None) -> dict:
    """Resolve one run's configuration the way n8n's Capture Trigger Name ->
    Set Max Limit Override -> Set Lookback Override -> Load Topic Config ->
    Set Run Mode -> Get Current Date -> Date & Time1 chain does.

    profile/mode: explicit, or routed from `trigger_name` (see resolve_profile);
    mode falls back to the trigger name, then the profile's default_mode, then
    'weekly'. Keyword overrides mirror the n8n webhook body fields:
    current_date (current_date_override: YYYY-MM-DD anchor, invalid values are
    ignored), lookback_days (lookback_days_override, used when > 0), max_limit
    (arXiv max_results override), test_mode (test_mode_override),
    llm_cache_bypass, trend_enabled (trend_enabled_override).
    start_date/end_date are the CLI's explicit window (end_date acts as the
    anchor when current_date is not given)."""
    if profile is None:
        raw_main, _ = _main_config()
        routed, fallback, fallback_reason = _route_trigger(raw_main, trigger_name)
        profile = routed["profile_id"]
    else:
        fallback, fallback_reason = False, ""
    raw, profile_data = _find_profile(profile)

    if mode is None:
        mode = mode_from_trigger(trigger_name, profile_data)
    modes = profile_data.get("modes") or {}
    if mode not in modes:
        raise ValueError(f"Mode '{mode}' not found in profile '{profile}' (available: {list(modes.keys())})")
    mode_cfg = modes[mode]

    # Load Topic Config: shared llm_models, overridden per profile.
    llm_models = {**(raw.get("llm_models") or {}), **(profile_data.get("llm_models") or {})}
    missing = [s for s in STAGE_KEYS if s not in llm_models]
    if missing:
        raise ValueError(
            f"Profile config is missing a model for stage(s): {', '.join(missing)}. "
            f"Every stage in STAGE_KEYS must be present in 'llm_models'."
        )

    # Lookback: override when a finite number > 0, else the mode's
    # lookback_days, else 1/7/30 by mode name.
    override_days = js_finite_number(lookback_days) if lookback_days is not None else None
    if override_days is not None and override_days > 0:
        lookback = override_days
    else:
        lookback = mode_cfg.get("lookback_days")
        if lookback is None:
            lookback = 1 if mode == "daily" else 7 if mode == "weekly" else 30
    range_days = max(0, (js_number(lookback) or 0) - 1)

    # Anchor ("today"): a valid current_date override, else the CLI end date,
    # else now. n8n silently ignores a malformed current_date_override.
    now = now or datetime.now()
    override = str(current_date or "")[:10]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", override):
        try:
            end_dt = datetime.strptime(override, "%Y-%m-%d")
        except ValueError:  # 2026-02-31: Luxon's fromISO is invalid -> $now
            end_dt = now
    elif end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = now
    end_dt = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        # Explicit CLI window: labels follow its real length.
        lookback = (end_dt - start_dt).days + 1
    else:
        start_dt = end_dt - timedelta(days=range_days)

    topic = profile_data.get("topic") or {}
    period_label = duration_label(lookback)
    period_title = title_case(period_label)
    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso = end_dt.strftime("%Y-%m-%d")
    profile_id = profile_data.get("profile_id") or topic.get("name") or "default"
    profile_slug = re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", str(profile_id).lower())) or "default"

    # trend_enabled: an explicit override wins (parseBool, fallback true); else
    # the mode's trend_enabled, else true. NOTE (n8n quirk, reproduced): the
    # live workflow's "Set Lookback Override" node hardcodes
    # trend_enabled_override = true, so trend analysis is ALWAYS on in n8n
    # whatever the profile says. Passing trend_enabled=False opts out.
    trend_enabled_profile = mode_cfg.get("trend_enabled", True)
    effective_trend = parse_bool(True if trend_enabled is None else trend_enabled, True)

    if max_limit is not None and max_limit != "":
        parsed_limit = js_finite_number(max_limit)
        if parsed_limit is None:
            raise ValueError(f"max_limit must be a number, got {max_limit!r}")
        max_limit = parsed_limit
    else:
        max_limit = None

    is_test = parse_bool(test_mode, False)
    cache_bypass = parse_bool(llm_cache_bypass, False) or os.environ.get("MIRA_LLM_CACHE_BYPASS") == "1"
    # Delivery is the report stage's concern; a missing recipient is not a
    # config error (test runs and callers with their own recipients need none).
    recipient = recipient_email or os.environ.get("RECIPIENT_EMAIL") or None
    email_cfgs = profile_data.get("email") or {}

    return {
        "profile_id": profile,
        "mode": mode,
        "topic": topic,
        "arxiv": profile_data.get("arxiv") or {},
        "thresholds": profile_data.get("thresholds") or {},
        "prompts": profile_data.get("prompts") or {},
        "media": profile_data.get("media", {}),
        "themes": profile_data.get("themes", []),
        # n8n never reads email[mode] (styling uses profile-level keys); a
        # mode with no block (cxl-research monthly) gets {} instead of a KeyError.
        "email_cfg": email_cfgs.get(mode) or {},
        "llm_models": llm_models,
        "mode_cfg": mode_cfg,
        "profile": {**profile_data, "llm_models": llm_models},
        "start_date": start_dt.strftime("%Y%m%d"),
        "end_date": end_dt.strftime("%Y%m%d"),
        "start_date_iso": start_iso,
        "end_date_iso": end_iso,
        "current_date": end_iso,
        "recipient_email": recipient,
        # --- Set Run Mode derived fields (n8n names in comments) ---
        "trigger_node": trigger_name or "",                       # triggerNode
        "profile_fallback": fallback,
        "profile_fallback_reason": fallback_reason,
        "profile_slug": profile_slug,                            # profileSlug
        "lookback_days": lookback,                               # lookbackDays
        "period_label": period_label,                            # periodLabel ("last week")
        "period_title": period_title,                            # periodTitle ("Last Week")
        "period_range": f"{start_iso} to {end_iso}",             # periodRange
        "digest_label": f"{topic.get('name') or 'Research'} Digest ({period_title})",  # digestLabel
        "selection_range_label": mode_cfg.get("selection_range_label") or _range_label(
            mode_cfg, "selection_min", "selection_max", mode, "2-5", "5-10"),
        "max_selection": js_or(mode_cfg.get("max_selection"), 10 if mode == "daily" else 20),
        "report_selection_range_label": mode_cfg.get("report_selection_range_label") or _range_label(
            mode_cfg, "report_selection_min", "report_selection_max", mode, "3-5", "7-10"),
        "report_max_selection": js_number(js_or(mode_cfg.get("report_max_selection"),
                                                5 if mode == "daily" else 10)),
        "max_limit": max_limit,                                  # maxLimit (arXiv override)
        "topic_name": topic.get("name") or "Research",           # topicName
        "topic_focus": topic.get("focus") or "research",         # topicFocus
        "assistant_signature": topic.get("assistant_signature") or "Research Assistant",
        "test_mode": is_test,                                    # isTestMode
        "llm_cache_bypass": cache_bypass,
        "trend_enabled": effective_trend,
        "trend_enabled_profile": trend_enabled_profile,
    }


def iso_date_arg(text: str) -> date:
    """argparse type= for date flags: clean CLI error instead of a traceback."""
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}")


DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MAX_TOKENS = 16000


def make_llm_client() -> OpenAI:
    """OpenAI-compatible client. OpenRouter by default; point MIRA_LLM_BASE_URL
    at a company/local endpoint to switch. Key: MIRA_LLM_API_KEY, falling back
    to OPENROUTER_API_KEY."""
    base_url = os.environ.get("MIRA_LLM_BASE_URL") or DEFAULT_LLM_BASE_URL
    api_key = os.environ.get("MIRA_LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("MIRA_LLM_API_KEY (or OPENROUTER_API_KEY) not set in .env or environment")
    return OpenAI(api_key=api_key, base_url=base_url)


def _default_max_tokens() -> int | None:
    raw = os.environ.get("MIRA_LLM_MAX_TOKENS")
    if raw is None or raw.strip() == "":
        return DEFAULT_LLM_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_LLM_MAX_TOKENS
    return value if value > 0 else None  # 0 / negative: send no cap


def llm_call(client: OpenAI, model: str, system: str, user: str, retries: int = 3,
             temperature: float | None = None,
             reasoning_effort: str | None = None,
             max_tokens: int | None = None,
             retry_wait: float = 10,
             schema: str | dict | None = None) -> str:
    # PR #30 review r2: n8n sends reasoning_effort=high ONLY on the per-paper
    # terra nodes (options.modelKwargs) — the CLI mirrors that for
    # affiliation/classification call sites (passed explicitly), not for
    # media/report/trend/etc. Default None = omit the parameter.
    extra = {} if temperature is None else {"temperature": temperature}
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort  # type: ignore[assignment]
    # Always cap output: without max_tokens OpenRouter reserves the model's full
    # output window against the prepaid balance and rejects the request when
    # the balance is low. Explicit argument > MIRA_LLM_MAX_TOKENS > 16000.
    cap = max_tokens if max_tokens is not None else _default_max_tokens()
    if cap is not None:
        extra["max_tokens"] = cap  # type: ignore[assignment]
    caller = sys._getframe(1).f_code.co_name  # stage label for the usage ledger
    # schema: answer through n8n's `format_final_json_response` tool (the
    # agent + Structured Output Parser protocol) instead of free JSON text.
    from mira import structured
    if schema is not None:
        schema_obj = structured.SCHEMAS[schema] if isinstance(schema, str) else schema
        system = structured.system_with_instruction(system)
        extra["tools"] = [structured.tool_for(schema_obj)]  # type: ignore[assignment]
    last_err: Exception | None = None
    budget_waits = 0
    attempt = -1
    while attempt < retries - 1:
        attempt += 1
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # OpenRouter usage accounting: returns the call's cost.
                extra_body={"usage": {"include": True}},
                **extra,
            )
            from mira import usage
            usage.record_llm(caller, model, getattr(resp, "usage", None), time.perf_counter() - t0)
            message = resp.choices[0].message
            calls = getattr(message, "tool_calls", None)
            if schema is not None and isinstance(calls, (list, tuple)) and calls:
                return structured.unwrap(calls[0].function.arguments)
            content = message.content
            if content is None:
                raise RuntimeError("LLM returned empty content (possible content filter)")
            return content
        except Exception as e:
            last_err = e
            # OpenRouter 402 "in_flight_budget_exhausted": the prepaid balance
            # is momentarily below the sum of in-flight reservations (e.g.
            # while an auto-reload lands). Wait it out instead of failing.
            if _is_budget_wait(e) and budget_waits < BUDGET_WAIT_MAX:
                budget_waits += 1
                time.sleep(BUDGET_WAIT_SECONDS)
                attempt -= 1  # a budget wait is not a failed attempt
                continue
            if attempt < retries - 1:
                time.sleep(retry_wait)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


BUDGET_WAIT_SECONDS = 30
BUDGET_WAIT_MAX = 20  # up to 10 minutes


def _is_budget_wait(e: Exception) -> bool:
    text = str(e)
    return "402" in text and ("in_flight" in text or "Payment required" in text or "more credits" in text)


def apply_template(template: str, variables: dict) -> str:
    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        return str(variables.get(key, match.group(0)))
    return re.sub(r"\{\{(\w+)\}\}", replace, template)


def parse_json_response(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
    return json.loads(text.strip())

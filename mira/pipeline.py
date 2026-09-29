from __future__ import annotations
from mira.paths import REPORT_FILES, LOCAL_CACHE, TEMP_DIR, SCRIPTS_DIR
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent.parent

# n8n fans its LLM nodes out across items; this loop used to be strictly serial,
# which is why a 366-paper weekly run took ~40 min in the CLI and ~21 in n8n.
CLASSIFY_CONCURRENCY = 8

# Ensure scripts/ is on the path and patch Docker-only path constants for local use
_scripts_dir = str(SCRIPTS_DIR)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

def _patch_pdf_paths() -> None:
    try:
        import download_full_arxiv_pdf as _m
        _m.CACHE_DIR = str(TEMP_DIR / "full_pdfs")
        _m.PDF_LIBRARY_DIR = str(TEMP_DIR / "pdf_library")
        _m.FULL_TEXT_CACHE_DIR = str(TEMP_DIR / "full_text_cache")
    except ImportError:
        pass  # scripts/ not available (test environment)

_patch_pdf_paths()


def _load_cache(name: str) -> dict:
    path = LOCAL_CACHE / f"{name}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"  WARNING: cache/{name}.json is malformed — starting with empty cache")
        return {}


def _save_cache(name: str, data: dict) -> None:
    path = LOCAL_CACHE / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


_STATS: Counter = Counter()
_STATS_LOCK = threading.Lock()


def _bump(key: str) -> None:
    with _STATS_LOCK:
        _STATS[key] += 1


def _safe_id(arxiv_id: str) -> str:
    """Same id-to-filename mapping n8n's cache-lookup Code nodes use."""
    bare = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", str(arxiv_id or ""))
    bare = re.sub(r"\.pdf$", "", bare, flags=re.IGNORECASE)
    return re.sub(r"[^a-zA-Z0-9._-]", "_", bare)


def _normalize_id(arxiv_id: str) -> str:
    """n8n's normalizeId: strip URL prefix and .pdf suffix, KEEP the version."""
    bare = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", str(arxiv_id or ""))
    return re.sub(r"\.pdf$", "", bare, flags=re.IGNORECASE)


def _djb2(value: str) -> str:
    """n8n's hash(): djb2 over JS string chars, unsigned 32-bit, hex.
    JS chars are UTF-16 code units — Python strings must be encoded to utf-16-le
    and read as uint16 pairs to match exactly."""
    data = value.encode("utf-16-le")
    h = 5381
    for i in range(0, len(data), 2):
        unit = data[i] | (data[i + 1] << 8)
        h = (((h << 5) + h) + unit) & 0xFFFFFFFF
    return format(h, "x")


def _apply_template_n8n(template: str, variables: dict) -> str:
    """Byte-identical port of n8n's applyTemplate (Build * Prompt nodes):
    replace {{ key }} (whitespace-tolerant) with the value; unknown keys and
    null/undefined values become EMPTY STRING (not the raw placeholder)."""
    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        v = variables.get(key)
        return "" if v is None else str(v)
    return re.sub(r"\{\{\s*([^}]+)\s*\}\}", replace, template)


def _shared_cache_dir(config: dict, stage: str) -> Path | None:
    """One JSON per paper under report-files/cache/<profile>/<stage>/ — the cache
    n8n already fills, SAME fingerprint scheme (djb2 over stage|schema|model|
    prompt_hash|system_hash|input_hash), so entries interoperate in BOTH
    directions. No profile_id (unit tests), MIRA_LLM_CACHE_BYPASS=1 or the run
    config's llm_cache_bypass (n8n webhook field) disables it."""
    profile = config.get("profile_id")
    if (not profile or os.environ.get("MIRA_LLM_CACHE_BYPASS") == "1"
            or config.get("llm_cache_bypass")):
        return None
    return REPORT_FILES / "cache" / profile / stage


# Per-stage index of the shared cache directory, built once per classify_papers()
# call. Replaces the round-1 glob-per-paper lookup, which cost ~22 ms/lookup on
# the real 24.6k-file cache dir (~220 s of pure glob per 9999-paper run).
_STAGE_INDEX: dict[tuple[str, str], dict] = {}
_STAGE_INDEX_LOCK = threading.Lock()


def _stage_index(cache_dir: Path, stage: str) -> dict[str, list]:
    key = (str(cache_dir), stage)
    with _STAGE_INDEX_LOCK:
        idx = _STAGE_INDEX.get(key)
        if idx is None or idx["_built_at"] < time.time() - 300:
            entries: dict[str, list] = {}
            if cache_dir.is_dir():
                for p in cache_dir.iterdir():
                    if not p.name.endswith(".json"):
                        continue
                    stem = p.name[:-5]
                    if "__" not in stem:
                        continue
                    sid = stem.split("__", 1)[0]
                    try:
                        entries.setdefault(sid, []).append(p)
                    except OSError:
                        continue
            idx = {"_built_at": time.time(), "entries": entries}
            _STAGE_INDEX[key] = idx
        return idx["entries"]


def _shared_cache_read(config: dict, stage: str, arxiv_id: str, model: str,
                       fingerprint: str | None = None) -> dict | None:
    """Read a cache entry. Exact-path lookup when the fingerprint is known
    (n8n contract); otherwise newest-first scan of this id's entries, preferring
    n8n-written files over mira-cli ones. Truncated/corrupt files are misses."""
    cache_dir = _shared_cache_dir(config, stage)
    if cache_dir is None or not cache_dir.is_dir():
        return None
    candidates: list[Path] = []
    safe = _safe_id(arxiv_id)
    idx = _stage_index(cache_dir, stage)
    # Same-id family: exact safe id plus version-suffixed variants
    # (n8n keeps 2608.11840v1; the CLI strips it). O(1) dict hits, not a
    # full-index walk (round-2 P2). When a fingerprint is known, ONLY
    # fingerprint-matching files are candidates — otherwise a prompt edit
    # would silently serve stale results (round-2 P1).
    family: list[Path] = []
    family.extend(idx.get(safe, []))
    # version-suffixed variants: sid == safe + "v" + digits (e.g. 2608.11840v2);
    # guard against prefix collisions (2608.1184 vs 2608.11840) via the digit check
    for sid, paths in idx.items():
        if sid.startswith(safe + "v"):
            suffix = sid[len(safe) + 1:]
            if suffix.isdigit():
                family.extend(paths)
    if fingerprint:
        exact = cache_dir / f"{safe}__{fingerprint}.json"
        if exact.is_file():
            candidates = [exact]
        else:
            candidates = [p for p in family if p.name.endswith(f"__{fingerprint}.json")]
    else:
        candidates = family

    def _rank(p: Path):
        # No fingerprint known: n8n-written entries first (they used n8n's exact
        # input pipeline), then newest. The family is small (a handful of files
        # per id), so the json parse for written_by is cheap.
        try:
            entry = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return (0, 0.0)
        by = 1 if entry.get("written_by") != "mira-cli" else 0
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (by, mtime)

    if fingerprint:
        # all candidates share the fingerprint; order by n8n-writer preference
        def _pref(p: Path):
            try:
                entry = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                return (0, 0.0)
            by = 1 if entry.get("written_by") != "mira-cli" else 0
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            return (by, mtime)
        candidates.sort(key=_pref, reverse=True)
    else:
        candidates.sort(key=_rank, reverse=True)
    for path in candidates:
        try:
            entry = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if entry.get("model") == model and isinstance(entry.get("result"), dict):
            return entry["result"]
    return None


def _stage_fingerprint(stage: str, model: str, prompt: str, system: str,
                       input_join: str) -> str:
    """Byte-identical port of n8n's Affiliation/Classification Cache Lookup:
    djb2 over 'stage|schema_version|model|prompt_hash|system_hash|input_hash'
    where *_hash are djb2 hex digests of the full rendered prompt/system text and
    the '\\n---\\n'-joined id/title/summary(/first_page_text) input."""
    prompt_hash = _djb2(prompt or "")
    system_hash = _djb2(system or "")
    input_hash = _djb2(input_join)
    return _djb2("|".join([stage, "1", model, prompt_hash, system_hash, input_hash]))


def _shared_cache_write(config: dict, stage: str, arxiv_id: str, model: str,
                        result: dict, fingerprint: str | None = None) -> None:
    cache_dir = _shared_cache_dir(config, stage)
    if cache_dir is None:
        return
    if fingerprint is None:
        fingerprint = hashlib.sha1(f"{stage}|{model}|{arxiv_id}".encode()).hexdigest()[:8]
    # n8n's Cache Lookup reads safeId(normalizeId(id)) — the VERSIONED form
    # (2608.11840v1). Writing under the bare id would be invisible to n8n
    # (round-2 P1): keep the version when the caller supplied raw_id.
    versioned = _normalize_id(arxiv_id) if arxiv_id.startswith("http") else None
    file_id = _safe_id(versioned) if versioned else _safe_id(arxiv_id)
    path = cache_dir / f"{file_id}__{fingerprint}.json"
    payload = {
        "arxiv_id": versioned or arxiv_id,
        "stage": stage,
        "cache_schema_version": 1,
        "model": model,
        "fingerprint": fingerprint,
        "written_by": "mira-cli",
        "result": result,
    }
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except OSError as e:
        print(f"  WARNING: could not write {stage} cache for {arxiv_id} — {e}")
        return
    # A fresh write must be visible to the scan path immediately (round-2 P2):
    # extend the in-memory index instead of waiting out the 300s TTL.
    with _STAGE_INDEX_LOCK:
        key = (str(cache_dir), stage)
        idx = _STAGE_INDEX.get(key)
        if idx is not None:
            sid = path.name[:-5].split("__", 1)[0]
            paths = idx["entries"].setdefault(sid, [])
            if path not in paths:
                paths.append(path)


def _cls_fingerprint(paper: dict, config: dict, user: str, system: str) -> tuple[Path, str, str]:
    """Compatibility API for scripts/reclassify_papers.py (PR #30 review r1):
    returns (cache_path, fingerprint, safe_id) for one paper's classification,
    using the same n8n-contract fingerprint as _classify_paper (including the
    full-URL id in the input hash — round-2 P1)."""
    model = config["llm_models"]["classification"]
    url_id = paper.get("raw_id") or f"http://arxiv.org/abs/{paper['id']}"
    input_join = "\n---\n".join([
        url_id,
        paper["title"], paper["summary"],
    ])
    fingerprint = _stage_fingerprint("classification", model, user, system, input_join)
    cache_dir = _shared_cache_dir(config, "classification") or (REPORT_FILES / "cache" / "default" / "classification")
    # n8n reads the VERSIONED safe id — mirror _shared_cache_write
    versioned = _normalize_id(url_id)
    return cache_dir / f"{_safe_id(versioned)}__{fingerprint}.json", fingerprint, _safe_id(versioned)


def _render_cls_prompts(paper: dict, config: dict) -> tuple[str, str]:
    """Compatibility API for scripts/reclassify_papers.py: renders the
    classification user/system prompts exactly as _classify_paper does
    (full-URL id into {{arxiv_id}} — round-2 P1)."""
    prompt = config["prompts"]["classification"]
    url_id = paper.get("raw_id") or f"http://arxiv.org/abs/{paper['id']}"
    user = _apply_template_n8n(prompt["user"], {
        "arxiv_id": url_id,
        "title": paper["title"],
        "summary": paper["summary"],
    })
    system = _apply_template_n8n(prompt.get("system", ""), {"topic_focus": config["topic"]["focus"]})
    return user, system


def _read_cache(cache_path: Path, fingerprint: str) -> dict | None:
    """Compatibility API for scripts/reclassify_papers.py: read a shared-cache
    entry by exact path, tolerant of truncated/corrupt files."""
    try:
        entry = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if entry.get("fingerprint") != fingerprint:
        return None
    return entry.get("result") if isinstance(entry.get("result"), dict) else None


def _get_affiliation(paper: dict, config: dict, client, cache: dict) -> dict:
    from mira.config import (apply_template, parse_json_response, llm_call,
                             model_for, record_parse_failure)

    arxiv_id = paper["id"]
    if arxiv_id in cache:
        _bump("aff_cache_hit")
        return cache[arxiv_id]

    model = model_for(config, "affiliation")
    prompt = config["prompts"]["affiliation"]
    system = prompt.get("system", "")
    # n8n's Build Affiliation Prompt substitutes data.id — the VERBATIM Atom <id>
    # (full URL, e.g. http://arxiv.org/abs/2608.11840v1) — into {{arxiv_id}}.
    # The CLI's bare version-stripped id would change the rendered prompt and
    # thus the fingerprint (round-2 P1). Fall back to http:// (arXiv Atom scheme).
    url_id = paper.get("raw_id") or f"http://arxiv.org/abs/{arxiv_id}"
    user = _apply_template_n8n(prompt["user"], {
        "arxiv_id": url_id,
        "title": paper["title"],
        "summary": paper["summary"],
        "authors": ", ".join(paper.get("authors", [])),
        "category": ", ".join(paper.get("categories", [])),
        "first_page_text": paper.get("first_page_text", ""),
        "topic_focus": config["topic"]["focus"],
    })
    # n8n's affiliation prompt builder does NOT substitute topic_focus into the
    # system template (system is used verbatim) — match that.
    fingerprint = _stage_fingerprint(
        "affiliation", model, user, system,
        "\n---\n".join([url_id,
                        paper["title"], paper["summary"], paper.get("first_page_text", "")]),
    )
    shared = _shared_cache_read(config, "affiliation", arxiv_id, model, fingerprint)
    if shared is not None:
        cache[arxiv_id] = shared
        _bump("aff_cache_hit")
        return shared

    _bump("aff_llm_call")
    raw = llm_call(client, model, system, user, reasoning_effort="high")

    try:
        result = parse_json_response(raw)
        if not isinstance(result, dict):
            raise ValueError("affiliation output is not a JSON object")
    except Exception:
        record_parse_failure("affiliation")
        # `_parse_error` tells classify_papers to treat the stage as failed:
        # n8n's structured parser errors the item out (error output unwired),
        # so the paper never gets a credibility_tier and the completeness gate
        # drops it. Not cached, so the next run retries.
        return {
            "affiliations": [], "author_affiliations": {},
            "credibility_tier": 1, "credibility_reasoning": "parse error",
            "_parse_error": True,
        }
    else:
        # n8n's "Distribute PDF Results" drops papers whose TRIMMED first-page
        # text is not > 50 chars; mirror exactly (trim + strict >) so the two
        # writers agree on what gets cached (round-2 P2).
        if len(paper.get("first_page_text", "").strip()) > 50:
            _shared_cache_write(config, "affiliation", url_id, model, result, fingerprint)

    cache[arxiv_id] = result
    return result


def _classify_paper(paper: dict, config: dict, client, cache: dict) -> dict:
    from mira.config import (apply_template, parse_json_response, llm_call,
                             model_for, record_parse_failure)

    arxiv_id = paper["id"]
    if arxiv_id in cache:
        _bump("cls_cache_hit")
        return cache[arxiv_id]

    model = model_for(config, "classification")
    prompt = config["prompts"]["classification"]
    system = _apply_template_n8n(prompt.get("system", ""), {"topic_focus": config["topic"]["focus"]})
    # n8n's "Build Classification Prompt" substitutes the FULL URL id (data.id)
    # into {{arxiv_id}} — match, or the fingerprint diverges (round-2 P1).
    url_id = paper.get("raw_id") or f"http://arxiv.org/abs/{arxiv_id}"
    user = _apply_template_n8n(prompt["user"], {
        "arxiv_id": url_id,
        "title": paper["title"],
        "summary": paper["summary"],
    })
    fingerprint = _stage_fingerprint(
        "classification", model, user, system,
        "\n---\n".join([url_id,
                        paper["title"], paper["summary"]]),
    )
    shared = _shared_cache_read(config, "classification", arxiv_id, model, fingerprint)
    if shared is not None:
        cache[arxiv_id] = shared
        _bump("cls_cache_hit")
        return shared

    _bump("cls_llm_call")
    raw = llm_call(client, model, system, user, reasoning_effort="high")

    try:
        result = parse_json_response(raw)
        if not isinstance(result, dict):
            raise ValueError("classification output is not a JSON object")
    except Exception:
        record_parse_failure("classification")
        # See _get_affiliation: flagged as a failed stage, not cached.
        return {"relevance_score": 0, "primary_topic": "Unknown", "credibility_tier": 1,
                "_parse_error": True}
    else:
        _shared_cache_write(config, "classification", url_id, model, result, fingerprint)

    cache[arxiv_id] = result
    return result


def _has_first_page_text(paper: dict) -> bool:
    """n8n's "Distribute PDF Results": only papers whose TRIMMED first-page
    text is > 50 chars reach the affiliation agent."""
    return len((paper.get("first_page_text") or "").strip()) > 50


def classify_papers(papers: list[dict], config: dict, client) -> list[dict]:
    """Affiliation + classification, with n8n's per-stage semantics:

    - affiliation runs only for papers with usable first-page text; the rest
      (and papers whose affiliation call/parse fails) get NO credibility_tier,
      so the completeness gate drops them — exactly as n8n does;
    - classification runs for every paper; a failed call/parse leaves the
      paper without relevance_score (dropped by the gate, excluded from stats);
    - stage outputs are normalized like n8n's Merge Affiliation Data /
      Normalize Classification Output (`value || default`)."""
    from mira.config import js_or

    aff_cache = _load_cache("affiliations")
    cls_cache = _load_cache("classifications")
    with _STATS_LOCK:
        _STATS.clear()

    def _classify_one(paper: dict) -> None:
        aff = None
        if _has_first_page_text(paper):
            try:
                aff = _get_affiliation(paper, config, client, aff_cache)
            except Exception as e:
                # One failed paper (e.g. provider 429 after retries) must not
                # abort the whole stage and discard the batch.
                _bump("stage_errors")
                print(f"  WARNING: affiliation failed for {paper['id']} — {e}. Skipping.")
            if aff is not None and aff.get("_parse_error"):
                aff = None
        else:
            _bump("aff_no_pdf_text")
        cls = None
        try:
            cls = _classify_paper(paper, config, client, cls_cache)
        except Exception as e:
            _bump("stage_errors")
            print(f"  WARNING: classify failed for {paper['id']} — {e}. Skipping.")
        if cls is not None and cls.get("_parse_error"):
            cls = None

        if aff is not None:
            paper.update({
                "affiliations": js_or(aff.get("affiliations"), []),
                "author_affiliations": js_or(aff.get("author_affiliations"), {}),
                "credibility_tier": js_or(aff.get("credibility_tier"), 0),
                "credibility_reasoning": js_or(aff.get("credibility_reasoning"), ""),
            })
        if cls is not None:
            paper.update({
                "primary_topic": js_or(cls.get("primary_topic"), ""),
                "secondary_topics": js_or(cls.get("secondary_topics"), []),
                "potential_impact": js_or(cls.get("potential_impact"), ""),
                "relevance_score": js_or(cls.get("relevance_score"), 0),
                "key_findings": js_or(cls.get("key_findings"), ""),
                "actionable": js_or(cls.get("actionable"), ""),
            })

    with ThreadPoolExecutor(max_workers=CLASSIFY_CONCURRENCY) as pool:
        list(pool.map(_classify_one, papers))

    _save_cache("affiliations", aff_cache)
    _save_cache("classifications", cls_cache)
    print(f"  Affiliation — cache hits: {_STATS['aff_cache_hit']}, LLM calls: {_STATS['aff_llm_call']}, "
          f"skipped (no first-page text): {_STATS['aff_no_pdf_text']}")
    print(f"  Classification — cache hits: {_STATS['cls_cache_hit']}, LLM calls: {_STATS['cls_llm_call']}")
    if _STATS["stage_errors"]:
        print(f"  WARNING: {_STATS['stage_errors']} stage call(s) failed and were skipped")
    from mira.config import parse_failure_report, format_parse_failures
    warning = format_parse_failures(parse_failure_report())
    if warning:
        print(f"  {warning}")
    return papers


def _to_int(v) -> int:
    """Coerce LLM-provided scores (often returned as JSON strings) to int."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Filter + rank (n8n: Score Completeness Gate, Filter by Relevance Score,
# Rank and Cap Filtered Papers, Format for Selection Agent, Build Selection
# Prompt). Papers are the CLI's flat dicts; the n8n `output.*` classification
# fields live at top level (relevance_score, primary_topic, ...).
# ---------------------------------------------------------------------------

class SelectionError(RuntimeError):
    """The run cannot continue past paper selection (n8n fails the execution)."""


class NoEligiblePapers(SelectionError):
    """No paper survived the completeness gate + thresholds. n8n simply stops
    (nothing downstream runs, no email); here it is an explicit, catchable
    error so a scheduler can tell "quiet period" from a real failure."""


def _normalize_arxiv_id(value) -> str:
    """Combine Report Data's normalizeId: strip the arxiv.org abs/pdf URL
    prefix and a .pdf suffix, keep the version, trim."""
    text = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", str(value or ""))
    return re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE).strip()


def _paper_url_id(paper: dict) -> str:
    """The id n8n carries for a paper: the verbatim Atom <id> URL."""
    return paper.get("raw_id") or paper.get("id") or ""


def _js_round1(value) -> float:
    """Math.round(v * 10) / 10 (JS rounds .5 up, not to even)."""
    import math
    return math.floor((value or 0) * 10 + 0.5) / 10


def _js_to_fixed(value: float, digits: int) -> float:
    """Number(value.toFixed(digits)): round the exact binary value half-up."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(value).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    out = float(q)
    return int(out) if out.is_integer() else out


def completeness_gate(papers: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    """Score Completeness Gate: keep papers whose relevance_score AND
    credibility_tier coerce to finite numbers (JS Number()); a missing key is
    `undefined` -> NaN -> dropped, while an explicit null is 0. Returns
    (valid papers with numeric scores + score_validation, dropped records)."""
    from mira.config import js_finite_number

    thresholds = config.get("thresholds") or {}
    thr_rel = js_finite_number(_threshold(thresholds, "relevance_score_min", 1))
    thr_cred = js_finite_number(_threshold(thresholds, "credibility_tier_min", 1))
    classified_count = sum(1 for p in papers if "relevance_score" in p)
    valid, dropped = [], []
    for paper in papers:
        rel = js_finite_number(paper["relevance_score"]) if "relevance_score" in paper else None
        cred = js_finite_number(paper["credibility_tier"]) if "credibility_tier" in paper else None
        if rel is None or cred is None:
            dropped.append({
                "arxiv_id": _normalize_arxiv_id(_paper_url_id(paper)),
                "missing_relevance_score": rel is None,
                "missing_credibility_tier": cred is None,
            })
            continue
        valid.append({
            **paper,
            "relevance_score": rel,
            "credibility_tier": cred,
            "score_validation": {
                "has_relevance_score": True,
                "has_credibility_tier": True,
                "dropped_count": 0,
                "threshold_relevance": thr_rel,
                "threshold_credibility": thr_cred,
                "originals_count": len(papers),
                "classified_count": classified_count,
            },
        })
    for paper in valid:
        paper["score_validation"]["dropped_count"] = len(dropped)
    if dropped:
        print(f"  [Score Completeness Gate] Dropped {len(dropped)} papers missing required "
              f"scores before threshold filtering.")
    return valid, dropped


def _threshold(thresholds: dict, key: str, default):
    """`cfg.thresholds?.key ?? default` (null and undefined both default)."""
    value = thresholds.get(key)
    return default if value is None else value


def apply_thresholds(valid: list[dict], config: dict) -> list[dict]:
    """Filter by Relevance Score on gate-validated papers: relevance_score >=
    relevance_score_min AND credibility_tier >= credibility_tier_min (each
    `?? 1`)."""
    from mira.config import js_number

    thresholds = config.get("thresholds") or {}
    min_rel = js_number(_threshold(thresholds, "relevance_score_min", 1))
    min_cred = js_number(_threshold(thresholds, "credibility_tier_min", 1))
    if min_rel is None or min_cred is None:  # NaN comparisons are always false
        return []
    return [p for p in valid if p["relevance_score"] >= min_rel and p["credibility_tier"] >= min_cred]


def filter_papers(papers: list[dict], config: dict) -> list[dict]:
    """Score Completeness Gate + Filter by Relevance Score."""
    valid, _ = completeness_gate(papers, config)
    return apply_thresholds(valid, config)


def rank_and_cap(papers: list[dict], config: dict) -> list[dict]:
    """Rank and Cap Filtered Papers: combined_score = relevance * w_rel +
    credibility * w_cred with the weights normalized to sum 1 (defaults
    0.6/0.4), rounded like toFixed(4); sort by combined, then relevance, then
    credibility (all descending, stable); keep the top max_filtered_papers
    (default 100, minimum 1)."""
    from mira.config import js_number, js_finite_number

    thresholds = config.get("thresholds") or {}
    max_papers = js_number(_threshold(thresholds, "max_filtered_papers", 100))
    w_rel_raw = js_number(_threshold(thresholds, "combined_relevance_weight", 0.6))
    w_cred_raw = js_number(_threshold(thresholds, "combined_credibility_weight", 0.4))
    if w_rel_raw is not None and w_cred_raw is not None and w_rel_raw + w_cred_raw > 0:
        weight_sum = w_rel_raw + w_cred_raw
        w_rel, w_cred = w_rel_raw / weight_sum, w_cred_raw / weight_sum
    else:  # sum <= 0 or NaN
        w_rel, w_cred = 0.6, 0.4

    def num(v) -> float:
        n = js_finite_number(v)
        return 0 if n is None else n

    ranked = []
    for paper in papers:
        combined = num(paper.get("relevance_score")) * w_rel + num(paper.get("credibility_tier")) * w_cred
        ranked.append({
            **paper,
            "combined_score": _js_to_fixed(combined, 4),
            "score_weights": {"relevance": _js_to_fixed(w_rel, 4),
                              "credibility": _js_to_fixed(w_cred, 4)},
        })
    ranked.sort(key=lambda p: (-num(p["combined_score"]), -num(p.get("relevance_score")),
                               -num(p.get("credibility_tier"))))
    # n8n: Math.max(1, NaN) is NaN and slice(0, NaN) is empty.
    limit = max(1, max_papers) if max_papers is not None else 0
    capped = ranked[:int(limit)] if limit != float("inf") else ranked
    print(f"  [Rank and Cap Filtered Papers] Input={len(papers)}, Output={len(capped)}, max={limit}")
    return capped


def format_for_selection(papers: list[dict]) -> list[dict]:
    """Format for Selection Agent: the per-paper object serialized into
    {{all_papers_json}}. Keys whose value would be `undefined` in n8n are
    omitted (JSON.stringify drops them)."""
    fields = (("title", "title"), ("authors", "authors"), ("affiliations", "affiliations"),
              ("credibility_tier", "credibility_tier"),
              ("credibility_reasoning", "credibility_reasoning"),
              ("primary_topic", "primary_topic"), ("secondary_topics", "secondary_topics"),
              ("potential_impact", "potential_impact"), ("relevance_score", "relevance_score"),
              ("key_findings", "key_findings"))
    out = []
    for p in papers:
        entry = {"arxiv_id": _paper_url_id(p)}
        for key, source in fields:
            if source in p:  # absent = undefined = omitted; None stays null
                entry[key] = p[source]
        # `summary || output.summary`: an empty abstract is undefined.
        if p.get("summary"):
            entry["abstract"] = p["summary"]
        out.append(entry)
    return out


def build_selection_prompt(pool: list[dict], config: dict) -> tuple[str, str]:
    """Build Selection Prompt: (user, system). Uses the node's own
    selection_range_label (no daily/weekly default for "0-0") and
    max_selection (0 when unset) — both differ from Set Run Mode's."""
    mode_cfg = config.get("mode_cfg") or {}
    prompts = (config.get("prompts") or {}).get("selection") or {}
    range_label = mode_cfg.get("selection_range_label") or \
        f"{mode_cfg.get('selection_min') or 0}-{mode_cfg.get('selection_max') or 0}"
    max_selection = mode_cfg.get("max_selection") or 0
    user = _apply_template_n8n(prompts.get("user") or "", {
        "period_label": config.get("period_label") or "recent period",
        "topic_focus": (config.get("topic") or {}).get("focus") or "research",
        "paper_count": len(pool),
        "all_papers_json": json.dumps(pool, indent=2, ensure_ascii=False),
        "selection_range_label": range_label,
        "max_selection": max_selection,
    })
    system = _apply_template_n8n(prompts.get("system") or "", {"max_selection": max_selection})
    return user, system


# ---------------------------------------------------------------------------
# Selection (n8n: Paper Selection Agent [retryOnFail, maxTries 3, 1 s wait],
# Validate Selection Output, Split Selected and Remaining Papers)
# ---------------------------------------------------------------------------

SELECTION_MAX_TRIES = 3
SELECTION_RETRY_WAIT = 1.0  # n8n's default waitBetweenTries


def _parse_json_object(text: str) -> dict:
    """Structured-output parsing: the whole reply, a ```json fenced block, or
    the outermost {...}; must be a JSON object."""
    from mira.config import parse_json_response
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text or "", flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    if text and "{" in text and "}" in text:
        candidates.append(text[text.index("{"):text.rindex("}") + 1])
    last: Exception | None = None
    for candidate in candidates:
        try:
            value = parse_json_response(candidate)
        except Exception as e:  # noqa: BLE001 — try the next extraction
            last = e
            continue
        if isinstance(value, dict):
            return value
        last = ValueError(f"expected a JSON object, got {type(value).__name__}")
    raise ValueError(f"could not parse a JSON object from the model output: {last}")


def validate_selection_output(output) -> dict:
    """Validate Selection Output, exactly: keep entries whose arxiv_id is a
    non-empty string; priority_rank = Number(rank) when finite, else the
    1-based position; raise when nothing valid was selected."""
    from mira.config import js_finite_number

    output = output if isinstance(output, dict) else {}

    def as_array(v):
        return v if isinstance(v, list) else []

    def as_text(v):
        return v.strip() if isinstance(v, str) else ""

    selected = [p for p in as_array(output.get("selected_papers"))
                if isinstance(p, dict) and as_text(p.get("arxiv_id"))]
    remaining = [p for p in as_array(output.get("remaining_papers"))
                 if isinstance(p, dict) and as_text(p.get("arxiv_id"))]
    if not selected:
        preview = json.dumps(output, ensure_ascii=False, separators=(",", ":"))[:800]
        raise SelectionError(
            f"Paper Selection Agent produced no valid selected_papers JSON. output preview: {preview}")

    def rank(p: dict, i: int):
        if "priority_rank" not in p:
            return i + 1
        n = js_finite_number(p["priority_rank"])
        return i + 1 if n is None else n

    return {
        "reasoning": as_text(output.get("reasoning")),
        "selected_papers": [{
            "arxiv_id": as_text(p.get("arxiv_id")),
            "selection_reasoning": as_text(p.get("selection_reasoning")),
            "priority_rank": rank(p, i),
        } for i, p in enumerate(selected)],
        "remaining_papers": [{
            "arxiv_id": as_text(p.get("arxiv_id")),
            "exclusion_reasoning": as_text(p.get("exclusion_reasoning")),
        } for p in remaining],
    }


def _run_selection_agent(user: str, system: str, config: dict, client) -> dict:
    from mira.config import llm_call, model_for, record_parse_failure

    model = model_for(config, "selection")
    last: Exception | None = None
    for attempt in range(SELECTION_MAX_TRIES):
        try:
            raw = llm_call(client, model, system, user, retries=1)
            return _parse_json_object(raw)
        except Exception as e:  # noqa: BLE001 — n8n retries the whole agent call
            if isinstance(e, ValueError):
                record_parse_failure("selection")
            last = e
            if attempt < SELECTION_MAX_TRIES - 1:
                time.sleep(SELECTION_RETRY_WAIT)
    raise SelectionError(f"Paper Selection Agent failed after {SELECTION_MAX_TRIES} tries: {last}")


def _match_selected(entries: list[dict], pool: list[dict]) -> tuple[list[dict], list[str]]:
    """Resolve validated selected_papers to pool papers, in the agent's order.

    n8n's Merge joins on exact string equality (selected arxiv_id == paper
    id, the full versioned URL). Here ids are compared after stripping the
    URL prefix, and a version-less id is accepted as a fallback, so a model
    that drops "http://arxiv.org/abs/" does not silently lose a paper (n8n
    would later synthesize a "deep analysis failed" placeholder for it).
    Duplicates are analyzed once; unknown ids are returned separately."""
    by_norm, by_base = {}, {}
    for paper in pool:
        norm = _normalize_arxiv_id(_paper_url_id(paper))
        by_norm.setdefault(norm, paper)
        by_base.setdefault(re.sub(r"v\d+$", "", norm), paper)
        by_base.setdefault(_normalize_arxiv_id(paper.get("id")), paper)
    matched, unknown, seen = [], [], set()
    for entry in entries:
        norm = _normalize_arxiv_id(entry["arxiv_id"])
        paper = by_norm.get(norm) or by_base.get(re.sub(r"v\d+$", "", norm))
        if paper is None:
            unknown.append(entry["arxiv_id"])
            continue
        key = id(paper)
        if key in seen:
            continue
        seen.add(key)
        matched.append({
            **paper,
            "arxiv_id": entry["arxiv_id"],
            "selection_reasoning": entry["selection_reasoning"],
            "priority_rank": entry["priority_rank"],
        })
    return matched, unknown


def select_papers_detailed(papers: list[dict], config: dict, client) -> dict:
    """Selection stage over the ranked pool. Returns
    {selected, remaining, output, unknown_ids, pool, prompt, system}.
    Raises NoEligiblePapers for an empty pool and SelectionError when the
    agent fails 3 times or selects nothing valid (n8n fails the run)."""
    if not papers:
        raise NoEligiblePapers(
            "No papers passed the completeness gate and relevance/credibility thresholds; "
            "nothing to select (n8n stops here without a report).")
    pool = format_for_selection(papers)
    user, system = build_selection_prompt(pool, config)
    output = validate_selection_output(_run_selection_agent(user, system, config, client))
    selected, unknown = _match_selected(output["selected_papers"], papers)
    if unknown:
        print(f"  WARNING: selection returned {len(unknown)} id(s) not in the candidate pool: "
              f"{unknown[:5]}")
    if not selected:
        raise SelectionError(
            f"Paper Selection Agent selected only ids that are not in the candidate pool: {unknown[:10]}")
    exclusion = {_normalize_arxiv_id(e["arxiv_id"]): e["exclusion_reasoning"]
                 for e in output["remaining_papers"]}
    chosen = {_normalize_arxiv_id(_paper_url_id(p)) for p in selected}
    remaining = []
    for paper in papers:
        norm = _normalize_arxiv_id(_paper_url_id(paper))
        if norm in chosen:
            continue
        remaining.append({**paper, "exclusion_reasoning": exclusion.get(norm, "")})
    return {"selected": selected, "remaining": remaining, "output": output,
            "unknown_ids": unknown, "pool": pool, "prompt": user, "system": system}


def select_papers(papers: list[dict], config: dict, client) -> tuple[list[dict], list[dict]]:
    """(selected, remaining). Selected papers carry arxiv_id (as returned by
    the agent), selection_reasoning and priority_rank; remaining papers carry
    exclusion_reasoning ('' when the agent did not list them)."""
    result = select_papers_detailed(papers, config, client)
    return result["selected"], result["remaining"]


# ---------------------------------------------------------------------------
# Deep analysis (n8n: Download Full PDF, Merge Full Text Data, Build Deep
# Analysis Prompt, Deep Analysis Agent [retryOnFail, maxTries 3, 10 s wait;
# batches of 5])
# ---------------------------------------------------------------------------

ANALYSIS_MAX_TRIES = 3
ANALYSIS_RETRY_WAIT = 10.0
ANALYSIS_CONCURRENCY = 5


def _analysis_max_chars() -> int:
    """n8n sends the FULL extracted text. MIRA_ANALYSIS_MAX_CHARS > 0 caps it
    (the old CLI hard-coded 50 000); unset/0 = no cap (n8n parity)."""
    try:
        return max(0, int(os.environ.get("MIRA_ANALYSIS_MAX_CHARS", "0") or 0))
    except ValueError:
        return 0


def fetch_full_text(paper: dict) -> dict:
    """Download Full PDF + Merge Full Text Data: full_text, page_count,
    pdf_download_success, pdf_error for one selected paper. The download id is
    the agent's arxiv_id (full versioned URL), like n8n, so the full-text
    cache is shared with it."""
    import download_full_arxiv_pdf as m  # type: ignore
    download_id = paper.get("arxiv_id") or _paper_url_id(paper)
    try:
        result = m.download_and_extract(download_id)
        if not isinstance(result, dict):
            raise ValueError("unexpected download result")
    except Exception as e:  # noqa: BLE001 — n8n records the failure and continues
        return {"full_text": "", "page_count": 0, "pdf_download_success": False,
                "pdf_error": f"Failed to parse PDF extraction output: {e}"}
    return {
        "full_text": result.get("full_text") or "",
        "page_count": result.get("page_count") or 0,
        "pdf_download_success": bool(result.get("success") or False),
        "pdf_error": result.get("error") or "",
    }


def build_analysis_prompt(paper: dict, config: dict, full_text: str,
                          pdf_download_success: bool) -> tuple[str, str]:
    """Build Deep Analysis Prompt: (user, system). The system prompt is used
    verbatim (no substitution), like the node."""
    prompts = (config.get("prompts") or {}).get("analysis") or {}
    authors = paper.get("authors")
    if isinstance(authors, list):
        authors = ", ".join("" if a is None else str(a) for a in authors)
    else:
        authors = "" if authors is None else authors
    max_chars = _analysis_max_chars()
    if max_chars and len(full_text) > max_chars:
        full_text = full_text[:max_chars]
    user = _apply_template_n8n(prompts.get("user") or "", {
        "arxiv_id": paper.get("arxiv_id") or _paper_url_id(paper),
        "title": paper.get("title"),
        "authors": authors,
        "primary_topic": paper.get("primary_topic") or "",
        "selection_reasoning": paper.get("selection_reasoning") or "",
        "full_text": full_text or "",
        "pdf_analysis_performed": "true" if pdf_download_success else "false",
    })
    return user, prompts.get("system") or ""


def _analyze_paper(paper: dict, config: dict, client) -> dict:
    """Download + analyze one selected paper (3 tries, 10 s apart). On final
    failure the paper is kept with analysis_failed=True, analysis_error and
    empty summaries (n8n's report stage synthesizes a fallback for it)."""
    from mira.config import llm_call, model_for, parse_bool, record_parse_failure

    model = model_for(config, "analysis")
    pdf = fetch_full_text(paper)
    user, system = build_analysis_prompt(paper, config, pdf["full_text"], pdf["pdf_download_success"])
    paper.update({k: v for k, v in pdf.items() if k != "full_text"})

    result, last = None, None
    for attempt in range(ANALYSIS_MAX_TRIES):
        try:
            raw = llm_call(client, model, system, user, retries=1)
            result = _parse_json_object(raw)
            break
        except Exception as e:  # noqa: BLE001 — retryOnFail covers any agent error
            if isinstance(e, ValueError):
                record_parse_failure("analysis")
            last = e
            if attempt < ANALYSIS_MAX_TRIES - 1:
                time.sleep(ANALYSIS_RETRY_WAIT)

    if result is None:
        paper.update({"large_summary": "", "short_summary": "", "pdf_analysis_performed": False,
                      "analysis_failed": True, "analysis_error": str(last)})
        return paper

    large = result.get("large_summary") or result.get("short_summary") or ""
    short = result.get("short_summary") or result.get("large_summary") or ""
    performed = result.get("pdf_analysis_performed")
    if performed is None:
        performed = pdf["pdf_download_success"]
    elif not isinstance(performed, bool):
        performed = parse_bool(performed, False)
    analysis_id = result.get("arxiv_id") if isinstance(result.get("arxiv_id"), str) else ""
    if analysis_id and _normalize_arxiv_id(analysis_id) != _normalize_arxiv_id(
            paper.get("arxiv_id") or _paper_url_id(paper)):
        # n8n merges the analysis back on this id and would drop a mismatch.
        print(f"  WARNING: analysis for {paper.get('arxiv_id')} came back as {analysis_id}")
    paper.update({"large_summary": large, "short_summary": short,
                  "pdf_analysis_performed": performed, "analysis_arxiv_id": analysis_id,
                  "analysis_failed": False})
    return paper


def analyze_papers(selected: list[dict], config: dict, client) -> list[dict]:
    """Analyze every selected paper with ANALYSIS_CONCURRENCY workers (n8n
    batches 5 at a time). Order is preserved; failures are flagged, not
    dropped."""
    from mira.config import model_for

    model_for(config, "analysis")  # a missing model is a config error, not a per-paper one

    def one(paper: dict) -> dict:
        try:
            return _analyze_paper(paper, config, client)
        except Exception as e:  # noqa: BLE001 — one paper must not sink the batch
            print(f"  WARNING: Failed to analyze {paper.get('id')} — {e}. Keeping it unanalyzed.")
            paper.update({"large_summary": paper.get("large_summary", ""),
                          "short_summary": paper.get("short_summary", ""),
                          "pdf_analysis_performed": False, "analysis_failed": True,
                          "analysis_error": str(e)})
            return paper

    if not selected:
        return []
    with ThreadPoolExecutor(max_workers=min(ANALYSIS_CONCURRENCY, len(selected))) as pool:
        return list(pool.map(one, selected))


# ---------------------------------------------------------------------------
# Statistics (n8n: Compute Paper Statistics)
# ---------------------------------------------------------------------------

def compute_paper_statistics(classified: list[dict], abstracts_analyzed: int) -> dict:
    """Compute Paper Statistics over every classified paper (the n8n "Merge
    Classification Results1" set: classification succeeded, before the gate).
    `abstracts_analyzed` is the deduplicated arXiv count."""
    from mira.config import js_finite_number

    rel = [n for n in (js_finite_number(p.get("relevance_score")) for p in classified) if n is not None]
    cred = [n for n in (js_finite_number(p["credibility_tier"]) if "credibility_tier" in p else None
                        for p in classified) if n is not None]
    topics: dict[str, int] = {}
    for p in classified:
        topic = p.get("primary_topic") or "Unknown"
        topics[topic] = topics.get(topic, 0) + 1
    impact = {"Breakthrough": 0, "High": 0, "Medium": 0, "Low": 0}
    for p in classified:
        value = p.get("potential_impact") or "Unknown"
        if isinstance(value, str) and value in impact:
            impact[value] += 1

    def high(p: dict) -> bool:
        n = js_finite_number(p["credibility_tier"]) if "credibility_tier" in p else None
        return n is not None and n >= 8

    return {
        "abstracts_analyzed": abstracts_analyzed,
        "total_relevant_papers": len(classified),
        "avg_relevance_score": _js_round1(sum(rel) / len(rel) if rel else 0),
        "avg_credibility_tier": _js_round1(sum(cred) / len(cred) if cred else 0),
        "high_credibility_count": sum(1 for p in classified if high(p)),
        "topic_distribution": [{"topic": t, "count": c} for t, c in
                               sorted(topics.items(), key=lambda kv: -kv[1])[:10]],
        "impact_distribution": impact,
    }


def run_pipeline(papers: list[dict], config: dict, client) -> dict:
    """classify -> completeness gate -> thresholds -> rank/cap -> select ->
    deep analysis, following the live n8n workflow node by node.

    Returns:
      selected         selected papers in the agent's order, analyzed (flat
                       paper dicts + arxiv_id, selection_reasoning,
                       priority_rank, large_summary, short_summary,
                       pdf_analysis_performed, page_count,
                       pdf_download_success, pdf_error, analysis_failed,
                       analysis_error?, combined_score, score_weights)
      remaining        ranked candidates not selected (+ exclusion_reasoning)
      total_scanned    papers received (deduplicated arXiv results)
      classified       papers whose classification succeeded (stats input)
      ranked           the capped, ranked candidate pool (selection input)
      selection_pool   the per-paper objects sent to the selection agent
      selection        validated agent output {reasoning, selected_papers,
                       remaining_papers}
      unknown_selected_ids  agent ids that matched no candidate
      gate_dropped     completeness-gate drop records
      stats            Compute Paper Statistics output
    Raises NoEligiblePapers / SelectionError (see select_papers_detailed)."""
    total = len(papers)
    papers = classify_papers(papers, config, client)
    classified = [p for p in papers if "relevance_score" in p]
    stats = compute_paper_statistics(classified, total)
    valid, dropped = completeness_gate(papers, config)
    filtered = apply_thresholds(valid, config)
    ranked = rank_and_cap(filtered, config)
    selection = select_papers_detailed(ranked, config, client)
    analyzed = analyze_papers(selection["selected"], config, client)
    return {
        "selected": analyzed,
        "remaining": selection["remaining"],
        "total_scanned": total,
        "classified": classified,
        "ranked": ranked,
        "selection_pool": selection["pool"],
        "selection": selection["output"],
        "unknown_selected_ids": selection["unknown_ids"],
        "gate_dropped": dropped,
        "stats": stats,
    }

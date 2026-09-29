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
    directions. No profile_id (unit tests) or MIRA_LLM_CACHE_BYPASS=1 disables it."""
    profile = config.get("profile_id")
    if not profile or os.environ.get("MIRA_LLM_CACHE_BYPASS") == "1":
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
    except Exception:
        record_parse_failure("affiliation")
        result = {
            "affiliations": [], "author_affiliations": {},
            "credibility_tier": 1, "credibility_reasoning": "parse error",
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
    except Exception:
        record_parse_failure("classification")
        result = {"relevance_score": 0, "primary_topic": "Unknown", "credibility_tier": 1}
    else:
        _shared_cache_write(config, "classification", url_id, model, result, fingerprint)

    cache[arxiv_id] = result
    return result


def classify_papers(papers: list[dict], config: dict, client) -> list[dict]:
    aff_cache = _load_cache("affiliations")
    cls_cache = _load_cache("classifications")
    with _STATS_LOCK:
        _STATS.clear()

    def _classify_one(paper: dict) -> None:
        try:
            aff = _get_affiliation(paper, config, client, aff_cache)
            cls = _classify_paper(paper, config, client, cls_cache)
        except Exception as e:
            # One failed paper (e.g. provider 429 after retries) must not abort
            # the whole stage and discard the batch — mirror analyze_papers().
            _bump("stage_errors")
            print(f"  WARNING: classify failed for {paper['id']} — {e}. Skipping.")
            return

        paper.update({
            "affiliations": aff.get("affiliations", []),
            "author_affiliations": aff.get("author_affiliations", {}),
            "credibility_tier": aff.get("credibility_tier", 1),
            "credibility_reasoning": aff.get("credibility_reasoning", ""),
            "primary_topic": cls.get("primary_topic", "Unknown"),
            "secondary_topics": cls.get("secondary_topics", []),
            "relevance_score": cls.get("relevance_score", 0),
            "key_findings": cls.get("key_findings", ""),
            "actionable": cls.get("actionable", "No"),
        })

    with ThreadPoolExecutor(max_workers=CLASSIFY_CONCURRENCY) as pool:
        list(pool.map(_classify_one, papers))

    _save_cache("affiliations", aff_cache)
    _save_cache("classifications", cls_cache)
    print(f"  Affiliation — cache hits: {_STATS['aff_cache_hit']}, LLM calls: {_STATS['aff_llm_call']}")
    print(f"  Classification — cache hits: {_STATS['cls_cache_hit']}, LLM calls: {_STATS['cls_llm_call']}")
    if _STATS["stage_errors"]:
        print(f"  WARNING: {_STATS['stage_errors']} paper(s) failed classification and were skipped")
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


def filter_papers(papers: list[dict], config: dict) -> list[dict]:
    min_rel = config["thresholds"]["relevance_score_min"]
    min_cred = config["thresholds"]["credibility_tier_min"]
    return [
        p for p in papers
        if _to_int(p.get("relevance_score", 0)) >= min_rel
        and _to_int(p.get("credibility_tier", 0)) >= min_cred
    ]


def select_papers(papers: list[dict], config: dict, client) -> tuple[list[dict], list[dict]]:
    from mira.config import (apply_template, parse_json_response, llm_call,
                             model_for, record_parse_failure)

    mode_cfg = config["mode_cfg"]
    prompt = config["prompts"]["selection"]
    user = apply_template(prompt["user"], {
        "period_label": config["mode"],
        "topic_focus": config["topic"]["focus"],
        "paper_count": str(len(papers)),
        "all_papers_json": json.dumps([
            {
                "arxiv_id": p["id"], "title": p["title"],
                "relevance_score": p.get("relevance_score", 0),
                "credibility_tier": p.get("credibility_tier", 0),
                "primary_topic": p.get("primary_topic", ""),
                "key_findings": p.get("key_findings", ""),
            }
            for p in papers
        ], indent=2),
        "selection_range_label": mode_cfg.get("selection_range_label", "5-10"),
    })
    model = model_for(config, "selection")
    raw = llm_call(client, model, prompt.get("system", ""), user)

    try:
        result = parse_json_response(raw)
        selected_ids = {p["arxiv_id"] for p in result.get("selected_papers", [])}
        reasoning_map = {p["arxiv_id"]: p.get("reasoning", "") for p in result.get("selected_papers", [])}
    except Exception:
        record_parse_failure("selection")
        n = mode_cfg.get("selection_max", 10)
        sorted_p = sorted(papers, key=lambda p: _to_int(p.get("relevance_score", 0)), reverse=True)
        selected_ids = {p["id"] for p in sorted_p[:n]}
        reasoning_map = {}

    selected, remaining = [], []
    for p in papers:
        if p["id"] in selected_ids:
            p["selection_reasoning"] = reasoning_map.get(p["id"], "")
            selected.append(p)
        else:
            remaining.append(p)
    return selected, remaining


def _analyze_paper(paper: dict, config: dict, client) -> dict:
    from mira.config import (apply_template, parse_json_response, llm_call,
                             model_for, record_parse_failure)
    import download_full_arxiv_pdf as m  # type: ignore

    pdf_result = m.download_and_extract(paper["id"])
    full_text = pdf_result.get("full_text", "")[:50000]
    pdf_ok = pdf_result.get("success", False)

    prompt = config["prompts"]["analysis"]
    model = model_for(config, "analysis")
    user = apply_template(prompt["user"], {
        "arxiv_id": paper["id"],
        "title": paper["title"],
        "authors": ", ".join(paper.get("authors", [])),
        "primary_topic": paper.get("primary_topic", ""),
        "selection_reasoning": paper.get("selection_reasoning", ""),
        "full_text": full_text,
    })

    try:
        raw = llm_call(client, model, prompt.get("system", ""), user)
        result = parse_json_response(raw)
    except Exception:
        record_parse_failure("analysis")
        result = {"large_summary": "", "short_summary": ""}

    paper["large_summary"] = result.get("large_summary", "")
    paper["short_summary"] = result.get("short_summary", "")
    paper["pdf_analysis_performed"] = pdf_ok
    return paper


def analyze_papers(selected: list[dict], config: dict, client) -> list[dict]:
    results = []
    for paper in selected:
        try:
            results.append(_analyze_paper(paper, config, client))
        except Exception as e:
            print(f"  WARNING: Failed to analyze {paper['id']} — {e}. Skipping.")
            paper.setdefault("large_summary", "")
            paper.setdefault("short_summary", "")
            paper.setdefault("pdf_analysis_performed", False)
            results.append(paper)
    return results


def run_pipeline(papers: list[dict], config: dict, client) -> dict:
    papers = classify_papers(papers, config, client)
    filtered = filter_papers(papers, config)
    selected, remaining = select_papers(filtered, config, client)
    analyzed = analyze_papers(selected, config, client)
    return {"selected": analyzed, "remaining": remaining, "total_scanned": len(papers)}

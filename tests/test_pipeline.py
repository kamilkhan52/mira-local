# tests/test_pipeline.py
from _paths import repoint
import json
import os
import time
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from mira.pipeline import _load_cache, _save_cache, _classify_paper, _get_affiliation


def test_load_cache_returns_empty_dict_when_missing(tmp_path, monkeypatch):
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    assert _load_cache("classifications") == {}


def test_save_and_load_cache_roundtrip(tmp_path, monkeypatch):
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    # no manual mkdir — _save_cache must create the directory
    _save_cache("classifications", {"2605.11277": {"relevance_score": 8}})
    loaded = _load_cache("classifications")
    assert loaded["2605.11277"]["relevance_score"] == 8


from mira.pipeline import _safe_id, _shared_cache_read, _shared_cache_write


def _shared_cache_config():
    return {
        "profile_id": "optical-interconnects",
        "prompts": {
            "classification": {"user": "Classify {{arxiv_id}}", "system": ""},
            "affiliation": {"user": "Affiliate {{arxiv_id}}: {{authors}} {{category}} {{first_page_text}}", "system": ""},
        },
        "llm_models": {"classification": "google/gemini-3.6-flash",
                       "affiliation": "google/gemini-3.6-flash"},
        "topic": {"focus": "optics"},
    }


def test_safe_id_matches_n8n_mapping():
    assert _safe_id("http://arxiv.org/abs/2608.11840v1") == "2608.11840v1"
    assert _safe_id("http://arxiv.org/pdf/cs/0701001.pdf") == "cs_0701001"


def test_shared_cache_roundtrip(tmp_path, monkeypatch):
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    config = _shared_cache_config()
    _shared_cache_write(config, "classification", "http://arxiv.org/abs/2608.11840v1",
                        "google/gemini-3.6-flash", {"relevance_score": 8})
    hit = _shared_cache_read(config, "classification", "http://arxiv.org/abs/2608.11840v1",
                             "google/gemini-3.6-flash")
    assert hit == {"relevance_score": 8}


def test_shared_cache_reads_n8n_written_entry(tmp_path, monkeypatch):
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    stage_dir = tmp_path / "report-files" / "cache" / "optical-interconnects" / "classification"
    stage_dir.mkdir(parents=True)
    # n8n's filename is <safe_id>__<its own djb2 fingerprint>.json — we match on the prefix
    (stage_dir / "2608.11840v1__45980a4e.json").write_text(json.dumps({
        "arxiv_id": "http://arxiv.org/abs/2608.11840v1",
        "model": "google/gemini-3.6-flash",
        "result": {"relevance_score": 5, "primary_topic": "Optical Switching"},
    }))
    config = _shared_cache_config()
    hit = _shared_cache_read(config, "classification", "http://arxiv.org/abs/2608.11840v1",
                             "google/gemini-3.6-flash")
    assert hit["primary_topic"] == "Optical Switching"
    # a model swap must not serve the old digest
    assert _shared_cache_read(config, "classification", "http://arxiv.org/abs/2608.11840v1",
                              "anthropic/claude-opus-4.8") is None


def test_shared_cache_matches_versioned_n8n_filename(tmp_path, monkeypatch):
    """fetch.py strips the version off ids; n8n's filenames keep it."""
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    stage_dir = tmp_path / "report-files" / "cache" / "optical-interconnects" / "classification"
    stage_dir.mkdir(parents=True)
    (stage_dir / "2608.11840v2__45980a4e.json").write_text(json.dumps({
        "model": "google/gemini-3.6-flash", "result": {"relevance_score": 6},
    }))
    config = _shared_cache_config()
    assert _shared_cache_read(config, "classification", "2608.11840",
                              "google/gemini-3.6-flash") == {"relevance_score": 6}
    # a shorter id must not collide with a longer one
    assert _shared_cache_read(config, "classification", "2608.1184",
                              "google/gemini-3.6-flash") is None


def test_shared_cache_disabled_without_profile_id(tmp_path, monkeypatch):
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    config = {**_shared_cache_config()}
    del config["profile_id"]
    _shared_cache_write(config, "classification", "2608.11840v1", "m", {"relevance_score": 8})
    assert not (tmp_path / "report-files").exists()
    assert _shared_cache_read(config, "classification", "2608.11840v1", "m") is None


def test_shared_cache_bypass_env(tmp_path, monkeypatch):
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    monkeypatch.setenv("MIRA_LLM_CACHE_BYPASS", "1")
    config = _shared_cache_config()
    assert _shared_cache_read(config, "classification", "2608.11840v1",
                              "google/gemini-3.6-flash") is None


# ---- n8n-contract interop tests (review round 1 fixes) ----

def _djb2_of(s: str) -> str:
    from mira.pipeline import _djb2
    return _djb2(s)


def _n8n_entry(tmp_path, arxiv_id_url, model, result, written_by=None, fp=None):
    """Write a cache entry shaped exactly like n8n's Write * Cache nodes."""
    from mira.pipeline import _safe_id, _djb2, _stage_fingerprint
    stage_dir = tmp_path / "report-files" / "cache" / "optical-interconnects" / "classification"
    stage_dir.mkdir(parents=True, exist_ok=True)
    if fp is None:
        prompt = f"Classify {{arxiv_id}}: {arxiv_id_url}"
        system = "You are a research classifier."
        input_join = f"{arxiv_id_url}\nPaper T\nAbstract S"
        ph, sh, ih = _djb2(prompt), _djb2(system), _djb2(input_join)
        fp = _djb2("|".join(["classification", "1", model, ph, sh, ih]))
    payload = {
        "arxiv_id": arxiv_id_url,
        "stage": "classification",
        "cache_schema_version": 1,
        "created_at": "2026-09-02T00:00:00.000Z",
        "model": model,
        "prompt_hash": _djb2(f"Classify {arxiv_id_url}"),
        "system_hash": _djb2("You are a research classifier."),
        "input_hash": _djb2(f"{arxiv_id_url}\nPaper T\nAbstract S"),
        "fingerprint": fp,
        "result": result,
    }
    (stage_dir / f"{_safe_id(arxiv_id_url)}__{fp}.json").write_text(json.dumps(payload))
    return fp


def test_fingerprint_interops_with_n8n_shaped_entry(tmp_path, monkeypatch):
    """A file written in n8n's exact on-disk shape must be readable, and the
    exact-path lookup must hit when the caller computes the fingerprint."""
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    model = "openai/gpt-5.6-terra"
    url = "https://arxiv.org/abs/2608.11840v1"
    # compute the fingerprint the way _classify_paper does
    from mira.pipeline import _stage_fingerprint
    prompt = f"Classify {{arxiv_id}}: {url}"
    system = "You are a research classifier."
    input_join = f"{url}\nPaper T\nAbstract S"
    fp = _stage_fingerprint("classification", model, prompt, "You are a research classifier.", input_join)
    _n8n_entry(tmp_path, url, model, {"relevance_score": 7, "primary_topic": "HBM"}, fp=fp)
    # patch _n8n_entry's write target (it used _shared_cache_config's profile) —
    # simpler: write again under the config's dir
    config = _shared_cache_config()
    from mira.pipeline import _shared_cache_dir
    stage_dir = _shared_cache_dir(config, "classification")
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / f"2608.11840v1__{fp}.json").write_text(json.dumps({
        "arxiv_id": url, "stage": "classification", "cache_schema_version": 1,
        "model": model, "fingerprint": fp,
        "prompt_hash": "x", "system_hash": "y", "input_hash": "z",
        "result": {"relevance_score": 7, "primary_topic": "HBM"},
    }))
    hit = _shared_cache_read(config, "classification", url, model, fp)
    assert hit == {"relevance_score": 7, "primary_topic": "HBM"}


def test_cli_prefers_n8n_written_entry_on_fingerprint_collision(tmp_path, monkeypatch):
    """Without a known fingerprint, the scan prefers n8n-written entries even
    when the mira-cli entry is newer (round-1 P1-2)."""
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    config = _shared_cache_config()
    from mira.pipeline import _shared_cache_dir, _safe_id
    stage_dir = _shared_cache_dir(config, "classification")
    stage_dir.mkdir(parents=True)
    # mira-cli entry, NEWER
    cli_path = stage_dir / f"{_safe_id('2608.11840v1')}__deadbeef.json"
    cli_path.write_text(json.dumps({
        "model": "google/gemini-3.6-flash", "fingerprint": "deadbeef",
        "written_by": "mira-cli", "result": {"relevance_score": 1, "primary_topic": "CLI"},
    }))
    os.utime(cli_path, (time.time(), time.time()))
    # n8n entry (no written_by field), OLDER — must still win the scan
    n8n_path = stage_dir / f"{_safe_id('2608.11840v1')}__cafebabe.json"
    n8n_path.write_text(json.dumps({
        "model": "google/gemini-3.6-flash", "fingerprint": "cafebabe",
        "result": {"relevance_score": 9, "primary_topic": "N8N"},
    }))
    os.utime(n8n_path, (time.time() - 100, time.time() - 100))
    hit = _shared_cache_read(config, "classification", "2608.11840v1", "google/gemini-3.6-flash")
    assert hit["primary_topic"] == "N8N"


def test_classify_write_uses_versioned_filename_from_bare_id_path(tmp_path, monkeypatch):
    """Round-3 P1: production passes the BARE id in paper['id']; the shared
    write must still land on n8n's versioned filename form
    (safeId(normalizeId(data.id)) = 2608.11840v1__<fp>.json)."""
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    from mira.pipeline import _classify_paper

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return types.SimpleNamespace(choices=[types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=json.dumps({"relevance_score": 7, "primary_topic": "HBM"})))])

    config = _shared_cache_config()
    paper = {"id": "2608.11840", "raw_id": "http://arxiv.org/abs/2608.11840v1",
             "title": "T", "summary": "S", "categories": ["cs.AR"],
             "first_page_text": "x" * 60}
    _classify_paper(paper, config, FakeClient(), {})
    cls_dir = tmp_path / "report-files" / "cache" / "optical-interconnects" / "classification"
    files = [p.name for p in cls_dir.glob("*.json")]
    assert files, "cache entry must be written"
    assert any(name.startswith("2608.11840v1__") for name in files), \
        f"expected VERSIONED filename, got {files}"


def test_affiliation_write_gated_on_first_page_text(tmp_path, monkeypatch):
    """Mirrors n8n's <50-char Distribute gate: short-text results are NOT shared."""
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    config = _shared_cache_config()
    from mira.pipeline import _get_affiliation

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    class R: ...
                    r = types.SimpleNamespace()
                    r.choices = [types.SimpleNamespace(message=types.SimpleNamespace(
                        content=json.dumps({"affiliations": [], "author_affiliations": {},
                                            "credibility_tier": 1, "credibility_reasoning": "x"})))]
                    return r

    paper = {"id": "2608.11840", "title": "T", "summary": "S", "authors": ["A"],
             "categories": ["cs.AR"], "first_page_text": "short"}
    cache = {}
    _get_affiliation(paper, config, FakeClient(), cache)
    aff_dir = tmp_path / "report-files" / "cache" / "optical-interconnects" / "affiliation"
    assert aff_dir.exists() is False or list(aff_dir.glob("*.json")) == []
    # long text passes the gate — different id so the in-memory cache doesn't
    # serve the first result (the gate is on the shared-cache WRITE path)
    paper2 = {**paper, "id": "2608.11841", "first_page_text": "x" * 60}
    _get_affiliation(paper2, config, FakeClient(), cache)
    assert list(aff_dir.glob("*.json"))


def test_classify_papers_survives_worker_exception(tmp_path, monkeypatch):
    """One failing paper must not abort the stage (round-1 P1-4)."""
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    from mira.pipeline import classify_papers
    calls = {"n": 0}

    class FlakyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    if "2608.BOOM" in kw["messages"][1]["content"]:
                        raise RuntimeError("simulated provider 429 after retries")
                    class R:
                        choices = [types.SimpleNamespace(message=types.SimpleNamespace(
                            content=json.dumps({"relevance_score": 7, "primary_topic": "OK"})))]
                    return R

    papers = [{"id": f"2608.{i:05d}", "title": f"T{i}", "summary": "S",
               "categories": ["cs.AR"], "first_page_text": "x" * 60} for i in range(6)]
    papers[3]["id"] = "2608.BOOM"
    out = classify_papers(papers, _shared_cache_config(), FlakyClient())
    assert len(out) == 6
    failed = [p for p in out if not p.get("primary_topic")]
    assert len(failed) == 1 and failed[0]["id"] == "2608.BOOM"


def test_stage_index_avoids_refixting_every_lookup(tmp_path, monkeypatch):
    """The per-stage index is built once and reused (perf contract)."""
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    from mira.pipeline import _stage_index
    config = _shared_cache_config()
    from mira.pipeline import _shared_cache_dir
    stage_dir = _shared_cache_dir(config, "classification")
    stage_dir.mkdir(parents=True)
    for i in range(30):
        (stage_dir / f"2608.{i:05d}v1__fp{i}.json").write_text(json.dumps({"model": "m", "result": {}}))
    d = _shared_cache_dir(config, "classification")
    idx1 = _stage_index(d, "classification")
    assert len(idx1) == 30
    # second call within TTL returns the SAME dict object (no rescan)
    assert _stage_index(d, "classification") is idx1


def test_classify_paper_returns_cached_without_llm_call():
    cache = {"2605.11277": {"relevance_score": 9, "primary_topic": "HBM"}}
    paper = {"id": "2605.11277", "title": "T", "summary": "S", "categories": ["cs.AR"]}
    config = {
        "prompts": {"classification": {"user": "", "system": ""}},
        "llm_models": {"classification": "model"},
        "topic": {"focus": "memory technology"},
    }
    client = MagicMock()
    result = _classify_paper(paper, config, client, cache)
    client.chat.completions.create.assert_not_called()
    assert result["relevance_score"] == 9


def test_get_affiliation_returns_cached_without_llm_call():
    cache = {"2605.11277": {"credibility_tier": 9, "affiliations": ["Stanford"]}}
    paper = {"id": "2605.11277", "title": "T", "summary": "S",
             "authors": ["A"], "categories": ["cs.AR"], "first_page_text": ""}
    config = {
        "prompts": {"affiliation": {"user": "", "system": ""}},
        "llm_models": {"affiliation": "model"},
        "topic": {"focus": "memory technology"},
    }
    client = MagicMock()
    result = _get_affiliation(paper, config, client, cache)
    client.chat.completions.create.assert_not_called()
    assert result["credibility_tier"] == 9


def test_classify_paper_calls_llm_on_cache_miss():
    cache = {}
    paper = {"id": "2605.11277", "title": "T", "summary": "S", "categories": ["cs.AR"]}
    config = {
        "prompts": {"classification": {"user": "Classify {{arxiv_id}}", "system": ""}},
        "llm_models": {"classification": "google/gemini-3.1-pro-preview"},
        "topic": {"focus": "memory technology"},
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = (
        '{"relevance_score": 7, "primary_topic": "HBM", "credibility_tier": 8}'
    )
    result = _classify_paper(paper, config, mock_client, cache)
    mock_client.chat.completions.create.assert_called_once()
    assert result["relevance_score"] == 7
    assert "2605.11277" in cache


from mira.pipeline import filter_papers, select_papers, run_pipeline


def test_filter_papers_removes_below_threshold():
    papers = [
        {"id": "1", "relevance_score": 7, "credibility_tier": 6},
        {"id": "2", "relevance_score": 3, "credibility_tier": 8},  # relevance too low
        {"id": "3", "relevance_score": 6, "credibility_tier": 3},  # credibility too low
    ]
    config = {"thresholds": {"relevance_score_min": 5, "credibility_tier_min": 5}}
    result = filter_papers(papers, config)
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_filter_papers_keeps_all_above_threshold():
    papers = [
        {"id": "1", "relevance_score": 8, "credibility_tier": 9},
        {"id": "2", "relevance_score": 5, "credibility_tier": 5},
    ]
    config = {"thresholds": {"relevance_score_min": 5, "credibility_tier_min": 5}}
    assert len(filter_papers(papers, config)) == 2


def test_select_papers_splits_selected_and_remaining():
    papers = [
        {"id": "2605.001", "title": "P1", "relevance_score": 9, "credibility_tier": 9,
         "primary_topic": "HBM", "key_findings": "fast"},
        {"id": "2605.002", "title": "P2", "relevance_score": 7, "credibility_tier": 7,
         "primary_topic": "DRAM", "key_findings": "efficient"},
    ]
    config = {
        "prompts": {"selection": {"user": "Select {{paper_count}} papers: {{all_papers_json}}", "system": ""}},
        "llm_models": {"selection": "model"},
        "topic": {"focus": "memory"},
        "mode": "weekly",
        "mode_cfg": {"selection_range_label": "1-2", "selection_max": 2},
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "selected_papers": [{"arxiv_id": "2605.001", "reasoning": "top paper"}],
        "remaining_papers": [{"arxiv_id": "2605.002"}],
    })
    selected, remaining = select_papers(papers, config, mock_client)
    assert len(selected) == 1
    assert selected[0]["id"] == "2605.001"
    assert selected[0]["selection_reasoning"] == "top paper"
    assert len(remaining) == 1
    assert remaining[0]["id"] == "2605.002"


import sys
from mira.pipeline import _analyze_paper, analyze_papers, run_pipeline


def test_analyze_paper_attaches_summaries(monkeypatch):
    mock_module = MagicMock()
    mock_module.download_and_extract.return_value = {
        "success": True, "full_text": "Full paper text here.", "page_count": 8,
    }
    monkeypatch.setitem(sys.modules, "download_full_arxiv_pdf", mock_module)
    monkeypatch.setattr("mira.pipeline._patch_pdf_paths", lambda: None)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "large_summary": "Detailed summary", "short_summary": "Brief",
    })

    paper = {"id": "2605.001", "title": "P1", "authors": ["A"],
             "primary_topic": "HBM", "selection_reasoning": "top"}
    config = {
        "prompts": {"analysis": {"user": "Analyze {{arxiv_id}}: {{full_text}}", "system": ""}},
        "llm_models": {"analysis": "model"},
    }
    result = _analyze_paper(paper, config, mock_client)
    assert result["large_summary"] == "Detailed summary"
    assert result["short_summary"] == "Brief"
    assert result["pdf_analysis_performed"] is True


def test_analyze_paper_handles_llm_failure(monkeypatch):
    mock_module = MagicMock()
    mock_module.download_and_extract.return_value = {"success": True, "full_text": "text"}
    monkeypatch.setitem(sys.modules, "download_full_arxiv_pdf", mock_module)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("network error")

    paper = {"id": "2605.001", "title": "P", "authors": [],
             "primary_topic": "HBM", "selection_reasoning": ""}
    config = {
        "prompts": {"analysis": {"user": "{{arxiv_id}}", "system": ""}},
        "llm_models": {"analysis": "model"},
    }
    result = _analyze_paper(paper, config, mock_client)
    assert result["large_summary"] == ""
    assert result["short_summary"] == ""


def test_run_pipeline_returns_correct_shape(monkeypatch):
    """Smoke test: run_pipeline chains classify → filter → select → analyze correctly."""
    papers = [
        {"id": "1", "title": "P1", "summary": "S1", "authors": [], "categories": ["cs.AR"],
         "first_page_text": ""},
        {"id": "2", "title": "P2", "summary": "S2", "authors": [], "categories": ["cs.AR"],
         "first_page_text": ""},
    ]
    config = {
        "prompts": {
            "affiliation": {"user": "{{arxiv_id}}", "system": ""},
            "classification": {"user": "{{arxiv_id}}", "system": ""},
            "selection": {"user": "{{all_papers_json}}", "system": ""},
            "analysis": {"user": "{{arxiv_id}}", "system": ""},
        },
        "llm_models": {"affiliation": "m", "classification": "m", "selection": "m", "analysis": "m"},
        "topic": {"focus": "memory"},
        "mode": "weekly",
        "mode_cfg": {"selection_max": 1, "selection_range_label": "1"},
        "thresholds": {"relevance_score_min": 1, "credibility_tier_min": 1},
    }

    # Patch classify_papers to avoid LLM calls
    def fake_classify(papers, config, client):
        for p in papers:
            p.update({"relevance_score": 8, "credibility_tier": 8, "primary_topic": "HBM",
                      "affiliations": [], "author_affiliations": {}, "credibility_reasoning": "",
                      "secondary_topics": [], "key_findings": "", "actionable": "No"})
        return papers

    # Patch analyze_papers to avoid PDF download + LLM calls
    def fake_analyze(selected, config, client):
        for p in selected:
            p.update({"large_summary": "LS", "short_summary": "SS", "pdf_analysis_performed": False})
        return selected

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = json.dumps({
        "selected_papers": [{"arxiv_id": "1", "reasoning": "best"}],
        "remaining_papers": [{"arxiv_id": "2"}],
    })

    monkeypatch.setattr("mira.pipeline.classify_papers", fake_classify)
    monkeypatch.setattr("mira.pipeline.analyze_papers", fake_analyze)

    result = run_pipeline(papers, config, mock_client)
    assert result["total_scanned"] == 2
    assert len(result["selected"]) == 1
    assert result["selected"][0]["id"] == "1"
    assert len(result["remaining"]) == 1


def test_analyze_paper_raises_when_analysis_model_missing(monkeypatch):
    """Regression: this used to silently resolve to the affiliation model."""
    mock_module = MagicMock()
    mock_module.download_and_extract.return_value = {"success": True, "full_text": "t"}
    monkeypatch.setitem(sys.modules, "download_full_arxiv_pdf", mock_module)

    paper = {"id": "1", "title": "P", "authors": [], "primary_topic": "",
             "selection_reasoning": ""}
    config = {
        "prompts": {"analysis": {"user": "{{arxiv_id}}", "system": ""}},
        "llm_models": {"affiliation": "google/gemini-3.5-flash-lite"},
    }
    with pytest.raises(KeyError) as exc:
        _analyze_paper(paper, config, MagicMock())
    assert "analysis" in str(exc.value)


def test_classify_paper_counts_parse_failure():
    from mira.config import reset_parse_failures, parse_failure_report

    reset_parse_failures()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.content = "not json"

    paper = {"id": "1", "title": "P", "summary": "S"}
    config = {
        "prompts": {"classification": {"user": "{{arxiv_id}}", "system": ""}},
        "llm_models": {"classification": "google/gemini-3.6-flash"},
        "topic": {"focus": "memory"},
    }
    result = _classify_paper(paper, config, mock_client, {})
    assert result["relevance_score"] == 0
    assert parse_failure_report().get("classification") == 1

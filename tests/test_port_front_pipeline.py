# tests/test_port_front_pipeline.py — classify semantics, gate, rank/cap,
# selection, deep analysis and stats, checked against the n8n node code.
from _paths import repoint
import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mira.pipeline as pl
from mira.pipeline import (NoEligiblePapers, SelectionError, apply_thresholds, build_analysis_prompt,
                           build_selection_prompt, classify_papers, completeness_gate,
                           compute_paper_statistics, format_for_selection, rank_and_cap,
                           run_pipeline, select_papers_detailed, validate_selection_output)
from test_port_front_n8n_js import requires_node, run_node

LIVE = json.loads((Path(__file__).parent.parent / "configs" /
                   "memory-innovation-profile.json").read_text())
PROFILE = next(p for p in LIVE["profiles"] if p["profile_id"] == "memory-innovation")


def _config(**over) -> dict:
    cfg = {
        "profile_id": None,  # no shared-cache writes from unit tests
        "mode": "weekly",
        "topic": PROFILE["topic"],
        "prompts": PROFILE["prompts"],
        "thresholds": dict(PROFILE["thresholds"]),
        "mode_cfg": PROFILE["modes"]["weekly"],
        "period_label": "last 8 days",
        "llm_models": LIVE["llm_models"],
    }
    cfg.update(over)
    return cfg


# --- fixture in n8n item shape, converted to the CLI's flat papers ----------

def _js(values):
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _orig(n, title, *, cred=None, summary="Abstract {n}.", authors=("A. One", "B. Two")):
    item = {"id": f"http://arxiv.org/abs/2609.{n:05d}v1", "title": title,
            # Real n8n items carry author/category as JSON.stringify'd arrays
            # ("Prep Data for Pipeline"), verified against execution 2085.
            "summary": summary.format(n=n), "author": _js(list(authors)),
            "category": _js(["cs.AR"]), "published": "2026-09-20T00:00:00.000Z", "run_date": "2026-09-28"}
    if cred is not None:  # affiliation stage ran (Merge Affiliation Data: `|| 0`)
        item.update({"arxiv_id": item["id"], "affiliations": ["Lab"], "author_affiliations": {},
                     "credibility_tier": cred, "credibility_reasoning": f"why {n}"})
    return item


def _cls(rel, topic="HBM", impact="High"):
    return {"primary_topic": topic, "secondary_topics": ["DRAM"], "potential_impact": impact,
            "relevance_score": rel, "key_findings": "finding", "actionable": "Yes"}


FIXTURE = [  # (original, normalized classification output or None)
    (_orig(1, "Top paper", cred=8), _cls(9)),
    (_orig(2, "String scores", cred="7"), _cls("7", topic="CXL", impact="Medium")),
    (_orig(3, "No first page (no affiliation)"), _cls(9)),
    (_orig(4, "Classification failed", cred=9), None),
    (_orig(5, "NaN relevance", cred=9), _cls("abc")),
    (_orig(6, "Below relevance", cred=9), _cls(4, impact="Low")),
    (_orig(7, "Zero credibility", cred=0), _cls(8)),
    (_orig(8, "Über-fast DRAM – “3D” {{x}}", cred=9, summary=""), _cls(9, topic="3D DRAM",
                                                                         impact="Breakthrough")),
    (_orig(9, "Tie on combined", cred=9.5), _cls(8)),
    (_orig(10, "Perfect", cred=10), _cls(10)),
]


def _n8n_items():
    originals = [copy.deepcopy(o) for o, _ in FIXTURE]
    classified = [{**copy.deepcopy(o), "output": {"arxiv_id": o["id"], **c}}
                  for o, c in FIXTURE if c is not None]
    return originals, classified


def _flat(orig: dict, cls: dict | None) -> dict:
    p = {"id": orig["id"].split("/abs/")[1].split("v")[0], "raw_id": orig["id"],
         "title": orig["title"], "summary": orig["summary"], "authors": json.loads(orig["author"]),
         "categories": json.loads(orig["category"]), "published": "2026-09-20", "first_page_text": ""}
    if "credibility_tier" in orig:
        p.update({k: orig[k] for k in ("affiliations", "author_affiliations", "credibility_tier",
                                       "credibility_reasoning")})
    if cls is not None:
        p.update(cls)
    return p


def _flat_papers():
    return [_flat(o, c) for o, c in FIXTURE]


# --- gate / thresholds / rank ------------------------------------------------

def test_completeness_gate_semantics():
    valid, dropped = completeness_gate(_flat_papers(), _config())
    assert [d["arxiv_id"] for d in dropped] == ["2609.00003v1", "2609.00004v1", "2609.00005v1"]
    assert dropped[0] == {"arxiv_id": "2609.00003v1", "missing_relevance_score": False,
                          "missing_credibility_tier": True}
    assert dropped[1]["missing_relevance_score"] and not dropped[1]["missing_credibility_tier"]
    two = next(p for p in valid if p["title"] == "String scores")
    assert two["relevance_score"] == 7 and two["credibility_tier"] == 7  # coerced like Number()
    assert all(p["score_validation"]["dropped_count"] == 3 for p in valid)
    assert valid[0]["score_validation"]["classified_count"] == 9
    kept = apply_thresholds(valid, _config())
    assert [p["title"] for p in kept] == ["Top paper", "String scores", "Über-fast DRAM – “3D” {{x}}",
                                          "Tie on combined", "Perfect"]


def test_explicit_null_score_counts_as_zero_like_number_null():
    valid, dropped = completeness_gate([{"id": "x", "relevance_score": None, "credibility_tier": None}], {})
    assert not dropped and valid[0]["relevance_score"] == 0 and valid[0]["credibility_tier"] == 0


def test_rank_and_cap_weights_ties_and_cap():
    valid, _ = completeness_gate(_flat_papers(), _config())
    ranked = rank_and_cap(apply_thresholds(valid, _config()), _config())
    assert [p["title"] for p in ranked][:3] == ["Perfect", "Über-fast DRAM – “3D” {{x}}", "Top paper"]
    top, tie = ranked[2], ranked[3]
    assert top["combined_score"] == tie["combined_score"] == 8.6  # tie broken by relevance
    assert tie["title"] == "Tie on combined"
    assert top["score_weights"] == {"relevance": 0.6, "credibility": 0.4}
    cfg = _config(thresholds={"relevance_score_min": 5, "credibility_tier_min": 5,
                              "max_filtered_papers": 2, "combined_relevance_weight": 3,
                              "combined_credibility_weight": 1})
    capped = rank_and_cap(apply_thresholds(valid, cfg), cfg)
    assert len(capped) == 2 and capped[0]["score_weights"] == {"relevance": 0.75, "credibility": 0.25}
    assert capped[1]["combined_score"] == 9.0  # 9*0.75 + 9*0.25
    zero = _config(thresholds={"max_filtered_papers": 0})
    assert len(rank_and_cap(valid, zero)) == 1  # Math.max(1, 0)


@requires_node
def test_gate_rank_format_and_selection_prompt_match_n8n(tmp_path):
    originals, classified = _n8n_items()
    run_mode = {"config": PROFILE, "mode": "weekly", "periodLabel": "last 8 days"}
    gate_js = run_node(tmp_path, "Score Completeness Gate", classified,
                       {"Set Run Mode": [run_mode], "Combine Original Data with PDF Extraction": originals})
    thr = PROFILE["thresholds"]
    filtered_js = [j for j in gate_js if j["output"]["relevance_score"] >= thr["relevance_score_min"]
                   and j["credibility_tier"] >= thr["credibility_tier_min"]]  # the IF node
    ranked_js = run_node(tmp_path, "Rank and Cap Filtered Papers", filtered_js, {"Set Run Mode": [run_mode]})
    formatted_js = run_node(tmp_path, "Format for Selection Agent", ranked_js)
    prompt_js = run_node(tmp_path, "Build Selection Prompt", [formatted_js], {"Set Run Mode": [run_mode]})[0]

    valid, _ = completeness_gate(_flat_papers(), _config())
    ranked = rank_and_cap(apply_thresholds(valid, _config()), _config())
    assert [p["raw_id"] for p in ranked] == [j["id"] for j in ranked_js]
    assert [p["combined_score"] for p in ranked] == [j["combined_score"] for j in ranked_js]
    pool = format_for_selection(ranked)
    assert pool == formatted_js["all_papers"]
    user, system = build_selection_prompt(pool, _config())
    assert user == prompt_js["selection_prompt"]
    assert system == prompt_js["selection_system"]
    assert "{{" not in system and "Select a maximum of 20 papers" in system


# --- selection -------------------------------------------------------------

VALIDATE_CASES = [
    {"reasoning": " r ", "selected_papers": [
        {"arxiv_id": "http://arxiv.org/abs/2609.00010v1", "selection_reasoning": " best ", "priority_rank": "2"},
        {"arxiv_id": " 2609.00001v1 ", "priority_rank": None},
        {"arxiv_id": "x3", "priority_rank": "abc"},
        {"arxiv_id": 5}, "junk", {"arxiv_id": "   "}],
     "remaining_papers": [{"arxiv_id": "r1", "exclusion_reasoning": 7}, {"nope": 1}]},
    {"selected_papers": [{"arxiv_id": "a", "priority_rank": True}], "remaining_papers": "x"},
]


@requires_node
@pytest.mark.parametrize("output", VALIDATE_CASES)
def test_validate_selection_output_matches_n8n(tmp_path, output):
    js = run_node(tmp_path, "Validate Selection Output", [{"output": output}])[0]["output"]
    assert validate_selection_output(output) == js


@requires_node
@pytest.mark.parametrize("output", [{}, {"selected_papers": []}, {"selected_papers": [{"arxiv_id": ""}]},
                                    {"reasoning": "x" * 900}])
def test_validate_selection_output_fails_like_n8n(tmp_path, output):
    with pytest.raises(RuntimeError) as js_err:
        run_node(tmp_path, "Validate Selection Output", [{"output": output}])
    with pytest.raises(SelectionError) as py_err:
        validate_selection_output(output)
    assert str(py_err.value) == str(js_err.value)


def _ranked_pool():
    valid, _ = completeness_gate(_flat_papers(), _config())
    return rank_and_cap(apply_thresholds(valid, _config()), _config())


def _patch_llm(monkeypatch, replies):
    calls = []

    def fake(client, model, system, user, retries=3, **kw):
        calls.append({"model": model, "system": system, "user": user, "retries": retries})
        reply = replies.pop(0) if isinstance(replies, list) else replies(model, system, user)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr("mira.config.llm_call", fake)
    monkeypatch.setattr(pl.time, "sleep", lambda s: calls.append({"sleep": s}))
    return calls


def test_selection_splits_and_keeps_reasoning(monkeypatch):
    reply = json.dumps({"reasoning": "strategy", "selected_papers": [
        {"arxiv_id": "http://arxiv.org/abs/2609.00008v1", "selection_reasoning": "3D", "priority_rank": 2},
        {"arxiv_id": "2609.00010", "selection_reasoning": "perfect", "priority_rank": 1},  # bare id
        {"arxiv_id": "http://arxiv.org/abs/2609.00008v1", "selection_reasoning": "dup"},
        {"arxiv_id": "http://arxiv.org/abs/2609.99999v1", "selection_reasoning": "hallucinated"}],
        "remaining_papers": [{"arxiv_id": "http://arxiv.org/abs/2609.00001v1", "exclusion_reasoning": "meh"}]})
    calls = _patch_llm(monkeypatch, ["```json\n" + reply + "\n```"])
    res = select_papers_detailed(_ranked_pool(), _config(), client=None)
    assert [p["title"] for p in res["selected"]] == ["Über-fast DRAM – “3D” {{x}}", "Perfect"]
    first = res["selected"][0]
    assert (first["selection_reasoning"], first["priority_rank"]) == ("3D", 2)
    assert first["arxiv_id"] == "http://arxiv.org/abs/2609.00008v1"
    assert res["unknown_ids"] == ["http://arxiv.org/abs/2609.99999v1"]
    assert [p["title"] for p in res["remaining"]] == ["Top paper", "Tie on combined", "String scores"]
    assert res["remaining"][0]["exclusion_reasoning"] == "meh"
    assert res["remaining"][1]["exclusion_reasoning"] == ""
    assert res["output"]["reasoning"] == "strategy"
    assert calls[0]["model"] == LIVE["llm_models"]["selection"] and calls[0]["retries"] == 1


def test_selection_retries_three_times_then_fails(monkeypatch):
    calls = _patch_llm(monkeypatch, ["not json", RuntimeError("503"), "[1, 2]"])
    with pytest.raises(SelectionError, match="failed after 3 tries"):
        select_papers_detailed(_ranked_pool(), _config(), client=None)
    assert [c.get("sleep") for c in calls if "sleep" in c] == [1.0, 1.0]
    assert len([c for c in calls if "user" in c]) == 3


def test_selection_with_nothing_selected_fails_the_run(monkeypatch):
    empty = json.dumps({"reasoning": "none", "selected_papers": []})
    calls = _patch_llm(monkeypatch, [empty, empty, empty])
    with pytest.raises(SelectionError, match="no usable selected_papers"):
        select_papers_detailed(_ranked_pool(), _config(), client=None)
    # Deviation from n8n (which fails the run at once): an answer with no usable
    # selections is retried like any other failed try.
    assert len([c for c in calls if "model" in c]) == 3


def test_selection_of_only_unknown_ids_fails(monkeypatch):
    _patch_llm(monkeypatch, [json.dumps({"selected_papers": [{"arxiv_id": "http://arxiv.org/abs/1"}]})])
    with pytest.raises(SelectionError, match="not in the candidate pool"):
        select_papers_detailed(_ranked_pool(), _config(), client=None)


def test_empty_pool_raises_no_eligible_papers():
    with pytest.raises(NoEligiblePapers):
        select_papers_detailed([], _config(), client=None)
    assert issubclass(NoEligiblePapers, SelectionError)


# --- deep analysis ------------------------------------------------------------

@requires_node
@pytest.mark.parametrize("success,text", [(True, "Full text with {{ braces }} and $1 and ünïcode."),
                                          (False, "")])
def test_analysis_prompt_matches_n8n(tmp_path, monkeypatch, success, text):
    monkeypatch.delenv("MIRA_ANALYSIS_MAX_CHARS", raising=False)
    item = {"selected_papers": {"arxiv_id": "http://arxiv.org/abs/2609.00008v1",
                                "selection_reasoning": "because", "priority_rank": 1},
            "id": "http://arxiv.org/abs/2609.00008v1", "title": "Über-fast DRAM",
            "author": _js(["A. One", "B. Two"]), "output": {"primary_topic": "3D DRAM"},
            "full_text": text, "pdf_download_success": success}
    js = run_node(tmp_path, "Build Deep Analysis Prompt", [item], {"Set Run Mode": [{"config": PROFILE}]})[0]
    paper = {"id": "2609.00008", "raw_id": item["id"], "arxiv_id": item["id"], "title": item["title"],
             "authors": json.loads(item["author"]), "primary_topic": "3D DRAM", "selection_reasoning": "because"}
    user, system = build_analysis_prompt(paper, _config(), text, success)
    assert user == js["analysis_prompt"] and system == js["analysis_system"]
    assert f'"pdf_analysis_performed": {"true" if success else "false"}' in user


def test_analysis_prompt_truncation_is_opt_in(monkeypatch):
    paper = {"arxiv_id": "a", "title": "t", "authors": [], "primary_topic": "", "selection_reasoning": ""}
    cfg = _config(prompts={"analysis": {"user": "[{{full_text}}]", "system": "s {{x}}"}})
    monkeypatch.delenv("MIRA_ANALYSIS_MAX_CHARS", raising=False)
    assert build_analysis_prompt(paper, cfg, "x" * 70000, True) == ("[" + "x" * 70000 + "]", "s {{x}}")
    monkeypatch.setenv("MIRA_ANALYSIS_MAX_CHARS", "10")
    assert build_analysis_prompt(paper, cfg, "x" * 70000, True)[0] == "[" + "x" * 10 + "]"


def _fake_pdf(monkeypatch, result=None, seen=None):
    mod = MagicMock()

    def dl(arxiv_id):
        if seen is not None:
            seen.append(arxiv_id)
        return result if result is not None else {"success": True, "full_text": "FULL", "page_count": 9}
    mod.download_and_extract.side_effect = dl
    monkeypatch.setitem(sys.modules, "download_full_arxiv_pdf", mod)


def test_analyze_paper_uses_agent_id_and_n8n_fallbacks(monkeypatch):
    seen = []
    _fake_pdf(monkeypatch, seen=seen)
    calls = _patch_llm(monkeypatch, [json.dumps({"arxiv_id": "http://arxiv.org/abs/2609.00008v1",
                                                 "short_summary": "short only",
                                                 "pdf_analysis_performed": "true"})])
    paper = {"id": "2609.00008", "raw_id": "http://arxiv.org/abs/2609.00008v1",
             "arxiv_id": "http://arxiv.org/abs/2609.00008v1", "title": "T", "authors": ["A"],
             "primary_topic": "P", "selection_reasoning": "R"}
    out = pl._analyze_paper(paper, _config(), client=None)
    assert seen == ["http://arxiv.org/abs/2609.00008v1"]
    assert out["large_summary"] == out["short_summary"] == "short only"
    assert out["pdf_analysis_performed"] is True and out["analysis_failed"] is False
    assert (out["page_count"], out["pdf_download_success"], out["pdf_error"]) == (9, True, "")
    assert "full_text" not in out
    assert calls[0]["model"] == LIVE["llm_models"]["analysis"] and "FULL" in calls[0]["user"]


def test_analyze_paper_three_tries_ten_seconds_then_flags(monkeypatch):
    _fake_pdf(monkeypatch, result={"success": False, "error": "HTTP 404", "full_text": ""})
    calls = _patch_llm(monkeypatch, [RuntimeError("a"), "garbage", RuntimeError("c")])
    paper = {"id": "1", "arxiv_id": "1", "title": "T", "authors": [], "primary_topic": "",
             "selection_reasoning": ""}
    out = pl._analyze_paper(paper, _config(), client=None)
    assert [c["sleep"] for c in calls if "sleep" in c] == [10.0, 10.0]
    assert out["analysis_failed"] is True and "c" in out["analysis_error"]
    assert out["large_summary"] == "" and out["pdf_analysis_performed"] is False
    assert out["pdf_download_success"] is False and out["pdf_error"] == "HTTP 404"


def test_analyze_papers_keeps_order_and_failures(monkeypatch):
    _fake_pdf(monkeypatch)

    def reply(model, system, user):
        if "BAD" in user:
            raise RuntimeError("down")
        return json.dumps({"large_summary": "L", "short_summary": "S", "pdf_analysis_performed": True})
    _patch_llm(monkeypatch, reply)
    cfg = _config(prompts={"analysis": {"user": "{{title}}", "system": ""}})
    papers = [{"id": str(i), "arxiv_id": str(i), "title": "BAD" if i == 2 else f"ok{i}"} for i in range(7)]
    out = pl.analyze_papers(papers, cfg, client=None)
    assert [p["id"] for p in out] == [str(i) for i in range(7)]
    assert [p["analysis_failed"] for p in out] == [False, False, True, False, False, False, False]


# --- classify semantics + stats -------------------------------------------------

def test_classify_papers_n8n_stage_semantics(tmp_path, monkeypatch):
    repoint(monkeypatch, "mira.pipeline", tmp_path)

    def reply(model, system, user):
        if "AFF" in user:
            return "not json" if "P3" in user else json.dumps({"credibility_tier": 0, "affiliations": None})
        if "P4" in user:
            raise RuntimeError("classification down")
        return json.dumps({"relevance_score": "6", "primary_topic": None})
    calls = _patch_llm(monkeypatch, reply)
    cfg = _config(prompts={"affiliation": {"user": "AFF {{title}}", "system": ""},
                           "classification": {"user": "CLS {{title}}", "system": ""}})
    papers = [{"id": f"260{i}", "title": f"P{i}", "summary": "s", "authors": [], "categories": [],
               "first_page_text": ("x" * 60) if i != 2 else "  short  "} for i in range(1, 5)]
    classify_papers(papers, cfg, client=None)
    p1, p2, p3, p4 = papers
    assert p1["credibility_tier"] == 0 and p1["affiliations"] == []          # `|| 0`, `|| []`
    assert (p1["relevance_score"], p1["primary_topic"], p1["actionable"]) == ("6", "", "")
    assert "credibility_tier" not in p2                                     # no first-page text
    assert not any("AFF P2" in c.get("user", "") for c in calls)
    assert "credibility_tier" not in p3 and p3["relevance_score"] == "6"    # affiliation parse error
    assert "relevance_score" not in p4 and p4["credibility_tier"] == 0      # classification failed
    valid, dropped = completeness_gate(papers, cfg)
    assert [p["title"] for p in valid] == ["P1"] and len(dropped) == 3


@requires_node
def test_statistics_match_n8n(tmp_path):
    originals, classified = _n8n_items()
    js = run_node(tmp_path, "Compute Paper Statistics", [{}],
                  {"Merge Classification Results1": classified, "Remove Duplicates": originals})[0]["stats"]
    flat_classified = [p for p in _flat_papers() if "relevance_score" in p]
    assert compute_paper_statistics(flat_classified, len(originals)) == js


# --- end to end ------------------------------------------------------------------

def test_run_pipeline_contract(monkeypatch):
    _fake_pdf(monkeypatch)

    def fake_classify(papers, config, client):
        for (orig, cls), p in zip(FIXTURE, papers):
            p.update({k: v for k, v in _flat(orig, cls).items() if k not in p})
        return papers

    def reply(model, system, user):
        if "deep full-text analysis" in user:  # the selection prompt (same model as analysis)
            return json.dumps({"reasoning": "r", "selected_papers": [
                {"arxiv_id": "http://arxiv.org/abs/2609.00010v1", "selection_reasoning": "perfect",
                 "priority_rank": 1}], "remaining_papers": []})
        return json.dumps({"arxiv_id": "http://arxiv.org/abs/2609.00010v1", "large_summary": "L",
                           "short_summary": "S", "pdf_analysis_performed": True})
    _patch_llm(monkeypatch, reply)
    monkeypatch.setattr(pl, "classify_papers", fake_classify)
    papers = [{"id": _flat(o, c)["id"], "raw_id": o["id"], "title": o["title"], "summary": o["summary"],
               "authors": json.loads(o["author"]), "categories": json.loads(o["category"])} for o, c in FIXTURE]
    res = run_pipeline(papers, _config(), client=None)
    assert set(res) == {"selected", "remaining", "total_scanned", "classified", "ranked", "selection_pool",
                        "selection", "unknown_selected_ids", "gate_dropped", "stats"}
    assert res["total_scanned"] == 10 and len(res["classified"]) == 9 and len(res["gate_dropped"]) == 3
    assert [p["title"] for p in res["selected"]] == ["Perfect"]
    sel = res["selected"][0]
    assert (sel["large_summary"], sel["short_summary"], sel["selection_reasoning"], sel["priority_rank"]) == \
        ("L", "S", "perfect", 1)
    assert sel["combined_score"] == 10 and sel["pdf_analysis_performed"] is True
    assert len(res["remaining"]) == len(res["ranked"]) - 1 == 4
    assert res["stats"]["abstracts_analyzed"] == 10 and res["stats"]["total_relevant_papers"] == 9

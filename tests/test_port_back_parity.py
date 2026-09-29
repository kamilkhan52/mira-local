"""Parity of the n8n back-half port against the ORIGINAL n8n Code-node JS.

tests/fixtures/port_back/n8n_goldens.json was produced by executing the
verbatim JavaScript of each n8n node (Compute Paper Statistics, Combine Report
Data, Build Report Prompt, Validate Report Output, Load Prior Reports, Build
Trend Prompt, Append Trend Section, Apply Email Styling) in Node on n8n-shaped
inputs derived from tests/fixtures/port_back/pipeline_fixture.json. These tests
recompute the same outputs in Python and require byte-identical results.
"""
import json
import shutil
from pathlib import Path

import pytest

from mira import render
from mira import report as R
from mira.render import js_json

FX = Path(__file__).parent / "fixtures" / "port_back"
GOLD = json.loads((FX / "n8n_goldens.json").read_text())
G = GOLD["outputs"]
FIXTURE = json.loads((FX / "pipeline_fixture.json").read_text())
CONFIG = FIXTURE["config"]

BODY_OK = "Hey there.\n\n## Executive Summary\n\n" + ("Text " * 60) + "\n\n## This Last Week in Numbers\n\n- x"
OUTPUT = {"subject": "MIRA  Digest - Top  Papers - 2026-09-28", "body": BODY_OK}


def _js(node_out):
    """Code nodes return [{json}] or {json}."""
    return node_out[0]["json"] if isinstance(node_out, list) else node_out["json"]


@pytest.fixture(scope="module")
def prepared():
    return R.build_report_inputs(FIXTURE["pipeline_result"], FIXTURE["media"], CONFIG)


def test_fixture_exercises_the_interesting_paths(prepared):
    s = prepared["combined"]["summary"]
    assert s["total_deep_analysis_fallbacks"] == 1        # paper C: empty analysis
    assert s["total_analyzed_not_selected"] == 2          # report cap 4 < 6 analyzed
    assert s["total_relevant"] == 13                      # all classified, incl. below threshold
    sources = [p["source"] for p in prepared["combined"]["remaining_context_papers"]]
    assert sources[:2] == ["analyzed_not_selected"] * 2 and "not_analyzed_remaining" in sources


def test_compute_paper_statistics_matches_n8n(prepared):
    inp = prepared["n8n_inputs"]
    py = R.compute_paper_statistics(inp["all_classified"], inp["abstracts_analyzed"])
    assert js_json(py) == js_json(_js(G["stats"]))


def test_combine_report_data_matches_n8n(prepared):
    assert js_json(prepared["combined"]) == js_json(_js(G["combine"]))


def test_combine_report_data_without_media_node_matches_n8n(prepared):
    inp = prepared["n8n_inputs"]
    py = R.combine_report_data(
        aggregated=inp["aggregated"], stats=prepared["combined"]["stats"], split_payload=inp["split_payload"],
        all_classified=inp["all_classified"], relevant_pool=inp["relevant_pool"], media_raw={},
        run_mode=prepared["run_mode"])
    assert js_json(py) == js_json(_js(G["combine_nomedia"]))


def test_build_report_prompt_matches_n8n(prepared):
    prompt, system = R.build_report_prompt(prepared["combined"], CONFIG, prepared["run_mode"])
    assert prompt == _js(G["prompt"])["report_prompt"]
    assert system == _js(G["prompt"])["report_system"]


def test_report_prompt_blanks_unknown_placeholders_like_n8n(prepared):
    prompt, _ = R.build_report_prompt(prepared["combined"], CONFIG, prepared["run_mode"])
    # {{mode_label}} / {{topic_short_label}} are not supplied by n8n → '' (live subjects have double spaces)
    assert '"subject": "MIRA  Digest - Top  Papers Published on Arxiv' in prompt
    assert "{{" not in prompt.split("Media Intelligence Data:")[0]
    assert prompt.rstrip().endswith("}")  # media section always appended last


@pytest.mark.parametrize("case", ["ok", "short", "placeholder", "noheading", "missing"])
def test_validate_report_output_matches_n8n(case):
    cases = {
        "ok": OUTPUT,
        "short": {"subject": "Short", "body": "tiny"},
        "placeholder": {"subject": "Example Subject line here", "body": "This is an example " + "x" * 250},
        "noheading": {"subject": "A perfectly fine subject", "body": "y" * 300},
        "missing": {},
    }
    golden = G[f"validate_{case}"]
    if isinstance(golden, dict) and "__error__" in golden:
        with pytest.raises(R.ReportValidationError) as exc:
            R.validate_report_output(cases[case])
        assert str(exc.value) == golden["__error__"]
    else:
        R.validate_report_output(cases[case])


def test_load_prior_reports_matches_n8n(tmp_path):
    for f in (FX / "prior_reports").iterdir():
        shutil.copy(f, tmp_path / f.name)
    record = {"report_id": "report-memory-technology-research-2026-09-28-X", "profile_id": "memory-innovation",
              "topic_name": "Memory Technology Research", "run_date": "2026-09-28"}
    py = R.load_prior_reports(tmp_path, record, 60)
    js = _js(G["load_prior"])
    assert js_json(py) == js_json(js["prior_reports"])
    ids = [r["report_id"] for r in py]
    # 60-day window incl. boundary, newest first, created_at fallback, dict-rooted file
    assert ids == ["report-a", "report-h", "report-j", "report-b", "report-l"]
    # excluded: out of window (c), is_test (d), "test" word in body (e), other profile (f),
    # future (g), other topic (i), invalid JSON (k), not matching report-*.json (notareport)
    assert not {"report-c", "report-d", "report-e", "report-f", "report-g", "report-i", "report-m"} & set(ids)


def test_build_trend_prompt_default_template_matches_n8n(prepared):
    prior = [{"run_date": "2026-09-21", "subject": "A", "body_markdown": "body a"},
             {"created_at": "2026-09-10T12:00:00.000Z", "subject": "H", "body_markdown": "body h"}]
    prompt, system = R.build_trend_prompt(OUTPUT, prior, CONFIG, prepared["run_mode"], 60)
    assert prompt == _js(G["trend_prompt_default"])["trend_prompt"]
    assert system == _js(G["trend_prompt_default"])["trend_system"] == ""


def test_build_trend_prompt_profile_template_matches_n8n(prepared):
    cfg = {**CONFIG, "prompts": {**CONFIG["prompts"], "trend": {
        "user": "T={{topic_name}} F={{ topic_focus }} W={{window_days}} N={{prior_report_count}} U={{unknown}}\n"
                "{{current_report_json}}\n{{prior_reports_json}}",
        "system": "Sys for {{topic_name}} / {{topic_focus}} {{window_days}}"}}}
    prompt, system = R.build_trend_prompt(OUTPUT, [], cfg, prepared["run_mode"], 60)
    assert prompt == _js(G["trend_prompt_custom"])["trend_prompt"]
    assert system == _js(G["trend_prompt_custom"])["trend_system"]


def test_append_trend_section_matches_n8n():
    assert R.append_trend_section(BODY_OK, "  Trend text.\n\nMore.  ", 60) == \
        _js(G["append_trend"])["output"]["body"]
    assert R.append_trend_section(BODY_OK, "   ", 60) == _js(G["append_trend_empty"])["output"]["body"] == BODY_OK


def test_apply_email_styling_matches_n8n(prepared):
    rm = prepared["run_mode"]
    html = render.style_email(
        GOLD["styling_html_content"], "Styled subject", header_title=rm["topicName"],
        digest_label=rm["digestLabel"], current_date=rm["currentDate"],
        colors=render.resolve_palette(CONFIG), stats_dashboard=prepared["combined"]["stats_dashboard"])
    assert html == _js(G["styling"])["html_body"]


def test_apply_email_styling_violet_theme_without_dashboard_matches_n8n(prepared):
    rm = prepared["run_mode"]
    html = render.style_email(
        GOLD["styling_html_content"], "", header_title=rm["topicName"], digest_label=rm["digestLabel"],
        current_date=rm["currentDate"], colors=render.resolve_palette({**CONFIG, "email_theme": "Purple"}))
    assert html == _js(G["styling_violet_nodash"])["html_body"]

"""End-to-end behaviour of mira.report.produce_report (n8n back half) with a
fake LLM: node order, record format and location, prior-report trend input,
parse fallback chain, validation failure, trend handling, PDF tolerance."""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

import mira.config as mira_config
from mira import report as R

FX = Path(__file__).parent / "fixtures" / "port_back"
FIXTURE = json.loads((FX / "pipeline_fixture.json").read_text())
NOW = datetime(2026, 9, 28, 10, 11, 12, 345000, tzinfo=timezone.utc)
GOOD_BODY = ("Hey there, Here's your last 8 days roundup.\n\n## Executive Summary\n\n" + "Findings. " * 40
             + "\n\n## This Last Week in Numbers\n\n- **57 paper abstracts analyzed** last 8 days\n\nUntil next time,")
GOOD_REPORT = json.dumps({"subject": "MIRA  Digest - Top  Papers - 2026-09-28 - [57 scanned, 6 analyzed]",
                          "body": GOOD_BODY})


class FakeLLM:
    """Routes calls by prompt content; records (model, system, user)."""

    def __init__(self, report=GOOD_REPORT, parse=None, trend='{"trend_section_markdown": "HBM keeps rising."}',
                 trend_fix=None):
        self.calls = []
        self.report, self.parse, self.trend, self.trend_fix = report, parse, trend, trend_fix

    def __call__(self, client, model, system, user, *args, **kwargs):
        self.calls.append((model, system, user))
        if user.startswith("Extract the subject and body"):
            return self.parse
        if user.startswith("Instructions:"):
            return self.trend_fix
        if "analyzing multi-week trends" in user:
            return self.trend
        return self.report

    def kinds(self):
        out = []
        for _, _, user in self.calls:
            out.append("parse" if user.startswith("Extract the subject") else
                       "trend_fix" if user.startswith("Instructions:") else
                       "trend" if "analyzing multi-week trends" in user else "report")
        return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    rf = tmp_path / "report-files"
    monkeypatch.setattr(R, "REPORT_FILES", rf)
    monkeypatch.setenv("MIRA_PDF", "0")
    fake = FakeLLM()
    monkeypatch.setattr(mira_config, "llm_call", fake)
    return {"rf": rf, "llm": fake, "config": json.loads(json.dumps(FIXTURE["config"]))}


def _seed_prior(rf: Path, name="report-memory-technology-research-2026-09-21-x.json", **over):
    d = rf / "prod" / "memory-innovation"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"report_id": name[:-5], "profile_id": "memory-innovation", "topic_name": "Memory Technology Research",
           "run_date": "2026-09-21", "created_at": "2026-09-21T05:00:00.000Z", "subject": "Prior digest subject",
           "body_markdown": "## Executive Summary\nPrior body about CXL.", "is_test": False}
    rec.update(over)
    (d / name).write_text(json.dumps([rec], indent=2))


def _run(env, **kw):
    return R.produce_report(FIXTURE["pipeline_result"], FIXTURE["media"], env["config"], None, now=NOW, **kw)


def test_prod_run_writes_record_then_trend_on_current_report(env):
    _seed_prior(env["rf"])
    res = _run(env)

    # n8n order: report generation, then trend (on the CURRENT report + priors)
    assert env["llm"].kinds() == ["report", "trend"]
    model, system, user = env["llm"].calls[0]
    assert model == "anthropic/claude-opus-5"
    assert "Media Intelligence Data:" in user and system.startswith("You are a research analyst")
    trend_model, _, trend_user = env["llm"].calls[1]
    assert trend_model == "anthropic/claude-opus-5"
    assert "Prior report count: 1" in trend_user and "Prior digest subject" in trend_user
    assert "Findings." in trend_user  # current report body is part of the trend input

    # record: prod/<slug>/report-<topic-slug>-<run_date>-<iso ts>.json, [record], 2-space JSON
    path = env["rf"] / "prod" / "memory-innovation" / \
        "report-memory-technology-research-2026-09-28-2026-09-28T10-11-12-345Z.json"
    assert res["record_path"] == path
    text = path.read_text()
    assert not text.endswith("\n")
    records = json.loads(text)
    assert isinstance(records, list) and len(records) == 1
    rec = records[0]
    assert list(rec) == ["report_id", "profile_id", "topic_name", "topic_focus", "period_label", "period_title",
                         "period_range", "run_date", "created_at", "subject", "body_markdown", "is_test",
                         "report_file"]
    assert rec["report_id"] == "report-memory-technology-research-2026-09-28-2026-09-28T10-11-12-345Z"
    assert rec["created_at"] == "2026-09-28T10:11:12.345Z"
    assert (rec["period_label"], rec["period_title"], rec["period_range"]) == \
        ("last 8 days", "Last 8 Days", "2026-09-21 to 2026-09-28")
    assert rec["is_test"] is False and rec["run_date"] == "2026-09-28"
    # the record is persisted BEFORE the trend section is appended
    assert "Multi-Week Trend Watch" not in rec["body_markdown"]
    assert res["body_markdown"].endswith("## Multi-Week Trend Watch (Last 60 Days)\n\nHBM keeps rising.")
    assert res["trend_applied"] is True and res["prior_report_count"] == 1

    # HTML: report-files/report-memory-<UTC ts>.html, styled, dashboard in place of "in Numbers"
    assert res["html_path"] == env["rf"] / "report-memory-2026-09-28T10-11-12.html"
    html = res["html_path"].read_text()
    assert html == res["html"]
    assert "Abstracts Analyzed" in html and "Research Themes (All Papers)" in html
    assert "Memory Technology Research Digest (Last 8 Days)" in html
    assert "#0891b2" in html  # n8n always renders teal
    assert res["pdf_path"] is None and res["pdf_error"] == "disabled"


def test_test_mode_writes_to_tests_dir_and_reads_prod_priors(env):
    _seed_prior(env["rf"])
    env["config"]["test_mode"] = True
    res = _run(env)
    assert res["record_path"].parent == env["rf"] / "tests" / "memory-innovation"
    assert res["report_record"]["is_test"] is True
    assert res["prior_report_count"] == 1  # priors always come from prod/
    assert not list((env["rf"] / "prod" / "memory-innovation").glob("*2026-09-28T10*"))


def test_trend_disabled_skips_trend_call(env):
    env["config"]["trend_enabled"] = False
    res = _run(env)
    assert env["llm"].kinds() == ["report"]
    assert "Multi-Week Trend Watch" not in res["body_markdown"] and res["trend_applied"] is False


def test_trend_runs_even_without_prior_reports(env):
    res = _run(env)
    assert env["llm"].kinds() == ["report", "trend"]
    assert "Prior report count: 0" in env["llm"].calls[1][2]
    assert res["trend_applied"] is True


def test_trend_autofix_pass_uses_sonnet(env):
    env["llm"].trend = "Sure! Here is the trend: HBM."
    env["llm"].trend_fix = '```json\n{"trend_section_markdown": "Fixed trend."}\n```'
    res = _run(env)
    assert env["llm"].kinds() == ["report", "trend", "trend_fix"]
    assert env["llm"].calls[2][0] == "anthropic/claude-sonnet-5"
    assert res["body_markdown"].endswith("Fixed trend.")


def test_trend_failure_is_tolerated(env):
    env["llm"].trend = "no json"
    env["llm"].trend_fix = "still no json"
    res = _run(env)
    assert res["trend_applied"] is False and "trend_section_markdown" in res["trend_error"]
    assert "Multi-Week Trend Watch" not in res["body_markdown"]
    assert res["html_path"].exists()


def test_parse_fallback_chain_on_raw_markdown(env):
    env["llm"].report = "MIRA Digest subject line\n\n" + GOOD_BODY
    env["llm"].parse = "```json\n" + GOOD_REPORT + "\n```"
    res = _run(env)
    assert env["llm"].kinds()[:2] == ["report", "parse"]
    parse_model, _, parse_user = env["llm"].calls[1]
    assert parse_model == "anthropic/claude-sonnet-5"
    assert "Report output:\nMIRA Digest subject line" in parse_user
    assert res["subject"].startswith("MIRA  Digest")


def test_parse_fallback_model_override(env):
    env["config"]["llm_models"] = {**env["config"]["llm_models"], "report_parse": "custom/parser"}
    env["llm"].report = "not json at all"
    env["llm"].parse = GOOD_REPORT
    _run(env)
    assert env["llm"].calls[1][0] == "custom/parser"


def test_lenient_parse_accepts_fenced_json_with_raw_newlines(env):
    # GOOD_BODY's real newlines land raw inside the JSON string (invalid under strict JSON)
    env["llm"].report = 'Here you go:\n```json\n{"subject": "A valid subject line", "body": "' + GOOD_BODY + '"}\n```'
    res = _run(env)
    assert env["llm"].kinds()[0] == "report" and "parse" not in env["llm"].kinds()
    assert res["subject"] == "A valid subject line"


def test_unparseable_report_raises_and_writes_nothing(env):
    env["llm"].report = "garbage"
    env["llm"].parse = "still garbage"
    with pytest.raises(R.ReportParseError):
        _run(env)
    assert not env["rf"].exists() or not list(env["rf"].rglob("*.json"))


def test_validation_failure_stops_before_persist(env):
    env["llm"].report = json.dumps({"subject": "Example subject", "body": "short"})
    with pytest.raises(R.ReportValidationError, match="Report validation failed"):
        _run(env)
    assert env["llm"].kinds() == ["report"]
    assert not env["rf"].exists() or not list(env["rf"].rglob("*.json"))


def test_literal_backslash_n_sequences_become_newlines(env):
    env["config"]["trend_enabled"] = False
    env["llm"].report = json.dumps({"subject": "A valid subject line",
                                    "body": GOOD_BODY + "\\n\\nLiteral escapes."})
    res = _run(env)
    assert res["body_markdown"].endswith("\n\nLiteral escapes.")
    assert res["report_record"]["body_markdown"].endswith("\\n\\nLiteral escapes.")  # record keeps raw body


# ---------------------------------------------------------------- PDF --

def test_pdf_generation_failure_is_tolerated(env, monkeypatch):
    monkeypatch.setenv("MIRA_PDF", "1")

    def boom(*a, **k):
        raise FileNotFoundError("npx")
    monkeypatch.setattr(R.subprocess, "run", boom)
    res = _run(env)
    assert res["pdf_path"] is None and "FileNotFoundError" in res["pdf_error"]
    assert res["html_path"].exists()


def test_pdf_nonzero_exit_and_timeout_are_tolerated(tmp_path, monkeypatch):
    html = tmp_path / "r.html"
    html.write_text("<p>x</p>")
    monkeypatch.setattr(R.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "Chrome not found"))
    path, err = R.generate_pdf(html, tmp_path / "r.pdf")
    assert path is None and "Chrome not found" in err

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("npx", 1)
    monkeypatch.setattr(R.subprocess, "run", slow)
    path, err = R.generate_pdf(html, tmp_path / "r.pdf")
    assert path is None and "TimeoutExpired" in err


def test_pdf_success_and_missing_output(tmp_path, monkeypatch):
    html, pdf = tmp_path / "r.html", tmp_path / "r.pdf"
    html.write_text("<p>x</p>")
    seen = {}

    def ok(cmd, **kw):
        seen.update(cmd=cmd, cwd=kw.get("cwd"))
        Path(cmd[-1]).write_bytes(b"%PDF-1.4")
        return subprocess.CompletedProcess(cmd, 0, "PDF saved", "")
    monkeypatch.setattr(R.subprocess, "run", ok)
    assert R.generate_pdf(html, pdf) == (pdf, None)
    assert seen["cmd"][:3] == ["npx", "tsx", "generate-pdf.ts"] and seen["cwd"] == str(R.CRAWLERS_DIR)

    monkeypatch.setattr(R.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    path, err = R.generate_pdf(html, pdf)
    assert path is None and "no PDF" in err


# ---------------------------------------------------------- run mode --

def test_run_mode_derivation_matches_set_run_mode():
    cfg = FIXTURE["config"]
    rm = R.resolve_run_mode(cfg)
    assert rm["digestLabel"] == "Memory Technology Research Digest (Last 8 Days)"
    assert rm["profileSlug"] == "memory-innovation" and rm["reportMaxSelection"] == 4
    daily = {**cfg, "mode": "daily", "mode_cfg": {"lookback_days": 1}, "start_date_iso": None,
             "end_date_iso": None, "current_date": "2026-09-28"}
    rmd = R.resolve_run_mode(daily)
    assert (rmd["periodLabel"], rmd["periodTitle"], rmd["periodRange"]) == \
        ("last day", "Last Day", "2026-09-28 to 2026-09-28")
    assert rmd["trend_enabled"] is True  # n8n default when modes.<mode>.trend_enabled is absent
    assert rmd["reportMaxSelection"] == 5 and rmd["reportSelectionRangeLabel"] == "3-5"
    explicit = R.resolve_run_mode({**cfg, "period_label": "last week", "period_title": "Last Week",
                                   "period_range": "a to b", "trend_enabled": False, "test_mode": "yes"})
    assert (explicit["periodLabel"], explicit["periodRange"], explicit["trend_enabled"], explicit["isTestMode"]) == \
        ("last week", "a to b", False, True)
    assert R._duration_label(30) == "last month" and R._duration_label(14) == "last 2 weeks"
    assert R._duration_label(90) == "last 3 months" and R._duration_label(8) == "last 8 days"


def test_explicit_selection_with_bare_ids_matches_derived_selection():
    pr = FIXTURE["pipeline_result"]
    derived = R.build_report_inputs(pr, FIXTURE["media"], FIXTURE["config"])
    split = derived["n8n_inputs"]["split_payload"]
    bare = {
        "selected_papers": [{**s, "arxiv_id": s["arxiv_id"].rsplit("/", 1)[-1].split("v")[0]}
                            for s in split["selected_papers"]],
        "remaining_papers": [{**r, "arxiv_id": "https://arxiv.org/abs/" + r["arxiv_id"].rsplit("/", 1)[-1]}
                             for r in split["remaining_papers"]],
    }
    explicit = R.build_report_inputs({**pr, "selection": bare}, FIXTURE["media"], FIXTURE["config"])
    assert R.js_json(explicit["combined"]) == R.js_json(derived["combined"])

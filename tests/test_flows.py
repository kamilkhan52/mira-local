"""End-to-end digest flow under a temporary Prefect server, with every external
dependency faked (arXiv, PDFs, LLM, email). Proves the ported front half
(config → pipeline) and back half (report → trend → delivery) fit together and
that the flow persists a report record."""
import json

import pytest

prefect = pytest.importorskip("prefect")
from prefect.testing.utilities import prefect_test_harness  # noqa: E402


@pytest.fixture(scope="module")
def harness():
    with prefect_test_harness():
        yield


def _papers(n=6):
    return [{
        "id": f"2609.0000{i}", "raw_id": f"http://arxiv.org/abs/2609.0000{i}v1",
        "title": f"CXL memory pooling study {i}",
        "summary": "We study CXL Type 3 memory expanders for pooling. " * 5,
        "published": "2026-09-10", "authors": ["A. Author", "B. Author"],
        "categories": ["cs.AR"], "first_page_text": "Samsung Electronics, Seoul. " * 10,
    } for i in range(n)]


def fake_llm(client, model, system, user, *args, **kwargs):
    text = f"{system}\n{user}"
    if "trend_section_markdown" in text:
        return json.dumps({"trend_section_markdown": "## Trends\n\nCXL pooling keeps rising."})
    if '"subject"' in text and '"body"' in text:
        body = "## Top Papers\n\n" + "CXL pooling results. " * 30 + "\n\n## Also Worth Noting\n\nMore."
        return json.dumps({"subject": "CXL Research Digest - September 2026", "body": body})
    if "selected_papers" in text:
        import re
        ids = list(dict.fromkeys(re.findall(r"http://arxiv\.org/abs/2609\.\d{5}v1", user)))
        return json.dumps({
            "reasoning": "Pick the strongest.",
            "selected_papers": [{"arxiv_id": ids[0], "selection_reasoning": "Best CXL paper.",
                                 "priority_rank": 1}],
            "remaining_papers": [{"arxiv_id": i, "exclusion_reasoning": "Less novel."} for i in ids[1:]],
        })
    if "large_summary" in text:
        return json.dumps({"large_summary": "Detailed analysis. " * 20, "short_summary": "Short."})
    if "credibility_tier" in text:
        return json.dumps({"affiliations": ["Samsung"], "author_affiliations": {"A. Author": ["Samsung"]},
                           "credibility_reasoning": "Major memory vendor.", "credibility_tier": 9})
    if "relevance_score" in text:
        return json.dumps({"primary_topic": "Memory Pooling & Disaggregation",
                           "secondary_topics": ["CXL Devices"], "potential_impact": "High - pooling",
                           "relevance_score": 8, "key_findings": "Pooling helps.",
                           "actionable": "Yes - evaluate"})
    raise AssertionError(f"unexpected LLM prompt: {text[:200]}")


def test_digest_flow_end_to_end(harness, monkeypatch, tmp_path):
    import mira.config
    import mira.pipeline
    import mira.report
    from mira_flows import digest as d
    from _paths import repoint

    for mod in ("mira.pipeline", "mira.report"):
        repoint(monkeypatch, mod, tmp_path)
    monkeypatch.setenv("MIRA_LLM_CACHE_BYPASS", "1")
    for mod in (mira.config, mira.pipeline, mira.report):
        if hasattr(mod, "llm_call"):
            monkeypatch.setattr(mod, "llm_call", fake_llm)
    monkeypatch.setattr(d, "make_llm_client", lambda: object())
    monkeypatch.setattr(d, "fetch_papers", lambda config: _papers())
    monkeypatch.setattr(d, "extract_first_pages", lambda papers: papers)
    import sys
    import types
    fake_pdf = types.ModuleType("download_full_arxiv_pdf")
    fake_pdf.download_and_extract = lambda arxiv_id, *a, **k: {
        "success": True, "full_text": "Full text. " * 50, "page_count": 10}
    monkeypatch.setitem(sys.modules, "download_full_arxiv_pdf", fake_pdf)
    sent = []
    monkeypatch.setattr(mira.report, "send_email",
                        lambda html, subject, config, **kw: sent.append((subject, kw)))

    out = d.digest_flow(profile="cxl-research", mode="monthly", current_date="2026-09-01",
                        test_mode=True, include_media=False, send_email=False, pdf=False,
                        recipients=["team@example.com"])

    assert out["status"] == "ok", out
    assert out["selected"] == 1
    record = json.loads(open(out["record_path"]).read())
    rec = record[0] if isinstance(record, list) else record
    assert rec["profile_id"] == "cxl-research"
    assert rec["is_test"] is True
    assert "/tests/" in out["record_path"]
    assert "## Top Papers" in rec["body_markdown"]

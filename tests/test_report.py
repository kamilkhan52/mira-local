# tests/test_report.py
import json
import pytest
from pathlib import Path
from mira.report import _build_stats, to_html, match_themes, format_theme_label, generate_report


# The real 9 canonical optical themes, in canonical order.
CANONICAL_THEMES = [
    "Scale-Up Architectures",
    "Memory Disaggregation & Pooling",
    "Modulator Technologies",
    "Light Sources & Lasers",
    "Advanced Packaging",
    "Optical Switching",
    "DSP & Retimers",
    "Photonic Integrated Circuits",
    "Optical Network Architecture",
]


def test_match_themes_primary_only():
    paper = {"primary_topic": "Optical Switching", "secondary_topics": []}
    assert match_themes(paper, CANONICAL_THEMES) == ["Optical Switching"]


def test_match_themes_primary_plus_main_category_secondary_overlap():
    paper = {
        "primary_topic": "Photonic Integrated Circuits",
        "secondary_topics": ["Optical Network Architecture"],
    }
    assert match_themes(paper, CANONICAL_THEMES) == [
        "Photonic Integrated Circuits",
        "Optical Network Architecture",
    ]


def test_match_themes_free_form_secondary_ignored():
    paper = {
        "primary_topic": "Modulator Technologies",
        "secondary_topics": ["thermal management", "AI clusters"],
    }
    assert match_themes(paper, CANONICAL_THEMES) == ["Modulator Technologies"]


def test_match_themes_unknown_primary_and_free_form_secondaries():
    paper = {
        "primary_topic": "Unknown",
        "secondary_topics": ["thermal management"],
    }
    assert match_themes(paper, CANONICAL_THEMES) == []


def test_match_themes_dedup_secondary_repeats_primary():
    paper = {
        "primary_topic": "Advanced Packaging",
        "secondary_topics": ["Advanced Packaging"],
    }
    assert match_themes(paper, CANONICAL_THEMES) == ["Advanced Packaging"]


def test_match_themes_qualified_variant_normalizes():
    paper = {
        "primary_topic": "Photonic Integrated Circuits (Silicon Photonics)",
        "secondary_topics": [],
    }
    assert match_themes(paper, CANONICAL_THEMES) == ["Photonic Integrated Circuits"]


def test_match_themes_ordering_primary_before_secondaries():
    paper = {
        "primary_topic": "DSP & Retimers",
        "secondary_topics": ["Light Sources & Lasers", "Scale-Up Architectures"],
    }
    assert match_themes(paper, CANONICAL_THEMES) == [
        "DSP & Retimers",
        "Light Sources & Lasers",
        "Scale-Up Architectures",
    ]


def test_format_theme_label_zero():
    assert format_theme_label([]) == ""


def test_format_theme_label_one():
    assert format_theme_label(["Optical Switching"]) == "(Optical Switching)"


def test_format_theme_label_two():
    assert format_theme_label(["Advanced Packaging", "Optical Switching"]) == (
        "(Advanced Packaging and Optical Switching)"
    )


def test_format_theme_label_three():
    assert format_theme_label(
        ["Modulator Technologies", "Advanced Packaging", "Optical Switching"]
    ) == "(Modulator Technologies, Advanced Packaging and Optical Switching)"


def test_build_stats_counts_correctly():
    selected = [
        {"id": "1", "relevance_score": 8, "credibility_tier": 9, "affiliations": ["Stanford"],
         "primary_topic": "HBM", "pdf_analysis_performed": True},
        {"id": "2", "relevance_score": 7, "credibility_tier": 8, "affiliations": ["NVIDIA"],
         "primary_topic": "PIM", "pdf_analysis_performed": False},
    ]
    remaining = [{"id": "3", "relevance_score": 6, "credibility_tier": 6, "affiliations": [],
                  "primary_topic": "DRAM"}]
    stats = _build_stats(selected, remaining, total_scanned=100)
    assert stats["papers_scanned"] == 100
    assert stats["papers_deep_analyzed"] == 2
    assert stats["papers_relevant"] == 3
    assert stats["pdf_success_count"] == 1


def test_to_html_contains_title():
    config = {
        "email_cfg": {
            "header_label": "Test Digest",
            "colors": {"primary": "#0891b2", "accent": "#14b8a6"},
        },
        "topic": {"name": "Memory Technology Research", "assistant_signature": "MIRA"},
        "current_date": "2026-05-18",
        "mode": "weekly",
    }
    html = to_html("## Hello\n\nTest content.", "Test Subject", config)
    assert "Test Subject" in html
    assert "Hello" in html
    assert "#0891b2" in html


def test_to_html_converts_markdown_links():
    config = {
        "email_cfg": {
            "header_label": "Test",
            "colors": {"primary": "#0891b2", "accent": "#14b8a6"},
        },
        "topic": {"name": "Memory", "assistant_signature": "MIRA"},
        "current_date": "2026-05-18",
        "mode": "weekly",
    }
    html = to_html("[Paper Title](http://arxiv.org/abs/2605.11277)", "Subject", config)
    assert 'href="http://arxiv.org/abs/2605.11277"' in html


def test_generate_report_attaches_theme_label_to_analyzed_papers(monkeypatch):
    import mira.config as mira_config

    captured = {}

    def fake_llm_call(client, model, system, user, *args, **kwargs):
        captured["user"] = user
        return json.dumps({"subject": "S", "body": "B"})

    monkeypatch.setattr(mira_config, "llm_call", fake_llm_call)

    selected = [
        {
            "id": "1",
            "primary_topic": "Photonic Integrated Circuits",
            "secondary_topics": ["Optical Network Architecture"],
            "affiliations": ["Ayar Labs"],
        },
    ]
    config = {
        "mode": "weekly",
        "current_date": "2026-07-20",
        "themes": CANONICAL_THEMES,
        "topic": {"focus": "optical interconnects", "name": "Optical", "short_label": "Optical"},
        "thresholds": {"relevance_score_min": 7, "credibility_tier_min": 6},
        "llm_models": {"report": "test-model"},
        "mode_cfg": {},
        "prompts": {"report": {"system": "sys", "user": "PAPERS: {{analyzed_papers_json}}"}},
    }

    result = generate_report(
        selected=selected,
        remaining=[],
        media=[],
        config=config,
        client=None,
        trend_section="",
        total_scanned=10,
    )

    assert result == {"subject": "S", "body": "B"}
    payload = json.loads(captured["user"].split("PAPERS: ", 1)[1])
    assert payload[0]["theme_label"] == (
        "(Photonic Integrated Circuits and Optical Network Architecture)"
    )
    # Caller's original list must not be mutated.
    assert "theme_label" not in selected[0]


def test_build_stats_pdf_success_count_handles_missing_key():
    selected = [
        {"id": "1", "affiliations": [], "primary_topic": "HBM"},  # no pdf_analysis_performed key
        {"id": "2", "affiliations": [], "primary_topic": "HBM", "pdf_analysis_performed": True},
    ]
    stats = _build_stats(selected, [], total_scanned=10)
    assert stats["pdf_success_count"] == 1


def test_generate_report_raises_when_report_model_missing():
    """Regression: report generation used to silently borrow the selection model."""
    config = {
        "mode": "weekly",
        "current_date": "2026-07-22",
        "themes": [],
        "topic": {"focus": "memory", "name": "Memory", "short_label": "Memory"},
        "thresholds": {"relevance_score_min": 7, "credibility_tier_min": 6},
        "llm_models": {"selection": "anthropic/claude-sonnet-5"},
        "mode_cfg": {},
        "prompts": {"report": {"system": "sys", "user": "PAPERS: {{analyzed_papers_json}}"}},
    }
    with pytest.raises(KeyError) as exc:
        generate_report(
            selected=[], remaining=[], media=[], config=config,
            client=None, trend_section="", total_scanned=0,
        )
    assert "report" in str(exc.value)


def test_default_model_helper_is_gone():
    import mira.report
    assert not hasattr(mira.report, "_default_model")

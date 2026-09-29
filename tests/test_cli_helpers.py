import argparse
import sys
from datetime import date
from types import SimpleNamespace

import pytest

import hypothesize
from mira.config import iso_date_arg, llm_call


def _spy_client(captured: dict):
    def create(**kw):
        captured.update(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))


def test_llm_call_passes_temperature_when_given():
    captured = {}
    llm_call(_spy_client(captured), "m", "sys", "usr", temperature=0.0)
    assert captured["temperature"] == 0.0


def test_llm_call_omits_temperature_by_default():
    captured = {}
    llm_call(_spy_client(captured), "m", "sys", "usr")
    assert "temperature" not in captured


def test_iso_date_arg_parses_valid_date():
    assert iso_date_arg("2026-07-15") == date(2026, 7, 15)


def test_iso_date_arg_rejects_garbage_with_argparse_error():
    with pytest.raises(argparse.ArgumentTypeError, match="YYYY-MM-DD"):
        iso_date_arg("not-a-date")


def test_dossier_filename_includes_run_time_stamp():
    # Same-day reruns must not collide: the name carries an HHMM stamp.
    name = hypothesize.out_name("CXL Memory Pooling", "2026-07-15-0938")
    assert name == "cxl-memory-pooling-2026-07-15-0938.md"


def _run_hypothesis_cli_without_pipeline(monkeypatch, tmp_path, profiles):
    captured = {}

    class FakeGraph:
        def nodes(self, data=False):
            assert data is True
            return [
                ("paper", {"entity_type": "Paper"}),
                ("seed", {"entity_type": "Topic"}),
            ]

    def build_corpus(_graph, selected_profiles, *, since):
        selected_profiles = tuple(selected_profiles)
        captured["selected_profiles"] = selected_profiles
        return SimpleNamespace(
            papers={"paper"},
            profile_ids=selected_profiles,
            topic_papers={"Seed": {"paper"}},
            related_topics=lambda _topic: set(),
        )

    target = SimpleNamespace(
        graphml=tmp_path / "unused.graphml",
        vdb_entities=tmp_path / "unused-vectors.json",
        base_url="http://unused.test",
    )
    monkeypatch.setattr(hypothesize, "ROOT", tmp_path)
    monkeypatch.setattr(hypothesize, "resolve_target", lambda _name: target)
    monkeypatch.setattr(hypothesize, "load_graph", lambda _path: FakeGraph())
    monkeypatch.setattr(hypothesize.Corpus, "build", build_corpus)
    monkeypatch.setattr(
        hypothesize, "match_anchor_topics", lambda _topic, _topics: ["Seed"]
    )
    monkeypatch.setattr(hypothesize, "mine_gaps", lambda _corpus, _expanded: [])
    monkeypatch.setattr(
        hypothesize.EntityVectors,
        "load",
        lambda _path: SimpleNamespace(side_similarity=lambda *args, **kwargs: 0),
    )
    monkeypatch.setattr(
        hypothesize, "score_candidates", lambda candidates, _semantic: candidates
    )
    monkeypatch.setattr(hypothesize, "make_llm_client", lambda: object())

    def render(meta, hypotheses, candidates):
        captured["meta"] = meta
        assert hypotheses == []
        assert candidates == []
        return "# controlled dossier\n"

    monkeypatch.setattr(hypothesize, "render_dossier", render)
    argv = ["hypothesize.py", "--graph", "combined", "--topic", "Seed"]
    for profile in profiles:
        argv.extend(["--profile", profile])
    argv.append("--no-external")
    monkeypatch.setattr(sys, "argv", argv)

    assert hypothesize.main() == 0
    captured["written"] = list(
        (tmp_path / "report-files" / "hypotheses").rglob("*.md")
    )
    return captured


def test_single_profile_cli_preserves_profile_output_directory(monkeypatch, tmp_path):
    """Making --profile repeatable must not move existing single-profile dossiers."""
    captured = _run_hypothesis_cli_without_pipeline(
        monkeypatch, tmp_path, ["memory-innovation"]
    )

    assert captured["selected_profiles"] == ("memory-innovation",)
    assert captured["meta"]["profiles"] == ("memory-innovation",)
    assert len(captured["written"]) == 1
    assert captured["written"][0].parent.name == "memory-innovation"


def test_repeated_profile_flags_reach_corpus_in_order_and_use_combined_output(
    monkeypatch, tmp_path
):
    """Collapsing repeated flags to the last profile defeats union hypothesis runs."""
    captured = _run_hypothesis_cli_without_pipeline(
        monkeypatch,
        tmp_path,
        ["memory-innovation", "optical-io", "storage-fabric"],
    )

    assert captured["selected_profiles"] == (
        "memory-innovation",
        "optical-io",
        "storage-fabric",
    )
    assert captured["meta"]["profiles"] == captured["selected_profiles"]
    assert len(captured["written"]) == 1
    assert captured["written"][0].parent.name == "combined"

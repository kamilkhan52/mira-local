from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from chatbot.config import Settings
from mira.exhaustive.types import EvidenceChunk, ResearchEvidenceBundle


def test_evidence_chunk_is_immutable():
    chunk = EvidenceChunk(
        chunk_id="chunk-a",
        content_hash="sha256:a",
        content="claim",
        domains=("memory", "optical"),
        file_paths=("a.md", "b.md"),
        paper_names=("Paper A",),
    )

    with pytest.raises(FrozenInstanceError):
        chunk.content = "changed"


def test_empty_bundle_has_explicit_incomplete_coverage_for_every_domain():
    bundle = ResearchEvidenceBundle.empty("query")

    assert dict(bundle.nodes_scanned) == {
        "memory": 0,
        "optical": 0,
        "storage": 0,
    }
    assert dict(bundle.edges_scanned) == {
        "memory": 0,
        "optical": 0,
        "storage": 0,
    }
    assert bundle.exhaustive is False


def test_exhaustive_settings_are_read_from_environment(monkeypatch, tmp_path):
    memory = tmp_path / "memory"
    optical = tmp_path / "optical"
    storage = tmp_path / "storage"
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    monkeypatch.setenv("EXHAUSTIVE_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("EXHAUSTIVE_RESEARCH_TOKEN", "research-secret")
    monkeypatch.setenv("EXHAUSTIVE_MEMORY_DIR", str(memory))
    monkeypatch.setenv("EXHAUSTIVE_OPTICAL_DIR", str(optical))
    monkeypatch.setenv("EXHAUSTIVE_STORAGE_DIR", str(storage))
    monkeypatch.setenv("EXHAUSTIVE_NODE_THRESHOLD", "0.51")
    monkeypatch.setenv("EXHAUSTIVE_EDGE_THRESHOLD", "0.47")
    monkeypatch.setenv("EXHAUSTIVE_BATCH_TOKENS", "64000")
    monkeypatch.setenv("EXHAUSTIVE_QUEUE_SIZE", "8")
    monkeypatch.setenv("EXHAUSTIVE_CONCURRENCY", "2")
    monkeypatch.setenv("EXHAUSTIVE_JOB_TIMEOUT_SEC", "1800")
    monkeypatch.setenv("EXHAUSTIVE_MAP_MODEL", "map/model")
    monkeypatch.setenv("EXHAUSTIVE_SYNTHESIS_MODEL", "synthesis/model")
    monkeypatch.setenv("MIRA_MAX_ESTIMATED_COST_USD", "3.5")
    monkeypatch.setenv("EXHAUSTIVE_MAX_PAPERS", "12")
    monkeypatch.setenv("EXHAUSTIVE_MAX_CHUNKS_PER_PAPER", "4")

    settings = Settings.from_env()

    assert settings.exhaustive_enabled is True
    assert settings.research_token == "research-secret"
    assert settings.exhaustive_memory_dir == memory
    assert settings.exhaustive_optical_dir == optical
    assert settings.exhaustive_storage_dir == storage
    assert settings.exhaustive_node_threshold == pytest.approx(0.51)
    assert settings.exhaustive_edge_threshold == pytest.approx(0.47)
    assert settings.exhaustive_batch_tokens == 64_000
    assert settings.exhaustive_queue_size == 8
    assert settings.exhaustive_concurrency == 2
    assert settings.exhaustive_job_timeout_sec == 1800
    assert settings.exhaustive_map_model == "map/model"
    assert settings.exhaustive_synthesis_model == "synthesis/model"
    assert settings.max_estimated_cost_usd == pytest.approx(3.5)
    assert settings.exhaustive_max_papers == 12
    assert settings.exhaustive_max_chunks_per_paper == 4


def test_exhaustive_settings_defaults_are_fail_closed(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    for name in (
        "EXHAUSTIVE_RETRIEVAL_ENABLED",
        "EXHAUSTIVE_RESEARCH_TOKEN",
        "EXHAUSTIVE_MEMORY_DIR",
        "EXHAUSTIVE_OPTICAL_DIR",
        "EXHAUSTIVE_STORAGE_DIR",
        "EXHAUSTIVE_NODE_THRESHOLD",
        "EXHAUSTIVE_EDGE_THRESHOLD",
        "EXHAUSTIVE_BATCH_TOKENS",
        "EXHAUSTIVE_QUEUE_SIZE",
        "EXHAUSTIVE_CONCURRENCY",
        "EXHAUSTIVE_JOB_TIMEOUT_SEC",
        "EXHAUSTIVE_MAP_MODEL",
        "EXHAUSTIVE_SYNTHESIS_MODEL",
        "MIRA_MAX_ESTIMATED_COST_USD",
        "EXHAUSTIVE_MAX_PAPERS",
        "EXHAUSTIVE_MAX_CHUNKS_PER_PAPER",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.exhaustive_enabled is False
    assert settings.research_token == ""
    assert settings.exhaustive_memory_dir == Path("data/lightrag/working_dir")
    assert settings.exhaustive_optical_dir == Path("data/lightrag/working_dir_optical")
    assert settings.exhaustive_storage_dir == Path("data/lightrag/working_dir_storage")
    assert settings.exhaustive_node_threshold == pytest.approx(0.42)
    assert settings.exhaustive_edge_threshold == pytest.approx(0.38)
    assert settings.exhaustive_batch_tokens == 80_000
    assert settings.exhaustive_queue_size == 20
    assert settings.exhaustive_concurrency == 1
    assert settings.exhaustive_job_timeout_sec == 3600
    # Resolved from the shared llm_models block, not hardcoded here.
    assert settings.exhaustive_map_model == "google/gemini-2.5-flash"
    assert settings.exhaustive_synthesis_model == "anthropic/claude-sonnet-5"
    assert settings.max_estimated_cost_usd == pytest.approx(25.0)
    assert settings.exhaustive_max_papers == 400
    assert settings.exhaustive_max_chunks_per_paper == 40


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EXHAUSTIVE_NODE_THRESHOLD", "-0.01"),
        ("EXHAUSTIVE_NODE_THRESHOLD", "1.01"),
        ("EXHAUSTIVE_EDGE_THRESHOLD", "-0.01"),
        ("EXHAUSTIVE_EDGE_THRESHOLD", "1.01"),
        ("EXHAUSTIVE_BATCH_TOKENS", "0"),
        ("EXHAUSTIVE_QUEUE_SIZE", "0"),
        ("EXHAUSTIVE_CONCURRENCY", "0"),
        ("EXHAUSTIVE_JOB_TIMEOUT_SEC", "0"),
        ("EXHAUSTIVE_MAX_PAPERS", "0"),
        ("EXHAUSTIVE_MAX_CHUNKS_PER_PAPER", "0"),
    ],
)
def test_exhaustive_settings_reject_invalid_bounds(monkeypatch, name, value):
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="exhaustive"):
        Settings.from_env()


def test_spend_ceiling_must_be_positive(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    monkeypatch.setenv("MIRA_MAX_ESTIMATED_COST_USD", "0")

    with pytest.raises(ValueError, match="max_estimated_cost_usd"):
        Settings.from_env()


def test_exhaustive_cli_runs_without_a_lightrag_api_key(monkeypatch, tmp_path):
    """--exhaustive never calls LightRAG, so it must not demand its key."""
    import hypothesize

    monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)
    captured = {}

    def fake_run_research_job(settings, kind, request, emit):
        captured["kind"] = kind
        captured["settings"] = settings
        emit({"name": "evidence_collected"})
        return SimpleNamespace(
            result=SimpleNamespace(markdown="# Dossier"),
            cost=SimpleNamespace(actual_cost_usd=0.5, cost_status="actual"),
            coverage=SimpleNamespace(
                nodes_scanned={"memory": 1, "optical": 2, "storage": 3},
                edges_scanned={"memory": 4, "optical": 5, "storage": 6},
            ),
        )

    monkeypatch.setattr(
        hypothesize, "run_research_job", fake_run_research_job
    )
    args = SimpleNamespace(
        topic=["HBM"],
        profile=["memory-innovation"],
        max_hypotheses=3,
        critic=False,
        no_external=True,
        out_dir=str(tmp_path),
    )

    assert hypothesize._run_exhaustive(args) == 0
    assert captured["kind"] == "hypotheses"
    assert captured["settings"].lightrag_api_key == ""
    written = list((tmp_path / "memory-innovation").glob("*.md"))
    assert len(written) == 1
    assert "# Dossier" in written[0].read_text()

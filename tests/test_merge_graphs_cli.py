from pathlib import Path

import merge_graphs


def test_restart_combined_force_recreates_atomic_directory_mount(monkeypatch, tmp_path):
    """An atomic working-dir swap needs a new container mount, not `up -d` alone."""
    calls = []
    monkeypatch.setattr(
        merge_graphs.subprocess,
        "run",
        lambda command, check: calls.append((command, check)),
    )

    merge_graphs.restart_combined(tmp_path / "compose.yml")

    assert calls == [(
        ["docker", "compose", "-f", str(tmp_path / "compose.yml"),
         "up", "-d", "--force-recreate", "lightrag-combined"],
        True,
    )]

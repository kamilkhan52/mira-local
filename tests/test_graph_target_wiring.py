"""Cross-graph contamination guard.

`retrieve_context` falls back to the memory server (:9621) when no base_url is
passed (mira/hypothesis/synthesis.py). So a CLI that resolved the optical
working_dir but left a `retrieve_context` call bare would read the optical
graphml while retrieving synthesis evidence from the *memory* graph — a wrong
answer, silently. That was gap 3 in the spec, and it was invisible precisely
because nothing failed.

These tests pin the invariant at the call sites rather than the transport: every
`retrieve_context(...)` in either CLI must pass base_url explicitly.
"""
import ast
from pathlib import Path

import pytest

from mira.config import ROOT
from mira.graph_target import resolve_target

CLIS = ["hypothesize.py", "discover.py"]


def _retrieve_context_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "retrieve_context"
    ]


@pytest.mark.parametrize("cli", CLIS)
def test_every_retrieve_context_call_passes_base_url(cli):
    calls = _retrieve_context_calls((ROOT / cli).read_text())
    assert calls, f"no retrieve_context calls found in {cli} — did it get renamed?"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "base_url" in kwargs, (
            f"{cli}:{call.lineno} calls retrieve_context without base_url — "
            "it would silently retrieve from the memory graph (:9621)"
        )


@pytest.mark.parametrize("cli", CLIS)
def test_retrieve_context_base_url_comes_from_the_target(cli):
    # Guards against a hardcoded URL creeping back in: the value must be read
    # off the resolved target, not spelled literally.
    calls = _retrieve_context_calls((ROOT / cli).read_text())
    for call in calls:
        node = next(k.value for k in call.keywords if k.arg == "base_url")
        assert isinstance(node, ast.Attribute) and node.attr == "base_url", (
            f"{cli}:{call.lineno} should pass base_url=target.base_url"
        )
        assert isinstance(node.value, ast.Name) and node.value.id == "target"


def test_optical_base_url_is_not_the_memory_server():
    # The premise the guard rests on: these genuinely differ, so a missed
    # base_url is a real contamination and not a no-op.
    assert resolve_target("optical").base_url != resolve_target("memory").base_url


@pytest.mark.parametrize("cli", CLIS)
def test_cli_exposes_graph_flag(cli):
    source = (ROOT / cli).read_text()
    assert '"--graph"' in source, f"{cli} must expose --graph"

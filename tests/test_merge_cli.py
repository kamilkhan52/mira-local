import json as _json
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

import merge_graphs
from mira.graph_merge import MergeError
from mira.merge_cli import make_openai_embedder, parse_env_file


def test_parse_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# comment\nA=1\nEMBEDDING_BINDING_API_KEY=sk-x=y\n\nB=two\n")
    env = parse_env_file(f)
    assert env == {"A": "1", "EMBEDDING_BINDING_API_KEY": "sk-x=y", "B": "two"}


def test_openai_embedder_batches_and_orders(monkeypatch):
    seen = []

    def handler(request):
        body = _json.loads(request.content)
        seen.append(body["input"])
        data = [{"index": i, "embedding": [float(len(t))] * 8}
                for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": data})

    transport = httpx.MockTransport(handler)
    embed = make_openai_embedder("sk-test", "https://example.test/v1",
                                 "text-embedding-3-small", batch_size=2,
                                 transport=transport)
    out = embed(["a", "bb", "ccc"])
    assert out.shape == (3, 8)
    assert seen == [["a", "bb"], ["ccc"]]
    assert out[1][0] == 2.0


def test_openai_embedder_orders_by_index_not_response_order():
    """Kills: embedder assuming response order instead of sorting by `index`.

    The fake returns rows in REVERSED index order; a correct embedder sorts by
    `index` and recovers the original order, so out[:, 0] == [0, 1, 2].
    """
    def handler(request):
        body = _json.loads(request.content)
        data = [{"index": i, "embedding": [float(i)] * 4}
                for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    transport = httpx.MockTransport(handler)
    embed = make_openai_embedder("k", "https://x.test/v1", "m",
                                 batch_size=10, transport=transport)
    out = embed(["a", "b", "c"])
    assert list(out[:, 0]) == [0.0, 1.0, 2.0]


def test_openai_embedder_retries_on_429_then_succeeds(monkeypatch):
    """Kills: giving up on the first 429 (must retry). Also pins first backoff=1s."""
    sleeps = []
    monkeypatch.setattr("mira.merge_cli.time.sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        body = _json.loads(request.content)
        data = [{"index": i, "embedding": [1.0] * 4}
                for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": data})

    transport = httpx.MockTransport(handler)
    embed = make_openai_embedder("k", "https://x.test/v1", "m",
                                 transport=transport)
    out = embed(["a"])
    assert calls["n"] == 2
    assert out.shape == (1, 4)
    assert sleeps == [1]  # 2 ** 0 on the first retry


def test_openai_embedder_raises_after_retries_exhausted(monkeypatch):
    """Kills: not raising MergeError when retries run out; also pins backoff=1,2,4."""
    sleeps = []
    monkeypatch.setattr("mira.merge_cli.time.sleep", lambda s: sleeps.append(s))

    def handler(request):
        return httpx.Response(503, text="unavailable")

    transport = httpx.MockTransport(handler)
    embed = make_openai_embedder("k", "https://x.test/v1", "m",
                                 max_retries=3, transport=transport)
    with pytest.raises(MergeError):
        embed(["a"])
    assert sleeps == [1, 2, 4]  # exponential backoff before the final raise


def test_openai_embedder_does_not_retry_on_400(monkeypatch):
    """Kills: retrying on 4xx other than 429 (a 400 is an immediate MergeError)."""
    monkeypatch.setattr("mira.merge_cli.time.sleep",
                        lambda s: pytest.fail("must not sleep/retry on 400"))
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    transport = httpx.MockTransport(handler)
    embed = make_openai_embedder("k", "https://x.test/v1", "m",
                                 transport=transport)
    with pytest.raises(MergeError):
        embed(["a"])
    assert calls["n"] == 1


def _merge_report(output_dir):
    return SimpleNamespace(
        nodes=3, edges=2, shared_entities=1, shared_relations=1,
        shared_chunks=1, reembedded=2, output_dir=output_dir,
    )


def _mock_embedder(monkeypatch):
    embed_fn = object()
    monkeypatch.setattr(
        merge_graphs, "parse_env_file",
        lambda path: {"EMBEDDING_BINDING_API_KEY": "test-key"},
    )
    monkeypatch.setattr(merge_graphs, "make_openai_embedder", lambda *args: embed_fn)
    return embed_fn


def test_cli_without_storage_keeps_single_pairwise_merge(tmp_path, monkeypatch):
    """Without --storage-dir the established two-way invocation is unchanged."""
    memory, optical, output = (tmp_path / name for name in ("memory", "optical", "combined"))
    calls = []

    def fake_run_merge(left, right, destination, **kwargs):
        calls.append((left, right, destination, kwargs))
        return _merge_report(destination)

    embed_fn = _mock_embedder(monkeypatch)
    monkeypatch.setattr(merge_graphs, "run_merge", fake_run_merge)
    monkeypatch.setattr(sys, "argv", [
        "merge_graphs.py", "--memory-dir", str(memory), "--optical-dir", str(optical),
        "--output-dir", str(output), "--skip-restart",
    ])

    assert merge_graphs.main() == 0
    assert calls == [
        (memory, optical, output, {"embed_fn": embed_fn, "dry_run": False}),
    ]


def test_cli_storage_chains_merges_rewrites_provenance_and_cleans_intermediate(
        tmp_path, monkeypatch):
    memory, optical, storage, output = (
        tmp_path / name for name in ("memory", "optical", "storage", "combined"))
    intermediate = output.with_name(f"{output.name}.chain-tmp")
    calls, sources, generated = [], {}, []

    def fake_run_merge(left, right, destination, **kwargs):
        calls.append((left, right, destination, kwargs))
        destination.mkdir()
        (destination / "provenance.json").write_text('{"legacy": true}')
        return _merge_report(destination)

    def fake_build_provenance_multi(loaded_sources, generated_at):
        generated.append(generated_at)
        assert loaded_sources == sources
        return {"entities": {"Shared": "memory+optical+storage"}}

    source_objects = {path: object() for path in (memory, optical, storage)}
    sources.update({
        "memory": source_objects[memory], "optical": source_objects[optical],
        "storage": source_objects[storage],
    })
    embed_fn = _mock_embedder(monkeypatch)
    monkeypatch.setattr(merge_graphs, "run_merge", fake_run_merge)
    monkeypatch.setattr(merge_graphs, "load_working_dir", lambda path: source_objects[path])
    monkeypatch.setattr(merge_graphs, "build_provenance_multi", fake_build_provenance_multi)
    monkeypatch.setattr(sys, "argv", [
        "merge_graphs.py", "--memory-dir", str(memory), "--optical-dir", str(optical),
        "--storage-dir", str(storage), "--output-dir", str(output), "--skip-restart",
    ])

    assert merge_graphs.main() == 0
    assert calls == [
        (memory, optical, intermediate, {"embed_fn": embed_fn, "dry_run": False}),
        (intermediate, storage, output, {"embed_fn": embed_fn, "dry_run": False}),
    ]
    assert json.loads((output / "provenance.json").read_text()) == {
        "entities": {"Shared": "memory+optical+storage"},
    }
    assert generated
    assert not intermediate.exists()


def test_cli_storage_keeps_intermediate_when_second_merge_aborts(tmp_path, monkeypatch):
    memory, optical, storage, output = (
        tmp_path / name for name in ("memory", "optical", "storage", "combined"))
    intermediate = output.with_name(f"{output.name}.chain-tmp")
    calls = []

    def fake_run_merge(left, right, destination, **kwargs):
        calls.append((left, right, destination, kwargs))
        if destination == output:
            raise MergeError("second merge failed")
        destination.mkdir()
        return _merge_report(destination)

    _mock_embedder(monkeypatch)
    monkeypatch.setattr(merge_graphs, "run_merge", fake_run_merge)
    monkeypatch.setattr(sys, "argv", [
        "merge_graphs.py", "--memory-dir", str(memory), "--optical-dir", str(optical),
        "--storage-dir", str(storage), "--output-dir", str(output), "--skip-restart",
    ])

    assert merge_graphs.main() == 1
    assert [call[:3] for call in calls] == [
        (memory, optical, intermediate), (intermediate, storage, output),
    ]
    assert intermediate.exists()


def test_cli_storage_dry_run_preflights_first_stage_without_an_api_key(
        tmp_path, monkeypatch, capsys):
    memory, optical, storage, output = (
        tmp_path / name for name in ("memory", "optical", "storage", "combined"))
    intermediate = output.with_name(f"{output.name}.chain-tmp")
    calls = []

    def fake_run_merge(left, right, destination, **kwargs):
        calls.append((left, right, destination, kwargs))
        return _merge_report(destination)

    monkeypatch.setattr(merge_graphs, "run_merge", fake_run_merge)
    monkeypatch.setattr(
        merge_graphs, "parse_env_file",
        lambda path: pytest.fail("dry runs must not read an API key"),
    )
    monkeypatch.setattr(
        merge_graphs, "make_openai_embedder",
        lambda *args: pytest.fail("dry runs must not create an embedder"),
    )
    monkeypatch.setattr(sys, "argv", [
        "merge_graphs.py", "--memory-dir", str(memory), "--optical-dir", str(optical),
        "--storage-dir", str(storage), "--output-dir", str(output), "--dry-run",
    ])

    assert merge_graphs.main() == 0
    assert calls == [
        (memory, optical, intermediate, {"embed_fn": None, "dry_run": True}),
    ]
    assert not intermediate.exists()
    assert not output.exists()
    assert "storage chaining requires a real run" in capsys.readouterr().out

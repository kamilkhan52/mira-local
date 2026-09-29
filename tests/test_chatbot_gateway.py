import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import chatbot.app as chatbot_app
from chatbot.app import (ChatRequest, ProvenanceStore, RateLimiter,
                         _is_safe_dossier_component, create_app)
from chatbot import config
from chatbot.config import Settings


@pytest.fixture()
def provenance_file(tmp_path):
    p = tmp_path / "provenance.json"
    p.write_text(json.dumps({
        "generated_at": "2026-07-22T00:00:00Z",
        "entities": {"HBM": "memory", "CPO": "optical", "Shared Topic": "both"},
        "chunks": {"chunk-1": "memory"},
        # Stored in STORAGE order (src<SEP>tgt as found in the KV store), which
        # is NOT alphabetically sorted here (H > C). The old sorted-
        # canonicalization would fail to find this on a reversed query.
        "relations": {"HBM<SEP>CPO": "both"},
    }))
    return p


@pytest.fixture()
def settings(provenance_file):
    return Settings(lightrag_url="http://upstream.test",
                    lightrag_api_key="test-key",
                    provenance_path=provenance_file,
                    rate_per_min=3, rate_concurrent=2,
                    max_query_chars=100, max_history_turns=4)


def test_provenance_lookup_and_reload(provenance_file):
    store = ProvenanceStore(provenance_file)
    assert store.domain("entities", "HBM") == "memory"
    assert store.domain("entities", "Nope") == "unknown"
    # The provenance writer stores relation keys in storage order
    # ("HBM<SEP>CPO" here), NOT sorted. Lookups must resolve regardless of the
    # endpoint order the upstream API hands us: exact match, then reversed.
    assert store.domain("relations", "HBM<SEP>CPO") == "both"   # exact
    assert store.domain("relations", "CPO<SEP>HBM") == "both"   # reversed
    assert store.domain("relations", "HBM<SEP>Nope") == "unknown"  # miss
    time.sleep(0.01)
    provenance_file.write_text(json.dumps(
        {"generated_at": "x", "entities": {"HBM": "both"},
         "chunks": {}, "relations": {}}))
    assert store.domain("entities", "HBM") == "both"


def test_rate_limiter_window_and_concurrency():
    rl = RateLimiter(per_min=2, concurrent=1)
    assert rl.acquire("1.2.3.4")
    rl.release("1.2.3.4")
    assert rl.acquire("1.2.3.4")
    rl.release("1.2.3.4")
    assert not rl.acquire("1.2.3.4")          # per-min budget spent
    assert rl.acquire("5.6.7.8")              # other IP unaffected
    assert not rl.acquire("5.6.7.8")          # concurrent=1, slot held
    rl.release("5.6.7.8")


def test_health_reports_upstream(settings, monkeypatch):
    app = create_app(settings)
    client = TestClient(app)

    async def fake_get(self, url, **kw):
        import httpx
        assert url == "/health"
        return httpx.Response(200, json={"status": "healthy"})

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_hypothesis_settings_read_from_environment(monkeypatch, tmp_path):
    """Protects the gateway from ignoring its hypothesis endpoint settings."""
    hypotheses_dir = tmp_path / "hypotheses"
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")
    monkeypatch.setenv("UPLOAD_TOKEN", "upload-secret")
    monkeypatch.setenv("HYPOTHESIS_TOKEN", "hypothesis-secret")
    monkeypatch.setenv("HYPOTHESIS_TIMEOUT_SEC", "45")
    monkeypatch.setenv("HYPOTHESES_DIR", str(hypotheses_dir))

    settings = Settings.from_env()

    assert settings.hypothesis_token == "hypothesis-secret"
    assert settings.hypothesis_token != settings.upload_token
    assert settings.hypothesis_timeout_sec == 45
    assert settings.hypotheses_dir == hypotheses_dir


def test_hypothesis_settings_defaults_from_clean_environment(monkeypatch):
    """Protects defaults when no hypothesis-related environment is configured."""
    for name in ("HYPOTHESIS_TOKEN", "HYPOTHESIS_TIMEOUT_SEC", "HYPOTHESES_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIGHTRAG_API_KEY", "test-key")

    settings = Settings.from_env()

    assert settings.hypothesis_token == ""
    assert settings.hypothesis_timeout_sec == 1200
    assert settings.hypotheses_dir == Path("data/report-files/hypotheses")


def test_hypothesis_token_defaults_and_fails_closed_when_unset():
    """Protects the hypothesis route from accepting a caller token by default."""
    settings = Settings(lightrag_api_key="test-key")
    validate_token = getattr(config, "is_hypothesis_token_valid", None)

    assert settings.hypothesis_token == ""
    assert settings.hypothesis_timeout_sec == 1200
    assert validate_token is not None
    assert not validate_token("presented-token", settings.hypothesis_token)
    assert validate_token("matching-token", "matching-token")
    assert not validate_token("wrong-token", "matching-token")


UPSTREAM_STREAM = [
    {"references": [{"reference_id": "1", "file_path": "arxiv.org/x"}]},
    {"response": "Hello "},
    {"response": "world"},
]
UPSTREAM_DATA = {
    "status": "success",
    "data": {
        "entities": [{"entity_name": "HBM", "entity_type": "topic",
                      "description": "d", "source_id": "s",
                      "file_path": "f", "created_at": "t"}],
        "relationships": [{"src_id": "HBM", "tgt_id": "CPO",
                           "description": "r", "keywords": "k", "weight": 1,
                           "source_id": "s", "file_path": "f",
                           "created_at": "t"}],
        "chunks": [], "references": [],
    },
    "metadata": {"query_mode": "mix",
                 "keywords": {"high_level": ["memory wall"]}},
}


def make_client(settings, monkeypatch, stream_lines=None, data_payload=None,
                stream_error=None, data_error=None):
    app = create_app(settings)
    import httpx

    async def fake_send(self, request, **kw):
        assert request.headers.get("x-api-key") == "test-key"
        if request.url.path == "/query/stream":
            if stream_error:
                raise stream_error
            body = b"".join(
                (json.dumps(line) + "\n").encode()
                for line in (stream_lines or UPSTREAM_STREAM))
            return httpx.Response(200, content=body, request=request)
        if request.url.path == "/query/data":
            if data_error:
                raise data_error
            return httpx.Response(200, json=data_payload or UPSTREAM_DATA,
                                  request=request)
        raise AssertionError(f"unexpected upstream path {request.url.path}")

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return TestClient(app)


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((ev, data))
    return events


def test_chat_happy_path(settings, monkeypatch):
    client = make_client(settings, monkeypatch)
    r = client.post("/api/chat", json={"query": "q", "mode": "mix"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names.count("token") == 2
    assert "sources" in names and names[-1] == "done"
    src = dict(events)["sources"]
    assert src["entities"][0]["domain"] == "memory"          # from provenance fixture
    assert src["relationships"][0]["domain"] == "both"       # HBM<SEP>CPO
    assert src["keywords"] == {"high_level": ["memory wall"]}


def test_chat_rejects_bad_input(settings, monkeypatch):
    client = make_client(settings, monkeypatch)
    assert client.post("/api/chat", json={"query": "q", "mode": "evil"}).status_code == 422
    assert client.post("/api/chat", json={"query": "x" * 101, "mode": "mix"}).status_code == 422
    assert client.post("/api/chat", json={"query": "q", "mode": "mix",
                                          "extra": 1}).status_code == 422
    long_history = [{"role": "user", "content": "h"}] * 5      # max_history_turns=4
    assert client.post("/api/chat", json={"query": "q", "mode": "mix",
                                          "history": long_history}).status_code == 422


def test_chat_history_total_size_capped(settings, monkeypatch):
    # Two assistant turns of 40 chars = 80 > the 50-char aggregate cap. Neither
    # is a user turn, so the per-turn user cap (100) wouldn't catch them — only
    # the total-history cap does.
    s = dataclasses.replace(settings, max_history_chars=50)
    client = make_client(s, monkeypatch)
    history = [{"role": "assistant", "content": "a" * 40},
               {"role": "assistant", "content": "b" * 40}]
    r = client.post("/api/chat", json={"query": "q", "mode": "mix",
                                       "history": history})
    assert r.status_code == 422
    # A long-but-under-cap assistant turn still passes.
    ok = client.post("/api/chat", json={"query": "q", "mode": "mix",
                     "history": [{"role": "assistant", "content": "a" * 40}]})
    assert ok.status_code == 200


def test_chat_rate_limited(settings, monkeypatch):
    client = make_client(settings, monkeypatch)
    for _ in range(3):                                         # rate_per_min=3
        assert client.post("/api/chat", json={"query": "q", "mode": "mix"}).status_code == 200
    r = client.post("/api/chat", json={"query": "q", "mode": "mix"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "60"


def test_chat_upstream_down(settings, monkeypatch):
    import httpx
    client = make_client(settings, monkeypatch,
                         stream_error=httpx.ConnectError("refused"),
                         data_error=httpx.ConnectError("refused"))
    events = parse_sse(client.post("/api/chat",
                                   json={"query": "q", "mode": "mix"}).text)
    assert events[-1][0] == "error"
    assert events[-1][1]["retryable"] is True


def test_chat_sources_failure_still_streams(settings, monkeypatch):
    import httpx
    client = make_client(settings, monkeypatch,
                         data_error=httpx.ConnectError("refused"))
    events = parse_sse(client.post("/api/chat",
                                   json={"query": "q", "mode": "mix"}).text)
    names = [e for e, _ in events]
    assert names.count("token") == 2 and names[-1] == "done"
    src = dict(events)["sources"]
    assert "error" in src


def test_chat_sources_emitted_when_resolved_after_stream(settings, monkeypatch):
    """When the stream closes before the mid-stream drain ever fires (here: an
    empty stream, zero loop iterations), the sources event must still be emitted
    from the post-loop `await`, ahead of `done`. This is the ONLY path that
    covers the post-loop branch; the happy path resolves sources mid-stream."""
    import httpx

    async def fake_send(self, request, **kw):
        assert request.headers.get("x-api-key") == "test-key"
        if request.url.path == "/query/stream":
            return httpx.Response(200, content=b"", request=request)  # no lines
        if request.url.path == "/query/data":
            return httpx.Response(200, json=UPSTREAM_DATA, request=request)
        raise AssertionError(f"unexpected path {request.url.path}")

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    client = TestClient(create_app(settings))
    events = parse_sse(client.post("/api/chat",
                                   json={"query": "q", "mode": "mix"}).text)
    names = [e for e, _ in events]
    assert names.count("token") == 0
    assert names == ["sources", "done"]
    src = dict(events)["sources"]
    assert src["entities"][0]["domain"] == "memory"


def test_chat_releases_slot_between_requests(provenance_file, monkeypatch):
    """The rate-limit slot must be released in `finally`; otherwise a second
    sequential request from the same IP hits the concurrency cap and 429s even
    though the per-minute window is nowhere near full."""
    s = Settings(lightrag_url="http://upstream.test", lightrag_api_key="test-key",
                 provenance_path=provenance_file,
                 rate_per_min=10, rate_concurrent=1,       # cap=1: no slack
                 max_query_chars=100, max_history_turns=4)
    client = make_client(s, monkeypatch)
    for _ in range(3):
        r = client.post("/api/chat", json={"query": "q", "mode": "mix"})
        assert r.status_code == 200                        # slot freed each time


def test_chat_no_double_release_active_lands_at_zero(provenance_file, monkeypatch):
    """A completed request must release its slot exactly once. With the
    generator's `finally` AND a BackgroundTask both wired to release, a missing
    idempotency guard would release twice, driving `_active[ip]` below the true
    value and corrupting a concurrent request's accounting. Assert the Counter
    lands at exactly 0 after each request and that three sequential requests
    from one IP all succeed under a concurrency cap of 1."""
    s = Settings(lightrag_url="http://upstream.test", lightrag_api_key="test-key",
                 provenance_path=provenance_file,
                 rate_per_min=50, rate_concurrent=1,   # cap=1: no slack at all
                 max_query_chars=100, max_history_turns=4)
    client = make_client(s, monkeypatch)
    limiter = client.app.state.limiter
    ip = "testclient"                                  # TestClient's client.host
    for _ in range(3):
        r = client.post("/api/chat", json={"query": "q", "mode": "mix"})
        assert r.status_code == 200
        parse_sse(r.text)                              # drain to run finally+bg
        # Exactly zero -- not negative. A double release would show -1 here.
        assert limiter._active[ip] == 0


def test_chat_double_release_does_not_steal_concurrent_slot(settings, monkeypatch):
    """The sharp half of the no-double-release requirement, driven through the
    REAL handler's `release_once` closure. On the normal path BOTH the
    generator's `finally` and the BackgroundTask fire the releaser for the same
    request. RateLimiter.release floors at zero, so a purely sequential double
    release is invisible -- which is why the sequential test above cannot catch
    a missing guard. But once a DIFFERENT request from the same IP has taken the
    freed slot, an unguarded second release steals it. Here: drive request A's
    generator to its `finally` (release #1), let B grab the slot, then fire A's
    BackgroundTask. The `released` guard must make that a no-op; without it,
    B's slot is freed and `_active[ip]` drops from 1 to 0."""
    import asyncio

    app = create_app(settings)
    make_client(settings, monkeypatch)         # patch httpx send so gen() runs

    route = next(r for r in app.routes if getattr(r, "path", None) == "/api/chat")
    handler = route.endpoint
    ip = "2.2.2.2"

    class FakeRequest:
        class client:
            host = ip

    async def scenario():
        resp = await handler(req=ChatRequest(query="q", mode="mix"),
                             request=FakeRequest())
        # Drain the body -> runs gen() to completion -> its `finally` fires
        # release_once once. Slot freed back to zero.
        async for _ in resp.body_iterator:
            pass
        assert app.state.limiter._active[ip] == 0
        # A DIFFERENT request B from the same IP now takes the freed slot.
        assert app.state.limiter.acquire(ip)
        assert app.state.limiter._active[ip] == 1
        # A's BackgroundTask fires on response finish. Guarded: no-op (already
        # released). Unguarded: release(ip) steals B's slot -> _active == 0.
        await resp.background()
        return app.state.limiter._active[ip]

    assert asyncio.run(scenario()) == 1        # B's slot survived A's second release


def test_chat_response_wires_background_release(settings, monkeypatch):
    """Construction-level proof for the never-started-generator leak window: the
    handler must attach the one-shot releaser as a Starlette BackgroundTask so
    the slot is freed even when the generator body never runs. Call the
    endpoint's coroutine directly (bypassing TestClient's eager stream drain),
    inspect `response.background`, then simulate the leak: acquire happened, the
    generator is never iterated, and running the background task frees the
    slot."""
    import asyncio
    import inspect
    from starlette.background import BackgroundTask

    app = create_app(settings)
    make_client(settings, monkeypatch)         # patches httpx.AsyncClient.send

    # Grab the /api/chat handler coroutine function off the app.
    route = next(r for r in app.routes if getattr(r, "path", None) == "/api/chat")
    handler = route.endpoint

    ip = "1.1.1.1"

    class FakeClient:
        host = ip

    class FakeRequest:
        client = FakeClient()

    req = ChatRequest(query="q", mode="mix")
    # Call the handler; do NOT iterate the returned StreamingResponse body, so
    # the generator's `finally` never runs -- exactly the leak scenario.
    sig = inspect.signature(handler)
    kwargs = {"req": req, "request": FakeRequest()} if "request" in sig.parameters \
        else {"req": req}
    response = asyncio.run(handler(**kwargs))

    # acquire() ran inside the handler; the slot is held.
    assert app.state.limiter._active[ip] == 1
    # The background task must be wired to the one-shot releaser.
    assert isinstance(response.background, BackgroundTask)
    # Running it (as Starlette does on finish/disconnect) frees the leaked slot.
    asyncio.run(response.background())
    assert app.state.limiter._active[ip] == 0
    # And it is one-shot: a second run does not drive the counter negative.
    asyncio.run(response.background())
    assert app.state.limiter._active[ip] == 0


def test_chat_history_goes_to_stream_not_data(settings, monkeypatch):
    """conversation_history is forwarded to /query/stream only, never to
    /query/data, and upstream bodies are built from validated fields only."""
    import httpx
    seen = {}

    async def fake_send(self, request, **kw):
        assert request.headers.get("x-api-key") == "test-key"
        seen[request.url.path] = json.loads(request.content)
        if request.url.path == "/query/stream":
            body = b"".join((json.dumps(l) + "\n").encode()
                            for l in UPSTREAM_STREAM)
            return httpx.Response(200, content=body, request=request)
        if request.url.path == "/query/data":
            return httpx.Response(200, json=UPSTREAM_DATA, request=request)
        raise AssertionError(f"unexpected path {request.url.path}")

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    client = TestClient(create_app(settings))
    r = client.post("/api/chat", json={
        "query": "q", "mode": "hybrid", "top_k": 7,
        "history": [{"role": "user", "content": "earlier"}]})
    assert r.status_code == 200
    parse_sse(r.text)  # drain

    stream_body = seen["/query/stream"]
    data_body = seen["/query/data"]
    # history reaches the stream endpoint...
    assert stream_body["conversation_history"] == [
        {"role": "user", "content": "earlier"}]
    assert stream_body["query"] == "q" and stream_body["mode"] == "hybrid"
    assert stream_body["top_k"] == 7
    # ...but never the data endpoint, and no stray client keys leak through
    assert "conversation_history" not in data_body
    assert "history" not in data_body and "extra" not in data_body
    assert data_body["query"] == "q" and data_body["mode"] == "hybrid"
    assert data_body["top_k"] == 7


# --- /api/upload: auth + rate limiting on the one write path -------------------

import dataclasses


def _upload_client(settings, monkeypatch, ingested=None):
    """TestClient whose upstream stubs the /documents/text ingest endpoint."""
    import httpx

    async def fake_send(self, request, **kw):
        assert request.headers.get("x-api-key") == "test-key"
        if request.url.path == "/documents/text":
            if ingested is not None:
                ingested.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "success"},
                                  request=request)
        raise AssertionError(f"unexpected upstream path {request.url.path}")

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return TestClient(create_app(settings))


def test_upload_disabled_when_no_token(settings, monkeypatch):
    # Fail closed: the default settings carry no upload_token, so even a well
    # formed request is refused and never reaches the upstream ingest endpoint.
    client = _upload_client(settings, monkeypatch)
    r = client.post("/api/upload", files={"file": ("n.txt", b"hello", "text/plain")})
    assert r.status_code == 403


def test_upload_rejects_wrong_token(settings, monkeypatch):
    s = dataclasses.replace(settings, upload_token="s3cret")
    client = _upload_client(s, monkeypatch)
    r = client.post("/api/upload", files={"file": ("n.txt", b"hi", "text/plain")},
                    headers={"X-Upload-Token": "nope"})
    assert r.status_code == 403


def test_upload_happy_path_with_token(settings, monkeypatch):
    ingested = []
    s = dataclasses.replace(settings, upload_token="s3cret")
    client = _upload_client(s, monkeypatch, ingested=ingested)
    r = client.post("/api/upload",
                    files={"file": ("notes.txt", b"real content", "text/plain")},
                    headers={"X-Upload-Token": "s3cret"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["filename"] == "notes.txt"
    assert ingested and ingested[0]["text"] == "real content"


def test_upload_sanitizes_filename(settings, monkeypatch):
    # A path-traversal filename is reduced to a safe basename before it reaches
    # LightRAG as metadata.file_path.
    ingested = []
    s = dataclasses.replace(settings, upload_token="s3cret")
    client = _upload_client(s, monkeypatch, ingested=ingested)
    r = client.post(
        "/api/upload",
        files={"file": ("../../../etc/passwd.txt", b"data", "text/plain")},
        headers={"X-Upload-Token": "s3cret"})
    assert r.status_code == 200
    assert r.json()["filename"] == "passwd.txt"
    assert ingested[0]["metadata"]["file_path"] == "passwd.txt"


def test_upload_too_large(settings, monkeypatch):
    # Small body (100 B) but max=8: Content-Length stays under the envelope
    # allowance, so the middleware lets it through and the handler's bounded
    # read is what rejects it.
    s = dataclasses.replace(settings, upload_token="s3cret", max_upload_bytes=8)
    client = _upload_client(s, monkeypatch)
    r = client.post("/api/upload",
                    files={"file": ("big.txt", b"x" * 100, "text/plain")},
                    headers={"X-Upload-Token": "s3cret"})
    assert r.status_code == 413


def test_upload_rejected_early_by_content_length(settings, monkeypatch):
    # A body whose declared length exceeds max + envelope is rejected in
    # middleware BEFORE the body is parsed/spooled and before the handler's
    # auth check — so no token is supplied yet the result is 413, not 403.
    s = dataclasses.replace(settings, upload_token="s3cret", max_upload_bytes=100)
    client = _upload_client(s, monkeypatch)
    r = client.post("/api/upload",
                    files={"file": ("big.txt", b"x" * 20000, "text/plain")})
    assert r.status_code == 413


def test_upload_rate_limited_and_releases_slot(settings, monkeypatch):
    # rate_per_min=3 in the fixture; authorized uploads share the chat budget.
    ingested = []
    s = dataclasses.replace(settings, upload_token="s3cret")
    client = _upload_client(s, monkeypatch, ingested=ingested)
    hdr = {"X-Upload-Token": "s3cret"}
    for _ in range(3):
        assert client.post("/api/upload",
                           files={"file": ("n.txt", b"data", "text/plain")},
                           headers=hdr).status_code == 200
    r = client.post("/api/upload", files={"file": ("n.txt", b"data", "text/plain")},
                    headers=hdr)
    assert r.status_code == 429 and r.headers["retry-after"] == "60"
    # The slot was released after each success, so concurrency never leaked.
    assert client.app.state.limiter._active["testclient"] == 0


# --- /api/hypothesis/dossiers: authenticated local dossier reads -------------

def _dossier_client(settings, tmp_path, *, token="hypothesis-secret",
                    root_name="hypotheses"):
    """A real gateway backed by a small, controlled dossier tree."""
    dossiers = tmp_path / root_name
    (dossiers / "memory").mkdir(parents=True)
    (dossiers / "memory" / "memory-gap.md").write_text("# Memory gap\n")
    (dossiers / "optical").mkdir()
    (dossiers / "optical" / "optical-gap.md").write_text("# Optical gap\n")
    return TestClient(create_app(dataclasses.replace(
        settings, hypothesis_token=token, hypotheses_dir=dossiers))), dossiers


def test_dossier_list_fails_closed_for_missing_or_wrong_hypothesis_token(
        settings, tmp_path):
    """Removing the dossier auth check would expose local research files."""
    client, _ = _dossier_client(settings, tmp_path)
    assert client.get("/api/hypothesis/dossiers").status_code == 403
    assert client.get("/api/hypothesis/dossiers",
                      headers={"X-Hypothesis-Token": "wrong"}).status_code == 403


def test_dossier_list_fails_closed_when_hypothesis_token_is_unconfigured(
        settings, tmp_path):
    """An empty server secret must not turn dossier reads into public routes."""
    client, _ = _dossier_client(settings, tmp_path, token="")
    r = client.get("/api/hypothesis/dossiers",
                   headers={"X-Hypothesis-Token": "hypothesis-secret"})
    assert r.status_code == 403


def test_dossier_list_returns_profile_filename_and_mtime_for_markdown_files(
        settings, tmp_path):
    """Dropping directory enumeration or metadata would break the dossier picker."""
    client, dossiers = _dossier_client(settings, tmp_path)
    memory = dossiers / "memory" / "memory-gap.md"
    optical = dossiers / "optical" / "optical-gap.md"
    import os
    os.utime(memory, (1_700_000_000, 1_700_000_000))
    os.utime(optical, (1_700_000_100, 1_700_000_100))

    r = client.get("/api/hypothesis/dossiers",
                   headers={"X-Hypothesis-Token": "hypothesis-secret"})

    assert r.status_code == 200
    assert r.json() == [
        {"profile": "memory", "filename": "memory-gap.md",
         "mtime": 1_700_000_000.0},
        {"profile": "optical", "filename": "optical-gap.md",
         "mtime": 1_700_000_100.0},
    ]


def test_dossier_fetch_returns_raw_markdown_for_an_listed_dossier(
        settings, tmp_path):
    """Replacing the dossier body with JSON or a download would break in-app rendering."""
    client, _ = _dossier_client(settings, tmp_path)

    r = client.get("/api/hypothesis/dossiers/memory/memory-gap.md",
                   headers={"X-Hypothesis-Token": "hypothesis-secret"})

    assert r.status_code == 200
    assert r.text == "# Memory gap\n"
    assert r.headers["content-type"].startswith("text/markdown")


def test_dossier_fetch_fails_closed_for_missing_or_wrong_hypothesis_token(
        settings, tmp_path):
    """Removing fetch auth separately from list auth would expose dossier bodies."""
    client, _ = _dossier_client(settings, tmp_path)
    url = "/api/hypothesis/dossiers/memory/memory-gap.md"
    assert client.get(url).status_code == 403
    assert client.get(url, headers={"X-Hypothesis-Token": "wrong"}).status_code == 403


def test_dossier_fetch_fails_closed_when_hypothesis_token_is_unconfigured(
        settings, tmp_path):
    """An unset server secret must disable the fetch route as well as listing."""
    client, _ = _dossier_client(settings, tmp_path, token="")
    r = client.get("/api/hypothesis/dossiers/memory/memory-gap.md",
                   headers={"X-Hypothesis-Token": "hypothesis-secret"})
    assert r.status_code == 403


def test_dossier_listing_and_fetch_use_a_fresh_server_listing(settings, tmp_path):
    """Caching a prior listing would hide dossiers created after the UI opened."""
    client, dossiers = _dossier_client(settings, tmp_path)
    headers = {"X-Hypothesis-Token": "hypothesis-secret"}
    assert client.get("/api/hypothesis/dossiers", headers=headers).status_code == 200
    (dossiers / "memory" / "new-gap.md").write_text("# Newly generated\n")

    fetched = client.get("/api/hypothesis/dossiers/memory/new-gap.md",
                         headers=headers)

    assert fetched.status_code == 200
    assert fetched.text == "# Newly generated\n"


def test_dossier_fetch_rejects_file_symlink_swapped_at_open_boundary(
        settings, tmp_path, monkeypatch):
    """The file must be opened no-follow after its listing membership is checked."""
    client, dossiers = _dossier_client(settings, tmp_path)
    headers = {"X-Hypothesis-Token": "hypothesis-secret"}
    original_file = dossiers / "memory" / "memory-gap.md"
    outside = tmp_path / "outside.md"
    outside.write_text("TOP SECRET SWAPPED DOSSIER")
    original_open = chatbot_app.os.open
    swapped = False

    def swap_before_file_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "memory-gap.md" and dir_fd is not None and not swapped:
            swapped = True
            original_file.unlink()
            original_file.symlink_to(outside)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(chatbot_app.os, "open", swap_before_file_open)
    r = client.get("/api/hypothesis/dossiers/memory/memory-gap.md", headers=headers)

    assert swapped
    assert r.status_code == 404
    assert "TOP SECRET SWAPPED DOSSIER" not in r.text


def test_dossier_fetch_rejects_ancestor_symlink_swapped_at_open_boundary(
        settings, tmp_path, monkeypatch):
    """Pinning only the root's final component follows a swapped ancestor link."""
    client, dossiers = _dossier_client(settings, tmp_path,
                                       root_name="container/hypotheses")
    headers = {"X-Hypothesis-Token": "hypothesis-secret"}
    original_container = tmp_path / "container"
    outside_container = tmp_path / "outside-container"
    (outside_container / "hypotheses" / "memory").mkdir(parents=True)
    (outside_container / "hypotheses" / "memory" / "memory-gap.md").write_text(
        "TOP SECRET ANCESTOR SWAP")
    original_open = chatbot_app.os.open
    swapped = False

    def swap_before_root_or_component_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and (Path(path) == dossiers or path == "container"):
            swapped = True
            original_container.rename(tmp_path / "original-container")
            original_container.symlink_to(outside_container, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(chatbot_app.os, "open", swap_before_root_or_component_open)
    r = client.get("/api/hypothesis/dossiers/memory/memory-gap.md", headers=headers)

    assert swapped
    assert r.status_code == 404
    assert "TOP SECRET ANCESTOR SWAP" not in r.text


def test_dossier_fetch_rejects_ancestor_swap_between_listing_and_root_pin(
        settings, tmp_path, monkeypatch):
    """Re-resolving settings after listing would pin the attacker's new tree."""
    client, dossiers = _dossier_client(settings, tmp_path,
                                       root_name="container/hypotheses")
    headers = {"X-Hypothesis-Token": "hypothesis-secret"}
    original_container = tmp_path / "container"
    outside_container = tmp_path / "outside-container"
    (outside_container / "hypotheses" / "memory").mkdir(parents=True)
    (outside_container / "hypotheses" / "memory" / "memory-gap.md").write_text(
        "TOP SECRET BETWEEN LISTING AND PIN")
    original_listing = chatbot_app._dossier_listing
    swapped = False

    def list_then_swap(root):
        nonlocal swapped
        entries = original_listing(root)
        if not swapped:
            swapped = True
            original_container.rename(tmp_path / "original-container")
            original_container.symlink_to(outside_container, target_is_directory=True)
        return entries

    monkeypatch.setattr(chatbot_app, "_dossier_listing", list_then_swap)
    r = client.get("/api/hypothesis/dossiers/memory/memory-gap.md", headers=headers)

    assert swapped
    assert r.status_code == 404
    assert "TOP SECRET BETWEEN LISTING AND PIN" not in r.text


def test_dossier_fetch_rejects_real_root_replacement_after_listing(
        settings, tmp_path, monkeypatch):
    """A new real directory at the same root path must fail identity checks."""
    client, dossiers = _dossier_client(settings, tmp_path)
    headers = {"X-Hypothesis-Token": "hypothesis-secret"}
    replacement = tmp_path / "replacement"
    (replacement / "memory").mkdir(parents=True)
    (replacement / "memory" / "memory-gap.md").write_text(
        "TOP SECRET REPLACED ROOT")
    original_listing = chatbot_app._dossier_listing
    swapped = False

    def list_then_replace(root):
        nonlocal swapped
        entries = original_listing(root)
        if not swapped:
            swapped = True
            dossiers.rename(tmp_path / "old-hypotheses")
            replacement.rename(dossiers)
        return entries

    monkeypatch.setattr(chatbot_app, "_dossier_listing", list_then_replace)
    r = client.get("/api/hypothesis/dossiers/memory/memory-gap.md", headers=headers)

    assert swapped
    assert r.status_code == 404
    assert "TOP SECRET REPLACED ROOT" not in r.text


@pytest.mark.parametrize("failure", ["missing_nonblock", "not_implemented",
                                     "type_error"])
def test_dossier_fetch_fails_closed_when_descriptor_opening_is_unavailable(
        settings, tmp_path, monkeypatch, failure):
    """Missing flags or dir-fd support must return 404 rather than a server error."""
    client, _ = _dossier_client(settings, tmp_path)
    client = TestClient(client.app, raise_server_exceptions=False)
    if failure == "missing_nonblock":
        monkeypatch.delattr(chatbot_app.os, "O_NONBLOCK")
    else:
        error = NotImplementedError if failure == "not_implemented" else TypeError

        def unsupported_open(*args, **kwargs):
            raise error("dir_fd unavailable")

        monkeypatch.setattr(chatbot_app.os, "open", unsupported_open)

    r = client.get("/api/hypothesis/dossiers/memory/memory-gap.md",
                   headers={"X-Hypothesis-Token": "hypothesis-secret"})

    assert r.status_code == 404


def test_dossier_listing_rejects_profile_and_file_symlinks(settings, tmp_path):
    """Following any symlink lets a listing escape its intended profile identity."""
    client, dossiers = _dossier_client(settings, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside.md").write_text("TOP SECRET")
    (dossiers / "outside-profile").symlink_to(outside, target_is_directory=True)
    (dossiers / "linked-optical-profile").symlink_to(
        dossiers / "optical", target_is_directory=True)
    (dossiers / "memory" / "outside.md").symlink_to(outside / "outside.md")
    (dossiers / "memory" / "same-profile-link.md").symlink_to(
        dossiers / "memory" / "memory-gap.md")
    (dossiers / "memory" / "cross-profile-link.md").symlink_to(
        dossiers / "optical" / "optical-gap.md")
    headers = {"X-Hypothesis-Token": "hypothesis-secret"}

    r = client.get("/api/hypothesis/dossiers", headers=headers)

    assert r.status_code == 200
    assert r.json() == [
        {"profile": "memory", "filename": "memory-gap.md",
         "mtime": (dossiers / "memory" / "memory-gap.md").stat().st_mtime},
        {"profile": "optical", "filename": "optical-gap.md",
         "mtime": (dossiers / "optical" / "optical-gap.md").stat().st_mtime},
    ]
    assert client.get("/api/hypothesis/dossiers/memory/cross-profile-link.md",
                      headers=headers).status_code == 404


@pytest.mark.parametrize("component", [
    "bad\nname", " leading", "trailing ", "x" * 256, "nested/name",
    r"nested\name",
])
def test_dossier_component_validation_rejects_unsafe_decoded_values(component):
    """Normalizing unsafe decoded components would make route validation ambiguous."""
    assert not _is_safe_dossier_component(component)


def test_dossier_fetch_rejects_names_absent_from_server_listing(settings, tmp_path):
    """Opening a user-composed path instead of a listed file exposes arbitrary files."""
    client, _ = _dossier_client(settings, tmp_path)
    assert client.get("/api/hypothesis/dossiers",
                      headers={"X-Hypothesis-Token": "hypothesis-secret"}).status_code == 200
    r = client.get("/api/hypothesis/dossiers/memory/not-listed.md",
                   headers={"X-Hypothesis-Token": "hypothesis-secret"})
    assert r.status_code == 404


@pytest.mark.parametrize("path", [
    "/api/hypothesis/dossiers/../outside.md",
    "/api/hypothesis/dossiers/../../outside.md",
    "/api/hypothesis/dossiers/%2e%2e/outside.md",
    "/api/hypothesis/dossiers/memory/%2e%2e%2foutside.md",
    "/api/hypothesis/dossiers/memory/%2Ftmp%2Foutside.md",
    "/api/hypothesis/dossiers/..%5Coutside/memory-gap.md",
    "/api/hypothesis/dossiers/memory/..%5Coutside.md",
])
def test_dossier_fetch_rejects_traversal_variants_without_reading_outside_root(
        settings, tmp_path, path):
    """Any bypass of listing membership or root containment leaks the sentinel."""
    client, dossiers = _dossier_client(settings, tmp_path)
    outside = dossiers.parent / "outside.md"
    outside.write_text("TOP SECRET OUTSIDE DOSSIER ROOT")
    assert client.get("/api/hypothesis/dossiers",
                      headers={"X-Hypothesis-Token": "hypothesis-secret"}).status_code == 200

    r = client.get(path, headers={"X-Hypothesis-Token": "hypothesis-secret"})

    assert r.status_code in (400, 404)
    assert "TOP SECRET OUTSIDE DOSSIER ROOT" not in r.text

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from chatbot.app import create_app
from chatbot.config import Settings
from chatbot.research_jobs import ResearchJobConflict


class StubResearchJobs:
    def __init__(self):
        self.records = {}
        self.owners = {}
        self.submissions = 0
        self.stopped = False

    async def submit(self, owner, kind, request):
        active = self.active_for_owner(owner)
        if active:
            raise ResearchJobConflict(active)
        self.submissions += 1
        job_id = f"job-{self.submissions}"
        self.owners[job_id] = owner
        self.records[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "queue_position": 1,
            "result": None,
            "estimated_cost_usd": 0.39,
            "actual_cost_usd": 0.0,
            "cost_status": "cost_pending",
        }
        return job_id

    def get(self, job_id, owner):
        if self.owners.get(job_id) != owner:
            return None
        return self.records[job_id]

    def active_for_owner(self, owner):
        for job_id, job_owner in self.owners.items():
            if (
                job_owner == owner
                and self.records[job_id]["status"] in {"queued", "running"}
            ):
                return job_id
        return None

    async def subscribe(self, job_id, owner) -> AsyncIterator[dict]:
        record = self.get(job_id, owner)
        if record is None:
            return
        yield {"event": "snapshot", "data": record}
        record.update({
            "status": "done",
            "queue_position": None,
            "actual_cost_usd": 0.36,
            "cost_status": "actual",
            "result": {
                "markdown": "# Answer",
                "coverage": {
                    "exhaustive": True,
                    "nodes_scanned": {
                        "memory": 11041,
                        "optical": 4010,
                        "storage": 3630,
                    },
                    "edges_scanned": {
                        "memory": 15910,
                        "optical": 7012,
                        "storage": 9324,
                    },
                },
                "citations": [{"chunk_id": "chunk-1"}],
            },
        })
        yield {"event": "done", "data": record}

    async def cancel(self, job_id, owner):
        record = self.get(job_id, owner)
        if record is None:
            return False
        record["status"] = "cancelled"
        return True

    async def shutdown(self):
        self.stopped = True


def _settings(tmp_path, **overrides):
    values = {
        "lightrag_api_key": "test",
        "provenance_path": tmp_path / "missing.json",
        "exhaustive_enabled": True,
        "research_token": "research-secret",
    }
    values.update(overrides)
    return Settings(**values)


def test_loopback_can_submit_and_response_has_queue_position(tmp_path):
    manager = StubResearchJobs()
    app = create_app(_settings(tmp_path), research_jobs=manager)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post(
            "/api/research/chat",
            json={"query": "How do the systems interact?", "history": []},
        )

    assert response.status_code == 202
    assert response.json() == {"job_id": "job-1", "queue_position": 1}


def test_remote_requires_token_and_forwarded_for_cannot_fake_loopback(tmp_path):
    manager = StubResearchJobs()
    app = create_app(_settings(tmp_path), research_jobs=manager)

    with TestClient(app, client=("100.64.1.9", 50000)) as remote:
        denied = remote.post(
            "/api/research/chat",
            headers={"X-Forwarded-For": "127.0.0.1"},
            json={"query": "question"},
        )
        allowed = remote.post(
            "/api/research/chat",
            headers={"X-Research-Token": "research-secret"},
            json={"query": "question"},
        )

    assert denied.status_code == 403
    assert allowed.status_code == 202
    assert manager.submissions == 1


def test_tailscale_proxy_header_does_not_gain_loopback_exemption(tmp_path):
    manager = StubResearchJobs()
    app = create_app(_settings(tmp_path), research_jobs=manager)

    with TestClient(app, client=("127.0.0.1", 50000)) as proxy:
        denied = proxy.post(
            "/api/research/chat",
            headers={"Tailscale-User-Login": "user@example.com"},
            json={"query": "question"},
        )
        allowed = proxy.post(
            "/api/research/chat",
            headers={
                "Tailscale-User-Login": "user@example.com",
                "X-Research-Token": "research-secret",
            },
            json={"query": "question"},
        )

    assert denied.status_code == 403
    assert allowed.status_code == 202


def test_active_conflict_owner_isolation_and_final_contract(tmp_path):
    manager = StubResearchJobs()
    app = create_app(_settings(tmp_path), research_jobs=manager)
    headers = {"X-Research-Token": "research-secret"}

    with TestClient(app, client=("100.64.1.9", 50000)) as owner:
        first = owner.post(
            "/api/research/chat", headers=headers, json={"query": "question"}
        )
        conflict = owner.post(
            "/api/research/chat", headers=headers, json={"query": "again"}
        )
        active = owner.get("/api/research/active", headers=headers)
        streamed = owner.get(
            f"/api/research/{first.json()['job_id']}/events",
            headers=headers,
        )
        final = owner.get(
            f"/api/research/{first.json()['job_id']}", headers=headers
        )

    with TestClient(app, client=("100.64.1.10", 50000)) as other:
        hidden = other.get(
            f"/api/research/{first.json()['job_id']}", headers=headers
        )

    assert conflict.status_code == 409
    assert conflict.json()["active_job_id"] == first.json()["job_id"]
    assert active.json()["active_job_id"] == first.json()["job_id"]
    assert "event: snapshot" in streamed.text
    assert "event: done" in streamed.text
    body = final.json()
    assert body["result"]["coverage"]["exhaustive"] is True
    assert body["result"]["citations"]
    assert body["estimated_cost_usd"] == 0.39
    assert body["actual_cost_usd"] == 0.36
    assert body["cost_status"] == "actual"
    assert hidden.status_code == 404


def test_hypothesis_submit_and_cancel_use_same_owner_scoped_manager(tmp_path):
    manager = StubResearchJobs()
    app = create_app(_settings(tmp_path), research_jobs=manager)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        submitted = client.post(
            "/api/research/hypotheses",
            json={
                "topics": ["HBM", "Silicon Photonics"],
                "profiles": ["memory-innovation"],
            },
        )
        cancelled = client.delete(
            f"/api/research/{submitted.json()['job_id']}"
        )

    assert submitted.status_code == 202
    assert manager.records["job-1"]["kind"] == "hypotheses"
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


def test_disabled_feature_returns_503_without_allocating_job(tmp_path):
    manager = StubResearchJobs()
    app = create_app(
        _settings(tmp_path, exhaustive_enabled=False),
        research_jobs=manager,
    )

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post(
            "/api/research/chat", json={"query": "question"}
        )

    assert response.status_code == 503
    assert manager.submissions == 0


def test_disabled_feature_returns_503_to_remote_client_before_auth(tmp_path):
    manager = StubResearchJobs()
    app = create_app(
        _settings(
            tmp_path,
            exhaustive_enabled=False,
            research_token="",
        ),
        research_jobs=manager,
    )

    with TestClient(app, client=("100.64.1.9", 50000)) as client:
        response = client.post(
            "/api/research/chat",
            json={"query": "question"},
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "exhaustive research is disabled"
    }
    assert manager.submissions == 0


def test_chat_request_forbids_bounded_retrieval_parameters(tmp_path):
    manager = StubResearchJobs()
    app = create_app(_settings(tmp_path), research_jobs=manager)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post(
            "/api/research/chat",
            json={"query": "question", "top_k": 10},
        )

    assert response.status_code == 422


def test_bounded_chat_route_is_disabled_while_exhaustive_mode_is_on(
    tmp_path,
):
    manager = StubResearchJobs()
    app = create_app(_settings(tmp_path), research_jobs=manager)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post("/api/chat", json={"query": "question"})

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "bounded chat is disabled while exhaustive research is enabled"
    )


def test_research_submissions_consume_the_shared_rate_limit(tmp_path):
    """A research job is the most expensive call this gateway accepts."""
    manager = StubResearchJobs()
    app = create_app(
        _settings(tmp_path, rate_per_min=1), research_jobs=manager
    )

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        body = {"query": "How do the systems interact?", "history": []}
        first = client.post("/api/research/chat", json=body)
        second = client.post("/api/research/chat", json=body)

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "60"
    # The concurrency slot is released once the job is queued, not held for
    # the lifetime of the run.
    assert app.state.limiter._active["127.0.0.1"] == 0

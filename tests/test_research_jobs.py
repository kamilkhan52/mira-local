import asyncio
import json
import time
from pathlib import Path

import pytest

from chatbot.research_jobs import (
    ResearchJobConflict,
    ResearchJobManager,
    ResearchJobManagerStopped,
    ResearchQueueFull,
    _json_safe,
)
from mira.exhaustive.compiler import CitationRecord
from mira.exhaustive.engine import HypothesisResearchResult
from mira.exhaustive.types import FileFingerprint, SnapshotFingerprint


def controllable_worker(kind, request, emit):
    emit({"name": "worker_started", "kind": kind})
    if request["mode"] == "cost-then-wait":
        emit({
            "name": "cost_actual",
            "actual_cost_usd": 0.125,
            "cost_status": "actual",
            "by_stage": {"evidence_map": 0.125},
        })
        time.sleep(10)
    elif request["mode"] == "slow":
        time.sleep(0.25)
    emit({"name": "cost_estimated", "total_with_reserve_usd": 0.2})
    return {"answer": request["mode"]}


def test_json_safe_serializes_snapshot_paths_at_the_job_boundary():
    fingerprint = SnapshotFingerprint(files=(
        FileFingerprint(
            path=Path("/tmp/graph.json"),
            device=1,
            inode=2,
            size=3,
            mtime_ns=4,
        ),
    ))

    payload = _json_safe({"fingerprint": fingerprint})

    assert json.loads(json.dumps(payload))["fingerprint"]["files"][0][
        "path"
    ] == "/tmp/graph.json"


async def _wait_for(manager, job_id, owner, status, timeout=3):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        record = manager.get(job_id, owner)
        if record and record["status"] in status:
            return record
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"job {job_id} did not reach {status}: "
        f"{manager.get(job_id, owner)}"
    )


def test_second_owner_queues_and_first_owner_reattaches():
    async def scenario():
        manager = ResearchJobManager(
            controllable_worker, concurrency=1, queue_size=4
        )
        try:
            first = await manager.submit(
                "owner-a", "chat", {"mode": "slow"}
            )
            second = await manager.submit(
                "owner-b", "chat", {"mode": "fast"}
            )

            assert manager.get(second, "owner-b")["status"] == "queued"
            assert manager.active_for_owner("owner-a") == first
            assert manager.get(first, "owner-b") is None
            assert manager.get(first, "owner-a")["id"] == first

            await _wait_for(manager, first, "owner-a", {"done"})
            done = await _wait_for(manager, second, "owner-b", {"done"})
            assert done["result"] == {"answer": "fast"}
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_one_active_job_per_owner_and_bounded_queue():
    async def scenario():
        manager = ResearchJobManager(
            controllable_worker, concurrency=1, queue_size=1
        )
        try:
            first = await manager.submit(
                "owner-a", "chat", {"mode": "slow"}
            )
            with pytest.raises(ResearchJobConflict) as conflict:
                await manager.submit(
                    "owner-a", "chat", {"mode": "fast"}
                )
            assert conflict.value.active_job_id == first

            await manager.submit("owner-b", "chat", {"mode": "slow"})
            with pytest.raises(ResearchQueueFull):
                await manager.submit(
                    "owner-c", "chat", {"mode": "fast"}
                )
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_cancel_terminates_worker_preserves_cost_and_starts_next_job():
    async def scenario():
        manager = ResearchJobManager(
            controllable_worker, concurrency=1, queue_size=3
        )
        try:
            first = await manager.submit(
                "owner-a", "chat", {"mode": "cost-then-wait"}
            )
            second = await manager.submit(
                "owner-b", "chat", {"mode": "fast"}
            )
            deadline = asyncio.get_running_loop().time() + 2
            while manager.get(first, "owner-a")["actual_cost_usd"] == 0:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.01)

            started = time.monotonic()
            assert await manager.cancel(first, "owner-a") is True
            cancelled = manager.get(first, "owner-a")
            assert cancelled["status"] == "cancelled"
            assert cancelled["actual_cost_usd"] == pytest.approx(0.125)
            assert time.monotonic() - started < 2

            await _wait_for(manager, second, "owner-b", {"done"})
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_subscribe_replays_snapshot_and_streams_terminal_event():
    async def scenario():
        manager = ResearchJobManager(
            controllable_worker, concurrency=1, queue_size=2
        )
        try:
            job_id = await manager.submit(
                "owner-a", "chat", {"mode": "fast"}
            )
            events = []
            async for event in manager.subscribe(job_id, "owner-a"):
                events.append(event)
            assert events[0]["event"] == "snapshot"
            assert events[-1]["event"] == "done"
            assert any(
                event["event"] == "progress"
                and event["data"]["name"] == "worker_started"
                for event in events
            )
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


def test_shutdown_cancels_active_and_refuses_new_work():
    async def scenario():
        manager = ResearchJobManager(
            controllable_worker, concurrency=1, queue_size=2
        )
        job_id = await manager.submit(
            "owner-a", "chat", {"mode": "cost-then-wait"}
        )
        await _wait_for(manager, job_id, "owner-a", {"running"})

        await manager.shutdown()

        assert manager.get(job_id, "owner-a")["status"] == "cancelled"
        with pytest.raises(ResearchJobManagerStopped):
            await manager.submit("owner-b", "chat", {"mode": "fast"})

    asyncio.run(scenario())


def test_hypothesis_job_record_carries_citations_not_the_evidence_corpus():
    """The transported result must not include the compiled corpus.

    The full evidence is multiple megabytes of chunk text; carrying it would
    put a copy in the job record, in every SSE terminal event, and in every
    GET /api/research/{job_id} response."""
    citations = tuple(
        CitationRecord(
            chunk_id=f"chunk-{index}",
            file_path=f"paper-{index}.md",
            title=f"Paper {index}",
            domains=("memory",),
            source_chunk_id=f"chunk-{index}",
        )
        for index in range(3)
    )
    result = HypothesisResearchResult(
        result={"markdown": "# Dossier"},
        citations=citations,
        coverage=None,
        estimated_cost=None,
        cost=None,
    )

    payload = _json_safe(result)

    assert set(payload) == {
        "result", "citations", "coverage", "estimated_cost", "cost",
    }
    assert [item["title"] for item in payload["citations"]] == [
        "Paper 0", "Paper 1", "Paper 2"
    ]
    assert "records" not in json.dumps(payload)
    assert len(json.dumps(payload)) < 2_000

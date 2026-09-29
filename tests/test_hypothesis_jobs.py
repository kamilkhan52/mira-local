import asyncio
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from chatbot.app import create_app
from chatbot.config import Settings
from chatbot.jobs import JobConflict, JobManager, JobManagerStopped


@pytest.fixture()
def hypothesis_stub(tmp_path):
    """A controllable child process; no production pipeline or data is used."""
    dossiers = tmp_path / "dossiers"
    script = tmp_path / "hypothesis_stub.py"
    script.write_text(
        f"""
import argparse
import signal
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--graph", required=True)
ap.add_argument("--topic", required=True)
ap.add_argument("--profile", action="append", required=True)
ap.add_argument("--max-hypotheses", type=int, required=True)
ap.add_argument("--max-candidates", type=int, required=True)
ap.add_argument("--critic", action="store_true")
ap.add_argument("--no-external", action="store_true")
ap.add_argument("--exhaustive", action="store_true")
ap.add_argument("--out-dir")
args = ap.parse_args()
if args.graph != "combined":
    raise SystemExit("gateway did not force the combined graph")

root = Path(args.out_dir) if args.out_dir else Path({str(dossiers)!r})
# Mirror hypothesize.py: a multi-profile run writes under combined/.
profile_dir = root / (args.profile[0] if len(args.profile) == 1 else "combined")
profile_dir.mkdir(parents=True, exist_ok=True)
dossier = profile_dir / "stub-dossier.md"

if args.topic.startswith("success"):
    print("[1/5] loading", flush=True)
    dossier.write_text("# Stub dossier\\n")
    print(f"Dossier written: {{dossier}}", flush=True)
    raise SystemExit(0)

if args.topic == "fanout-error":
    time.sleep(0.1)
    for stage in range(1, 6):
        print(f"[{{stage}}/5] fanout {{stage}}", flush=True)
    print("fanout complete with controlled error", file=sys.stderr, flush=True)
    raise SystemExit(4)

if args.topic == "argv-check":
    expected = (
        args.profile == ["memory-innovation", "optical-io"]
        and args.max_hypotheses == 7
        and args.max_candidates == 19
        and args.critic
        and args.no_external
        and args.exhaustive
    )
    if not expected:
        print(f"unexpected argv: {{args}}", file=sys.stderr, flush=True)
        raise SystemExit(8)
    dossier.write_text("# Argv dossier\\n")
    print(f"Dossier written: {{dossier}}", flush=True)
    raise SystemExit(0)

if args.topic == "slow-success":
    print("[1/5] loading slowly", flush=True)
    time.sleep(0.3)
    dossier.write_text("# Slow stub dossier\\n")
    print(f"Dossier written: {{dossier}}", flush=True)
    raise SystemExit(0)

if args.topic == "timeout-ignore-term":
    term_marker = root / "term-observed"
    def ignore_term(_signum, _frame):
        term_marker.write_text("SIGTERM")
    signal.signal(signal.SIGTERM, ignore_term)
    print("[1/5] waiting forever", flush=True)
    started = time.monotonic()
    while time.monotonic() - started < 1.0:
        time.sleep(0.02)
    print("stub watchdog self-exited", file=sys.stderr, flush=True)
    raise SystemExit(91)

if args.topic == "missing-five":
    print("[1/5] loading", flush=True)
    dossier.write_text("# Fallback dossier\\n")
    raise SystemExit(0)

if args.topic == "warnings":
    print("[1/5] loading", flush=True)
    print("DEGRADED: venue corpus missing", flush=True)
    print("WARNING: entity vectors unavailable", file=sys.stderr, flush=True)
    dossier.write_text("# Warning dossier\\n")
    print(f"Dossier written: {{dossier}}", flush=True)
    raise SystemExit(0)

if args.topic == "dossier-degraded":
    dossier.write_text(
        "# Stub dossier\\n"
        "- **Degraded:** stale venue corpus\\n"
        "- **Degraded:** retrieval unavailable\\n"
    )
    print(f"Dossier written: {{dossier}}", flush=True)
    raise SystemExit(0)

if args.topic == "stages-error":
    for stage in range(1, 6):
        print(f"[{{stage}}/5] stage {{stage}}", flush=True)
    print("discarded-prefix-" + ("x" * 9000) + "-TAIL_MARKER",
          file=sys.stderr, flush=True)
    raise SystemExit(3)

if args.topic == "-dash-topic":
    # argparse in THIS child rejects a bare "-dash-topic" as an option token
    # unless the parent used the --opt=value form.
    dossier.write_text("# Dash dossier\\n")
    print(f"Dossier written: {{dossier}}", flush=True)
    raise SystemExit(0)

if args.topic == "check-lines":
    print("[4/5] Synthesizing 2 hypotheses ...", flush=True)
    print("      \\u2713 CXL pooling \\u00d7 silicon photonics", flush=True)
    print("      \\u2713 wear leveling \\u00d7 optical switching", flush=True)
    dossier.write_text("# Check dossier\\n")
    print(f"Dossier written: {{dossier}}", flush=True)
    raise SystemExit(0)

if args.topic == "escape-dossier":
    outside = root.parent / "outside.md"
    outside.write_text("# Outside the dossier root\\n")
    print(f"Dossier written: {{outside}}", flush=True)
    raise SystemExit(0)

if args.topic == "symlink-dossier":
    secret = root.parent / "secret.md"
    secret.write_text("# Secret outside the root\\n")
    profile_dir.mkdir(parents=True, exist_ok=True)
    link = profile_dir / "link.md"
    if not link.exists():
        link.symlink_to(secret)
    print(f"Dossier written: {{link}}", flush=True)
    raise SystemExit(0)

if args.topic == "txt-dossier":
    notes = profile_dir / "notes.txt"
    notes.write_text("not a dossier\\n")
    print(f"Dossier written: {{notes}}", flush=True)
    raise SystemExit(0)

if args.topic == "long-success":
    print("[1/5] loading", flush=True)
    time.sleep(1.5)
    dossier.write_text("# Long dossier\\n")
    print(f"Dossier written: {{dossier}}", flush=True)
    raise SystemExit(0)

if args.topic == "stderr-flood":
    # Far past a 64 KiB pipe buffer: an undrained stderr would block the child
    # forever and the run would never finish.
    blob = "F" * 1000
    for _ in range(1200):
        print(blob, file=sys.stderr, flush=False)
    sys.stderr.flush()
    print("[1/5] flooded", flush=True)
    dossier.write_text("# Flood dossier\\n")
    print(f"Dossier written: {{dossier}}", flush=True)
    raise SystemExit(0)

if args.topic == "hold-stderr-open":
    # A grandchild inherits stderr and keeps the pipe open after this process
    # exits, so an unbounded drain would never see EOF.
    import subprocess
    subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stderr=sys.stderr,
    )
    print("[1/5] spawned holder", flush=True)
    dossier.write_text("# Holder dossier\\n")
    print(f"Dossier written: {{dossier}}", flush=True)
    sys.stdout.flush()
    raise SystemExit(0)

raise SystemExit("unknown stub mode")
"""
    )
    return script, dossiers


def _params(topic):
    return {
        "topic": topic,
        "profiles": ["memory-innovation"],
        "max_hypotheses": 5,
        "max_candidates": 12,
        "critic": False,
        "no_external": False,
    }


def test_legacy_hypothesis_cli_obeys_exhaustive_feature_flag(tmp_path):
    disabled = create_app(Settings(
        lightrag_api_key="test-key",
        provenance_path=tmp_path / "missing.json",
        exhaustive_enabled=False,
    ))
    enabled = create_app(Settings(
        lightrag_api_key="test-key",
        provenance_path=tmp_path / "missing.json",
        exhaustive_enabled=True,
    ))

    assert "--exhaustive" not in disabled.state.jobs._argv(_params("topic"))
    assert "--exhaustive" in enabled.state.jobs._argv(_params("topic"))
    assert JobManager().exhaustive_enabled is False


def test_exhaustive_flag_reaches_the_spawned_cli_through_the_endpoint(
    hypothesis_gateway,
):
    """The flag is only worth carrying if a real POST can reach it.

    The route used to refuse every request while exhaustive mode was on, so
    the manager's --exhaustive branch could never run."""
    client, manager, _dossiers = hypothesis_gateway
    manager.exhaustive_enabled = True
    client.app.state.settings.exhaustive_enabled = True
    argv = []
    original = manager._argv

    def record(params):
        built = original(params)
        argv.append(built)
        return built

    manager._argv = record

    submitted = client.post(
        "/api/hypothesis",
        headers=_hypothesis_headers(),
        json={"topics": ["success"], "profiles": ["memory-innovation"]},
    )

    assert submitted.status_code == 202
    assert argv and "--exhaustive" in argv[0]


async def _terminal_events(manager, job_id):
    async def collect():
        events = []
        async for event in manager.subscribe(job_id):
            events.append(event)
            if event["event"] in {"done", "error"}:
                break
        return events

    return await asyncio.wait_for(collect(), timeout=2)


async def _wait_for_terminal_record(manager, job_id):
    async def poll():
        while True:
            record = manager.get(job_id)
            if record["status"] != "running":
                return record
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(poll(), timeout=2)


def test_stage_markers_stream_as_progress_events_in_order(hypothesis_stub, tmp_path):
    """Removing best-effort marker parsing would leave the UI on a spinner."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_id = await manager.submit(_params("stages-error"))
        events = await _terminal_events(manager, job_id)
        await manager.shutdown()
        return events

    events = asyncio.run(scenario())

    assert [
        event["data"]["stage"]
        for event in events
        if event["event"] == "progress"
    ] == [1, 2, 3, 4, 5]


def test_exit_zero_emits_done_with_existing_dossier_path(
    hypothesis_stub, tmp_path
):
    """Treating exit zero alone as success would expose a nonexistent dossier."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_id = await manager.submit(_params("success"))
        events = await _terminal_events(manager, job_id)
        record = manager.get(job_id)
        await manager.shutdown()
        return events, record

    events, record = asyncio.run(scenario())
    done = [event for event in events if event["event"] == "done"]

    expected = dossiers / "memory-innovation" / "stub-dossier.md"
    assert done == [{"event": "done", "data": {"dossier_path": str(expected)}}]
    assert record["status"] == "done"
    assert record["dossier_path"] == str(expected)


def test_nonzero_exit_emits_bounded_stderr_tail(hypothesis_stub, tmp_path):
    """Keeping all stderr would make long CLI failures grow job memory without bound."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_id = await manager.submit(_params("stages-error"))
        events = await _terminal_events(manager, job_id)
        record = manager.get(job_id)
        await manager.shutdown()
        return events, record

    events, record = asyncio.run(scenario())
    error = [event for event in events if event["event"] == "error"][-1]
    message = error["data"]["message"]

    assert message.endswith("-TAIL_MARKER")
    assert "discarded-prefix" not in message
    assert len(message.encode("utf-8")) <= 8 * 1024
    assert record["error"] == message


def test_second_submit_is_rejected_with_active_job_id(hypothesis_stub, tmp_path):
    """Replacing the global slot with a check-then-set race would allow two costly runs."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        first_id = await manager.submit(_params("slow-success"))
        with pytest.raises(JobConflict) as caught:
            await manager.submit(_params("success"))
        events = await _terminal_events(manager, first_id)
        await manager.shutdown()
        return first_id, caught.value.active_job_id, events

    first_id, active_id, events = asyncio.run(scenario())

    assert active_id == first_id
    assert events[-1]["event"] == "done"


def test_slot_released_after_success(hypothesis_stub, tmp_path):
    """Leaking the global slot on success would make every later POST return 409."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        first_id = await manager.submit(_params("success"))
        await _terminal_events(manager, first_id)
        second_id = await manager.submit(_params("stages-error"))
        await _terminal_events(manager, second_id)
        await manager.shutdown()
        return first_id, second_id

    first_id, second_id = asyncio.run(scenario())
    assert second_id != first_id


def test_slot_released_after_error(hypothesis_stub, tmp_path):
    """Leaking the global slot on nonzero exit would require a gateway restart."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        first_id = await manager.submit(_params("stages-error"))
        await _terminal_events(manager, first_id)
        second_id = await manager.submit(_params("success"))
        await _terminal_events(manager, second_id)
        await manager.shutdown()
        return first_id, second_id

    first_id, second_id = asyncio.run(scenario())
    assert second_id != first_id


def test_slot_released_after_disconnect_while_child_finishes(
    hypothesis_stub, tmp_path
):
    """Closing an SSE listener must neither cancel the durable run nor leak its slot."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        first_id = await manager.submit(_params("slow-success"))
        listener = manager.subscribe(first_id)
        assert (await anext(listener))["event"] == "snapshot"
        assert (await anext(listener))["event"] == "progress"
        await listener.aclose()
        detached_record = manager.get(first_id)
        finished_record = await _wait_for_terminal_record(manager, first_id)
        second_id = await manager.submit(_params("stages-error"))
        await _terminal_events(manager, second_id)
        await manager.shutdown()
        return detached_record, finished_record, first_id, second_id

    detached, finished, first_id, second_id = asyncio.run(scenario())

    assert detached["status"] == "running"
    assert finished["status"] == "done"
    assert Path(finished["dossier_path"]).read_text() == "# Slow stub dossier\n"
    assert second_id != first_id


def test_slot_released_after_timeout_with_term_then_kill(
    hypothesis_stub, tmp_path
):
    """A timeout must escalate past an ignored SIGTERM and still free the run slot."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=0.1,
            kill_grace_sec=0.1,
        )
        started = asyncio.get_running_loop().time()
        first_id = await manager.submit(_params("timeout-ignore-term"))
        events = await _terminal_events(manager, first_id)
        elapsed = asyncio.get_running_loop().time() - started
        second_id = await manager.submit(_params("success"))
        await _terminal_events(manager, second_id)
        await manager.shutdown()
        return events, elapsed, first_id, second_id

    events, elapsed, first_id, second_id = asyncio.run(scenario())
    message = [event for event in events if event["event"] == "error"][-1][
        "data"
    ]["message"]

    assert message == "hypothesis generation timed out"
    assert (dossiers / "term-observed").read_text() == "SIGTERM"
    assert elapsed < 0.8
    assert second_id != first_id


def test_slot_released_after_shutdown_and_live_child_is_reaped(
    hypothesis_stub, tmp_path
):
    """Gateway shutdown must not leave either an orphan child or a held slot."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
            kill_grace_sec=0.1,
        )
        first_id = await manager.submit(_params("timeout-ignore-term"))
        listener = manager.subscribe(first_id)
        await anext(listener)
        await anext(listener)
        await listener.aclose()
        first_process = manager._jobs[first_id].process
        started = asyncio.get_running_loop().time()
        await manager.shutdown()
        elapsed = asyncio.get_running_loop().time() - started
        shutdown_record = manager.get(first_id)
        # The slot must be free, but shutdown is terminal: it is checked
        # directly rather than by starting another job, because a manager that
        # accepted new work after shutdown would leave a child running past
        # the gateway's own lifetime.
        released = manager._active_job_id is None
        refused = None
        try:
            await manager.submit(_params("success"))
        except Exception as exc:
            refused = exc
        return first_process, shutdown_record, elapsed, released, refused

    process, record, elapsed, released, refused = asyncio.run(scenario())

    assert process.returncode is not None
    assert record["status"] == "error"
    assert record["error"] == "gateway shutdown"
    assert elapsed < 0.8
    assert released, "shutdown must release the run slot"
    assert isinstance(refused, JobManagerStopped)


def test_slot_released_after_stdout_reader_exception(
    hypothesis_stub, tmp_path
):
    """A crashed stdout parser must reap the child, emit error, and free the slot."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
            kill_grace_sec=0.1,
        )

        async def broken_reader(_job):
            raise RuntimeError("controlled reader crash")

        manager._read_stdout = broken_reader
        started = asyncio.get_running_loop().time()
        first_id = await manager.submit(_params("timeout-ignore-term"))
        first_process = manager._jobs[first_id].process
        events = await _terminal_events(manager, first_id)
        elapsed = asyncio.get_running_loop().time() - started
        manager._read_stdout = JobManager._read_stdout.__get__(manager)
        second_id = await manager.submit(_params("success"))
        await _terminal_events(manager, second_id)
        await manager.shutdown()
        return first_process, events, elapsed, first_id, second_id

    process, events, elapsed, first_id, second_id = asyncio.run(scenario())
    message = [event for event in events if event["event"] == "error"][-1][
        "data"
    ]["message"]

    assert message == "stdout reader failed: controlled reader crash"
    assert process.returncode is not None
    assert elapsed < 0.8
    assert second_id != first_id


def test_late_subscriber_gets_snapshot_then_live_done(hypothesis_stub, tmp_path):
    """Omitting the snapshot would make a reopened tab lose all prior progress."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_id = await manager.submit(_params("slow-success"))
        while manager.get(job_id)["stage"] is None:
            await asyncio.sleep(0.01)
        events = await _terminal_events(manager, job_id)
        await manager.shutdown()
        return events

    events = asyncio.run(scenario())

    assert events[0]["event"] == "snapshot"
    assert events[0]["data"]["stage"] == 1
    assert events[0]["data"]["status"] == "running"
    assert events[-1]["event"] == "done"


def test_snapshot_subscriber_drains_terminal_queued_while_snapshot_is_paused(
    hypothesis_stub, tmp_path
):
    """A job finishing after snapshot yield must not make the generator return early."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_id = await manager.submit(_params("slow-success"))
        listener = manager.subscribe(job_id)
        snapshot = await anext(listener)
        finished = await _wait_for_terminal_record(manager, job_id)

        # The job ran to completion while the consumer was parked on the
        # snapshot, so everything emitted after registration -- progress AND
        # the terminal event -- is already sitting in this listener's queue.
        # The generator must hand all of it over instead of returning early
        # because job.status is now terminal.
        async def drain():
            tail = []
            async for event in listener:
                tail.append(event)
                if event["event"] in {"done", "error"}:
                    break
            return tail

        tail = await asyncio.wait_for(drain(), timeout=1)
        await listener.aclose()
        await manager.shutdown()
        return snapshot, finished, tail

    snapshot, finished, tail = asyncio.run(scenario())

    assert snapshot["event"] == "snapshot"
    assert snapshot["data"]["status"] == "running"
    assert finished["status"] == "done"
    # The terminal event survived the snapshot window rather than being
    # stranded in the queue behind an early return.
    assert tail[-1]["event"] == "done"
    # Progress queued alongside it survived too, in order. Stages before
    # registration are legitimately absent, so this checks ordering rather
    # than a fixed set.
    stages = [e["data"]["stage"] for e in tail if e["event"] == "progress"]
    assert stages == sorted(stages)


def test_terminal_is_published_only_after_slot_release_and_immediate_submit_works(
    hypothesis_stub, tmp_path
):
    """A client reacting to done must never race the prior job's slot cleanup."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        published_with_active_slot = []
        real_emit = manager._emit

        def observing_emit(job, event, data):
            if event in {"done", "error"}:
                published_with_active_slot.append(manager._active_job_id)
            real_emit(job, event, data)

        manager._emit = observing_emit
        first_id = await manager.submit(_params("success"))
        events = await _terminal_events(manager, first_id)
        assert events[-1]["event"] == "done"
        second_id = await manager.submit(_params("stages-error"))
        await _terminal_events(manager, second_id)
        await manager.shutdown()
        return published_with_active_slot, first_id, second_id

    active_ids, first_id, second_id = asyncio.run(scenario())

    assert active_ids == [None, None]
    assert second_id != first_id


def test_missing_five_marker_uses_newest_profile_dossier_fallback(
    hypothesis_stub, tmp_path
):
    """Requiring [5/5] or its path line would reject a valid exit-zero run."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_id = await manager.submit(_params("missing-five"))
        events = await _terminal_events(manager, job_id)
        record = manager.get(job_id)
        await manager.shutdown()
        return events, record

    events, record = asyncio.run(scenario())

    assert events[-1] == {
        "event": "done",
        "data": {
            "dossier_path": str(
                dossiers / "memory-innovation" / "stub-dossier.md"
            )
        },
    }
    assert record["stage"] == 1


def test_degraded_output_streams_warnings_before_done(hypothesis_stub, tmp_path):
    """Dropping degraded stdout/stderr would hide reduced-quality evidence."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_id = await manager.submit(_params("warnings"))
        events = await _terminal_events(manager, job_id)
        record = manager.get(job_id)
        await manager.shutdown()
        return events, record

    events, record = asyncio.run(scenario())
    warning_events = [
        event["data"]["warnings"]
        for event in events
        if event["event"] == "warnings"
    ]

    assert len(warning_events) == 2
    assert len(warning_events[0]) == 1
    assert set(warning_events[-1]) == {
        "venue corpus missing",
        "entity vectors unavailable",
    }
    assert set(record["warnings"]) == {
        "venue corpus missing",
        "entity vectors unavailable",
    }
    assert events[-1]["event"] == "done"


def test_dossier_degraded_entries_emit_warnings_before_done(
    hypothesis_stub, tmp_path
):
    """Degraded notes not printed by the CLI must still reach listeners before done."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_id = await manager.submit(_params("dossier-degraded"))
        events = await _terminal_events(manager, job_id)
        await manager.shutdown()
        return events

    events = asyncio.run(scenario())
    event_names = [event["event"] for event in events]

    assert event_names[-2:] == ["warnings", "done"]
    assert events[-2]["data"]["warnings"] == [
        "stale venue corpus",
        "retrieval unavailable",
    ]


def test_profile_discovery_regex_scans_report_ids_and_caches_by_mtime(
    tmp_path, monkeypatch
):
    """A full graph parse per request would make the profiles endpoint too costly."""
    graphml = tmp_path / "combined.graphml"
    graphml.write_text(
        """
<graphml><graph>
  <node id="memory-innovation-2026-07-27">
    <data key="kind">Report</data>
  </node>
  <node id="storage-fabric-2026-07-26">
    <data key="kind">Report</data>
  </node>
  <node id="not-a-report"><data key="kind">Topic</data></node>
  <node id="memory-innovation-2026-07-25">
    <data key="kind">Report</data>
  </node>
</graph></graphml>
"""
    )
    manager = JobManager(
        root=tmp_path,
        script_path=tmp_path / "unused.py",
        hypotheses_dir=tmp_path / "dossiers",
        graphml_path=graphml,
    )
    original_read_text = Path.read_text
    reads = 0

    def counting_read_text(path, *args, **kwargs):
        nonlocal reads
        if path == graphml:
            reads += 1
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    assert manager.discover_profiles() == [
        "memory-innovation",
        "storage-fabric",
    ]
    assert manager.discover_profiles() == [
        "memory-innovation",
        "storage-fabric",
    ]
    assert reads == 1

    graphml.write_text(
        '<graphml><graph><node id="optical-io-2026-07-27">'
        '<data key="kind">Report</data></node></graph></graphml>'
    )
    changed_ns = graphml.stat().st_mtime_ns + 1_000_000_000
    os.utime(graphml, ns=(changed_ns, changed_ns))

    assert manager.discover_profiles() == ["optical-io"]
    assert reads == 2


@pytest.mark.parametrize(
    "body",
    [
        {"topics": [""], "profiles": ["memory"]},
        {"topics": ["x" * 201], "profiles": ["memory"]},
        {"topics": ["seed"], "profiles": []},
        {"topics": ["seed"], "profiles": ["memory"], "max_hypotheses": 0},
        {"topics": ["seed"], "profiles": ["memory"], "max_hypotheses": 11},
        {"topics": ["seed"], "profiles": ["memory"], "max_candidates": 0},
        {"topics": ["seed"], "profiles": ["memory"], "max_candidates": 31},
        {"topics": ["seed"], "profiles": ["memory", "memory"]},
        {"topics": ["seed"], "profiles": ["memory"], "unexpected": True},
    ],
)
def test_hypothesis_request_forbids_extra_and_enforces_bounds(body):
    """Weak request validation could launch unbounded or ambiguous paid runs."""
    from chatbot.app import HypothesisRequest

    with pytest.raises(ValidationError):
        HypothesisRequest.model_validate(body)


@pytest.fixture()
def hypothesis_gateway(hypothesis_stub, tmp_path):
    script, dossiers = hypothesis_stub
    graphml = tmp_path / "combined.graphml"
    graphml.write_text(
        """
<graphml><graph>
  <node id="memory-innovation-2026-07-27">
    <data key="kind">Report</data>
  </node>
  <node id="optical-io-2026-07-27">
    <data key="kind">Report</data>
  </node>
  <node id="CXL Memory Pooling"><data key="kind">Topic</data></node>
  <node id="Optical Interconnects"><data key="kind">Topic</data></node>
  <node id="AI &amp; Storage"><data key="kind">Topic</data></node>
  <node id="Orphan Topic"><data key="kind">Topic</data></node>
  <!-- stub control modes double as topic names in route tests -->
  <node id="success"><data key="kind">Topic</data></node>
  <node id="slow-success"><data key="kind">Topic</data></node>
  <node id="long-success"><data key="kind">Topic</data></node>
  <node id="stages-error"><data key="kind">Topic</data></node>
  <node id="fanout-error"><data key="kind">Topic</data></node>
  <node id="argv-check"><data key="kind">Topic</data></node>
  <node id="timeout-ignore-term"><data key="kind">Topic</data></node>
  <node id="missing-five"><data key="kind">Topic</data></node>
  <node id="warnings"><data key="kind">Topic</data></node>
  <node id="dossier-degraded"><data key="kind">Topic</data></node>
  <node id="check-lines"><data key="kind">Topic</data></node>
  <node id="escape-dossier"><data key="kind">Topic</data></node>
  <node id="symlink-dossier"><data key="kind">Topic</data></node>
  <node id="txt-dossier"><data key="kind">Topic</data></node>
  <node id="stderr-flood"><data key="kind">Topic</data></node>
  <node id="hold-stderr-open"><data key="kind">Topic</data></node>
  <node id="-dash-topic"><data key="kind">Topic</data></node>
  <node id="P1"><data key="kind">Paper</data></node>
  <node id="P2"><data key="kind">Paper</data></node>
  <node id="P3"><data key="kind">Paper</data></node>
  <edge source="P1" target="CXL Memory Pooling">
    <data key="keywords">primary_topic topic</data></edge>
  <edge source="P2" target="CXL Memory Pooling">
    <data key="keywords">also_covers topic</data></edge>
  <edge source="P3" target="CXL Memory Pooling">
    <data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="Optical Interconnects">
    <data key="keywords">primary_topic topic</data></edge>
  <edge source="P2" target="AI &amp; Storage">
    <data key="keywords">primary_topic topic</data></edge>
  <edge source="P3" target="Orphan Topic">
    <data key="keywords">authored_by author</data></edge>
  <edge source="P1" target="success"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="slow-success"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="long-success"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="stages-error"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="fanout-error"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="argv-check"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="timeout-ignore-term"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="missing-five"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="warnings"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="dossier-degraded"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="check-lines"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="escape-dossier"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="symlink-dossier"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="txt-dossier"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="stderr-flood"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="hold-stderr-open"><data key="keywords">primary_topic topic</data></edge>
  <edge source="P1" target="-dash-topic"><data key="keywords">primary_topic topic</data></edge>
</graph></graphml>
"""
    )
    settings = Settings(
        lightrag_api_key="test-key",
        hypothesis_token="hypothesis-secret",
        hypotheses_dir=dossiers,
        hypothesis_timeout_sec=5,
        provenance_path=tmp_path / "missing-provenance.json",
    )
    app = create_app(settings)
    manager = JobManager(
        root=tmp_path,
        script_path=script,
        hypotheses_dir=dossiers,
        graphml_path=graphml,
        timeout_sec=5,
        kill_grace_sec=0.1,
    )
    app.state.jobs = manager
    # Real loopback peer: the auth rule keys off the socket address, so the
    # default "testclient" host would not exercise the intended path.
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        yield client, manager, dossiers


def _hypothesis_headers():
    return {"X-Hypothesis-Token": "hypothesis-secret"}


def test_all_hypothesis_job_routes_require_hypothesis_token(hypothesis_gateway):
    """Every job route must stay gated against REMOTE peers. Loopback callers are
    exempt by design (they can already run hypothesize.py directly), so the gate
    is asserted from a non-loopback peer -- otherwise adding one ungated route
    would expose paid runs or their results to the tailnet."""
    client, _, _ = hypothesis_gateway
    body = {"topics": ["success"], "profiles": ["memory-innovation"]}

    with _remote_client(client) as remote:
        assert remote.post("/api/hypothesis", json=body).status_code == 403
        # Auth is decided before body validation, so a malformed body still 403s.
        assert remote.post("/api/hypothesis", json={}).status_code == 403
        assert remote.get("/api/hypothesis/profiles").status_code == 403
        assert remote.get("/api/hypothesis/topics").status_code == 403
        assert remote.get("/api/hypothesis/dossiers").status_code == 403
        assert remote.get("/api/hypothesis/missing").status_code == 403
        assert remote.get("/api/hypothesis/missing/events").status_code == 403

    # And the same routes are reachable without a token from loopback.
    assert client.get("/api/hypothesis/profiles").status_code == 200
    assert client.get("/api/hypothesis/topics").status_code == 200


def test_profiles_route_and_submit_reject_profiles_absent_from_server_scan(
    hypothesis_gateway,
):
    """Trusting client profile strings would bypass graph-backed profile validation."""
    client, _, _ = hypothesis_gateway
    headers = _hypothesis_headers()

    profiles = client.get("/api/hypothesis/profiles", headers=headers)
    invalid = client.post(
        "/api/hypothesis",
        headers=headers,
        json={"topics": ["success"], "profiles": ["not-in-graph"]},
    )

    assert profiles.status_code == 200
    assert profiles.json() == {
        "profiles": ["memory-innovation", "optical-io"]
    }
    assert invalid.status_code == 422
    assert invalid.json()["invalid_profiles"] == ["not-in-graph"]


def test_post_events_and_job_get_complete_with_dossier_markdown(
    hypothesis_gateway,
):
    """Breaking the POST→SSE→record contract would strand completed dossiers."""
    client, _, dossiers = hypothesis_gateway
    headers = _hypothesis_headers()

    submitted = client.post(
        "/api/hypothesis",
        headers=headers,
        json={"topics": ["success"], "profiles": ["memory-innovation"]},
    )
    assert submitted.status_code == 202
    assert set(submitted.json()) == {"job_id"}
    job_id = submitted.json()["job_id"]

    streamed = client.get(
        f"/api/hypothesis/{job_id}/events", headers=headers
    )
    record = client.get(f"/api/hypothesis/{job_id}", headers=headers)

    assert streamed.status_code == 200
    assert "event: snapshot" in streamed.text
    assert "event: progress" in streamed.text
    assert "event: done" in streamed.text
    assert str(dossiers / "memory-innovation" / "stub-dossier.md") in streamed.text
    assert record.status_code == 200
    assert record.json()["status"] == "done"
    assert record.json()["dossier_markdown"] == "# Stub dossier\n"


def test_second_post_returns_409_with_active_job_id(hypothesis_gateway):
    """A generic 409 without the active id prevents clients from reattaching."""
    client, _, _ = hypothesis_gateway
    headers = _hypothesis_headers()

    first = client.post(
        "/api/hypothesis",
        headers=headers,
        json={"topics": ["slow-success"], "profiles": ["memory-innovation"]},
    )
    second = client.post(
        "/api/hypothesis",
        headers=headers,
        json={"topics": ["success"], "profiles": ["memory-innovation"]},
    )
    client.get(
        f"/api/hypothesis/{first.json()['job_id']}/events", headers=headers
    )

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["active_job_id"] == first.json()["job_id"]


def test_unknown_job_and_event_stream_return_404(hypothesis_gateway):
    """Unknown ids must fail before an SSE response is committed."""
    client, _, _ = hypothesis_gateway
    headers = _hypothesis_headers()

    assert client.get("/api/hypothesis/nope", headers=headers).status_code == 404
    assert (
        client.get("/api/hypothesis/nope/events", headers=headers).status_code
        == 404
    )


def test_multiple_listeners_receive_the_same_live_progress_fanout(
    hypothesis_stub, tmp_path
):
    """Sharing one queue would split events between listeners instead of fanning out."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_id = await manager.submit(_params("fanout-error"))
        first, second = await asyncio.gather(
            _terminal_events(manager, job_id),
            _terminal_events(manager, job_id),
        )
        await manager.shutdown()
        return first, second

    first, second = asyncio.run(scenario())

    for events in (first, second):
        assert [
            event["data"]["stage"]
            for event in events
            if event["event"] == "progress"
        ] == [1, 2, 3, 4, 5]
        assert events[-1]["event"] == "error"


def test_finished_job_ring_retains_only_the_last_twenty(
    hypothesis_stub, tmp_path
):
    """Unbounded job records would leak memory in the long-running gateway."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
        )
        job_ids = []
        for _ in range(21):
            job_id = await manager.submit(_params("success"))
            await _terminal_events(manager, job_id)
            job_ids.append(job_id)
        records = [manager.get(job_id) for job_id in job_ids]
        await manager.shutdown()
        return records

    records = asyncio.run(scenario())

    assert records[0] is None
    assert all(record is not None for record in records[1:])
    assert records[-1]["status"] == "done"


def test_subprocess_uses_fixed_argv_and_never_interprets_topic_as_shell(
    hypothesis_stub, tmp_path
):
    """A shell command would execute metacharacters embedded in a research topic."""
    script, dossiers = hypothesis_stub
    shell_marker = tmp_path / "SHELL_WAS_USED"
    dangerous_topic = f"success; touch {shell_marker}"

    async def scenario():
        manager = JobManager(
            root=tmp_path,
            script_path=script,
            hypotheses_dir=dossiers,
            graphml_path=tmp_path / "combined.graphml",
            timeout_sec=5,
            # The stub's argv check requires --exhaustive, which the manager
            # only appends when the feature is switched on.
            exhaustive_enabled=True,
        )
        params = _params(dangerous_topic)
        params.update(
            {
                "profiles": ["memory-innovation", "optical-io"],
                "max_hypotheses": 7,
                "max_candidates": 19,
                "critic": True,
                "no_external": True,
            }
        )
        injection_id = await manager.submit(params)
        injection_events = await _terminal_events(manager, injection_id)
        params["topic"] = "argv-check"
        argv_id = await manager.submit(params)
        argv_events = await _terminal_events(manager, argv_id)
        await manager.shutdown()
        return injection_events, argv_events

    injection_events, argv_events = asyncio.run(scenario())

    assert injection_events[-1]["event"] == "done"
    assert argv_events[-1]["event"] == "done"
    assert not shell_marker.exists()


def test_gateway_lifespan_shutdown_reaps_live_hypothesis_child(
    hypothesis_stub, tmp_path
):
    """Wiring only manager.shutdown without the FastAPI lifespan leaves an orphan."""
    script, dossiers = hypothesis_stub
    graphml = tmp_path / "combined.graphml"
    graphml.write_text(
        '<graphml><graph><node id="memory-innovation-2026-07-27">'
        '<data key="kind">Report</data></node>'
        '<node id="timeout-ignore-term"><data key="kind">Topic</data></node><node id="PX"><data key="kind">Paper</data></node><edge source="PX" target="timeout-ignore-term"><data key="keywords">primary_topic topic</data></edge>'
        '</graph></graphml>'
    )
    settings = Settings(
        lightrag_api_key="test-key",
        hypothesis_token="hypothesis-secret",
        hypotheses_dir=dossiers,
        hypothesis_timeout_sec=5,
        provenance_path=tmp_path / "missing-provenance.json",
    )
    app = create_app(settings)
    manager = JobManager(
        root=tmp_path,
        script_path=script,
        hypotheses_dir=dossiers,
        graphml_path=graphml,
        timeout_sec=5,
        kill_grace_sec=0.1,
    )
    app.state.jobs = manager

    with TestClient(app) as client:
        submitted = client.post(
            "/api/hypothesis",
            headers=_hypothesis_headers(),
            json={
                "topics": ["timeout-ignore-term"],
                "profiles": ["memory-innovation"],
            },
        )
        job_id = submitted.json()["job_id"]
        deadline = time.monotonic() + 1
        while manager.get(job_id)["stage"] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        process = manager._jobs[job_id].process

    assert submitted.status_code == 202
    assert process.returncode is not None
    assert manager.get(job_id)["status"] == "error"
    assert manager.get(job_id)["error"] == "gateway shutdown"


def _manager(tmp_path, script, dossiers, **kwargs):
    kwargs.setdefault("timeout_sec", 5)
    return JobManager(
        root=tmp_path,
        script_path=script,
        hypotheses_dir=dossiers,
        graphml_path=tmp_path / "combined.graphml",
        **kwargs,
    )


def test_submit_is_refused_once_shutdown_has_started(hypothesis_stub, tmp_path):
    """Shutdown drops the lock before terminating; a submit racing into that
    window would spawn a child that outlives the gateway."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = _manager(tmp_path, script, dossiers)
        # No active job: a plain JobConflict cannot mask the real invariant,
        # which is that shutdown permanently closes the manager to new work.
        await manager.shutdown()
        refused = None
        try:
            await manager.submit(_params("success"))
        except Exception as exc:
            refused = exc
        return refused, manager

    refused, manager = asyncio.run(scenario())

    assert refused is not None, "submit after shutdown must be refused"
    assert not isinstance(refused, JobConflict), (
        "refusal must be a shutdown refusal, not an ordinary active-job conflict"
    )
    assert manager._active_job_id is None


def test_reported_dossier_outside_root_is_rejected(hypothesis_stub, tmp_path):
    """A `Dossier written:` line is child-controlled input, not a trusted path."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = _manager(tmp_path, script, dossiers)
        job_id = await manager.submit(_params("escape-dossier"))
        events = await _terminal_events(manager, job_id)
        await manager.shutdown()
        return manager.get(job_id), events

    record, events = asyncio.run(scenario())

    assert record["status"] == "error"
    assert events[-1]["event"] == "error"
    assert record["dossier_path"] is None


def test_reported_dossier_symlink_escape_is_rejected(hypothesis_stub, tmp_path):
    """A symlink inside the root must not launder a target outside it."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = _manager(tmp_path, script, dossiers)
        job_id = await manager.submit(_params("symlink-dossier"))
        await _terminal_events(manager, job_id)
        await manager.shutdown()
        return manager.get(job_id)

    record = asyncio.run(scenario())

    assert record["status"] == "error"
    assert record["dossier_path"] is None


def test_reported_non_markdown_dossier_is_rejected(hypothesis_stub, tmp_path):
    """Only .md dossiers are servable; anything else is a malformed run."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = _manager(tmp_path, script, dossiers)
        job_id = await manager.submit(_params("txt-dossier"))
        await _terminal_events(manager, job_id)
        await manager.shutdown()
        return manager.get(job_id)

    record = asyncio.run(scenario())

    assert record["status"] == "error"
    assert record["dossier_path"] is None


def test_out_dir_is_passed_to_child_so_fallback_scans_where_it_wrote(
    hypothesis_stub, tmp_path
):
    """Without --out-dir the child writes to its own default while the manager
    scans hypotheses_dir, so the fallback looks in the wrong place."""
    script, _unused = hypothesis_stub
    configured = tmp_path / "configured-dossiers"

    async def scenario():
        manager = _manager(tmp_path, script, configured)
        job_id = await manager.submit(_params("missing-five"))
        await _terminal_events(manager, job_id)
        await manager.shutdown()
        return manager.get(job_id)

    record = asyncio.run(scenario())

    assert record["status"] == "done"
    assert record["dossier_path"] is not None
    assert str(configured) in record["dossier_path"]


def test_drain_after_child_exit_is_bounded_by_timeout(hypothesis_stub, tmp_path):
    """A grandchild holding stderr open must not hang the job past its deadline."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = _manager(tmp_path, script, dossiers, timeout_sec=1.0)
        job_id = await manager.submit(_params("hold-stderr-open"))
        started = time.monotonic()
        record = await asyncio.wait_for(
            _wait_for_terminal_record(manager, job_id), timeout=8
        )
        elapsed = time.monotonic() - started
        await manager.shutdown()
        return elapsed, record

    elapsed, record = asyncio.run(scenario())

    assert elapsed < 6, f"drain was not bounded by the deadline (took {elapsed:.1f}s)"
    # The child exited 0 and wrote a real dossier. An undrainable pipe is a
    # reporting problem, not a failed run -- downgrading it to an error would
    # discard minutes of completed LLM work.
    assert record["status"] == "done"
    assert any("drain" in warning for warning in record["warnings"])


def test_per_hypothesis_check_lines_stream_as_progress(hypothesis_stub, tmp_path):
    """The UI shows per-hypothesis ticks during the long synthesis stage."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = _manager(tmp_path, script, dossiers)
        job_id = await manager.submit(_params("check-lines"))
        events = await _terminal_events(manager, job_id)
        await manager.shutdown()
        return events

    events = asyncio.run(scenario())

    hypotheses = [
        event["data"].get("hypothesis")
        for event in events
        if event["event"] == "progress" and event["data"].get("hypothesis")
    ]
    assert hypotheses == [
        "CXL pooling × silicon photonics",
        "wear leveling × optical switching",
    ]


def test_leading_dash_topic_reaches_child_intact(hypothesis_stub, tmp_path):
    """A topic starting with '-' must not be parsed as an option by the child."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = _manager(tmp_path, script, dossiers)
        job_id = await manager.submit(_params("-dash-topic"))
        await _terminal_events(manager, job_id)
        await manager.shutdown()
        return manager.get(job_id)

    record = asyncio.run(scenario())

    assert record["status"] == "done", record["error"]


def test_topic_with_nul_is_rejected_before_spawn(hypothesis_stub, tmp_path):
    """An embedded NUL cannot cross the argv boundary; fail cleanly, not with a
    raw ValueError from the spawn, and never leak the run slot."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = _manager(tmp_path, script, dossiers)
        rejected = None
        try:
            await manager.submit(_params("bad\x00topic"))
        except Exception as exc:
            rejected = exc
        follow_up = await manager.submit(_params("success"))
        await _terminal_events(manager, follow_up)
        await manager.shutdown()
        return rejected

    rejected = asyncio.run(scenario())

    assert isinstance(rejected, ValueError)


def test_large_stderr_does_not_deadlock_the_child(hypothesis_stub, tmp_path):
    """stderr is drained concurrently; if it were read only after exit, a child
    writing past the pipe buffer would block forever and never exit."""
    script, dossiers = hypothesis_stub

    async def scenario():
        manager = _manager(tmp_path, script, dossiers, timeout_sec=10)
        job_id = await manager.submit(_params("stderr-flood"))
        await asyncio.wait_for(_wait_for_terminal_record(manager, job_id), timeout=10)
        await manager.shutdown()
        return manager.get(job_id)

    record = asyncio.run(scenario())

    assert record["status"] == "done", record["error"]


def test_sse_client_disconnect_does_not_kill_the_run(hypothesis_stub, tmp_path):
    """Closing the browser tab must detach the listener only. The run costs
    minutes of LLM spend and writes a durable file, so it continues.

    Driven through ASGITransport with a real cancelled request task: the
    synchronous TestClient runs each ASGI call to completion and so cannot
    express a mid-stream disconnect at all.
    """
    import contextlib

    import httpx

    script, dossiers = hypothesis_stub
    graphml = tmp_path / "disconnect.graphml"
    graphml.write_text(
        """
<graphml><graph>
  <node id="memory-innovation-2026-07-27"><data key="kind">Report</data></node>
  <node id="long-success"><data key="kind">Topic</data></node><node id="PX"><data key="kind">Paper</data></node><edge source="PX" target="long-success"><data key="keywords">primary_topic topic</data></edge>
</graph></graphml>
"""
    )
    settings = Settings(
        lightrag_api_key="test-key",
        hypothesis_token="hypothesis-secret",
        hypotheses_dir=dossiers,
        hypothesis_timeout_sec=10,
        provenance_path=tmp_path / "missing-provenance.json",
    )
    app = create_app(settings)
    manager = JobManager(
        root=tmp_path,
        script_path=script,
        hypotheses_dir=dossiers,
        graphml_path=graphml,
        timeout_sec=10,
        kill_grace_sec=0.1,
    )
    app.state.jobs = manager
    headers = _hypothesis_headers()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway"
        ) as client:
            submitted = await client.post(
                "/api/hypothesis",
                headers=headers,
                json={
                    "topics": ["long-success"],
                    "profiles": ["memory-innovation"],
                    "max_hypotheses": 5,
                    "max_candidates": 12,
                    "critic": False,
                    "no_external": False,
                },
            )
            assert submitted.status_code == 202, submitted.text
            job_id = submitted.json()["job_id"]

            async def consume():
                async with client.stream(
                    "GET", f"/api/hypothesis/{job_id}/events", headers=headers
                ) as response:
                    async for _line in response.aiter_lines():
                        pass

            listener = asyncio.create_task(consume())
            await asyncio.sleep(0.3)
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await listener

            mid_run = manager.get(job_id)

            async def poll():
                while manager.get(job_id)["status"] == "running":
                    await asyncio.sleep(0.02)
                return manager.get(job_id)

            final = await asyncio.wait_for(poll(), timeout=10)
            await manager.shutdown()
            return mid_run, final

    mid_run, final = asyncio.run(scenario())

    # Proves the disconnect landed while the child was still working; without
    # it the test could pass simply by racing past a finished job.
    assert mid_run["status"] == "running", "disconnect did not land mid-run"
    assert final["status"] == "done", final["error"]
    assert final["dossier_path"] is not None


# ---------------------------------------------------------------- UX revision

def test_loopback_requests_skip_the_hypothesis_token(hypothesis_gateway):
    """Local access already implies the ability to run hypothesize.py directly,
    so the token buys nothing against a loopback caller. TestClient requests
    arrive from testclient/127.0.0.1."""
    client, _m, _d = hypothesis_gateway
    assert client.get("/api/hypothesis/profiles").status_code == 200
    assert client.get("/api/hypothesis/topics").status_code == 200


def _remote_client(client):
    """Same app, but the request arrives from a non-loopback peer (e.g. tailnet)."""
    return TestClient(client.app, client=("100.64.1.9", 50000))


def test_non_loopback_requests_still_require_the_token(hypothesis_gateway):
    """The gate must stay closed for tailnet/LAN peers, who can burn LLM budget."""
    client, _m, _d = hypothesis_gateway
    with _remote_client(client) as remote:
        assert remote.get("/api/hypothesis/profiles").status_code == 403
        assert remote.get(
            "/api/hypothesis/profiles",
            headers=_hypothesis_headers()).status_code == 200


def test_forwarded_for_cannot_fake_loopback(hypothesis_gateway):
    """Trusting X-Forwarded-For would let any remote peer claim to be local."""
    client, _m, _d = hypothesis_gateway
    with _remote_client(client) as remote:
        spoof = {"X-Forwarded-For": "127.0.0.1", "X-Real-IP": "127.0.0.1"}
        assert remote.get("/api/hypothesis/profiles", headers=spoof).status_code == 403


def test_topics_endpoint_lists_names_with_paper_counts(hypothesis_gateway):
    """The picker needs real names; counts let the user judge which are usable."""
    client, _m, _d = hypothesis_gateway
    r = client.get("/api/hypothesis/topics")
    assert r.status_code == 200
    topics = r.json()["topics"]
    by_name = {t["name"]: t["papers"] for t in topics}
    assert "CXL Memory Pooling" in by_name
    assert by_name["CXL Memory Pooling"] >= 1
    # Zero-paper topics cannot yield a hypothesis, so offering them is a trap.
    assert all(t["papers"] > 0 for t in topics)
    # HTML entities in node ids must be decoded for display.
    assert not any("&amp;" in t["name"] for t in topics)
    # Sorted by count, richest first.
    assert [t["papers"] for t in topics] == sorted(
        (t["papers"] for t in topics), reverse=True)


def test_submit_accepts_multiple_topics_and_rejects_unknown_ones(hypothesis_gateway):
    """Multi-select is how a user aims at a specific cross-domain pair."""
    client, _m, _d = hypothesis_gateway
    bad = client.post("/api/hypothesis", json={
        "topics": ["CXL Memory Pooling", "not-a-real-topic"],
        "profiles": ["memory-innovation"]})
    assert bad.status_code == 422
    body = bad.json()
    assert "not-a-real-topic" in body["invalid_topics"]
    assert body["available_topics"]

    empty = client.post("/api/hypothesis", json={
        "topics": [], "profiles": ["memory-innovation"]})
    assert empty.status_code == 422

    dupes = client.post("/api/hypothesis", json={
        "topics": ["CXL Memory Pooling", "CXL Memory Pooling"],
        "profiles": ["memory-innovation"]})
    assert dupes.status_code == 422

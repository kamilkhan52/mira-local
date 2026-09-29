"""Durable owner-scoped FIFO jobs for exhaustive research."""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import inspect
import multiprocessing
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable


TERMINAL_STATUSES = {"done", "error", "cancelled"}


class ResearchJobConflict(RuntimeError):
    def __init__(self, active_job_id: str):
        self.active_job_id = active_job_id
        super().__init__(f"research job already active: {active_job_id}")


class ResearchQueueFull(RuntimeError):
    """The bounded waiting queue has no free slot."""


class ResearchJobManagerStopped(RuntimeError):
    """No new work is accepted after shutdown starts."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {
            item.name: _json_safe(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    return value


def _child_main(
    worker: Callable[[str, dict, Callable[[Any], None]], Any],
    kind: str,
    request: dict,
    connection: Any,
) -> None:
    def emit(event: Any) -> None:
        connection.send(("progress", _json_safe(event)))

    try:
        result = worker(kind, request, emit)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        connection.send(("done", _json_safe(result)))
    except BaseException as exc:
        try:
            connection.send(("error", {
                "type": type(exc).__name__,
                "message": str(exc),
            }))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


@dataclass
class ResearchJob:
    id: str
    owner: str
    kind: str
    request: dict
    created_at: float
    status: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    cost_status: str = "cost_pending"
    cost_by_stage: dict[str, float] = field(default_factory=dict)
    last_progress: dict | None = None
    listeners: set[asyncio.Queue] = field(default_factory=set, repr=False)
    process: multiprocessing.Process | None = field(default=None, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)
    stop_reason: str | None = field(default=None, repr=False)

    def record(self, queue_position: int | None = None) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "request": _json_safe(self.request),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "queue_position": queue_position,
            "result": _json_safe(self.result),
            "error": self.error,
            "estimated_cost_usd": self.estimated_cost_usd,
            "actual_cost_usd": self.actual_cost_usd,
            "cost_status": self.cost_status,
            "cost_by_stage": dict(self.cost_by_stage),
            "last_progress": _json_safe(self.last_progress),
        }


class ResearchJobManager:
    """Run exhaustive jobs in killable children with a bounded FIFO."""

    def __init__(
        self,
        worker: Callable[[str, dict, Callable[[Any], None]], Any],
        *,
        concurrency: int = 1,
        queue_size: int = 20,
        timeout_sec: float = 3600,
        terminal_limit: int = 20,
        poll_interval: float = 0.01,
    ):
        if concurrency <= 0 or queue_size <= 0 or terminal_limit <= 0:
            raise ValueError("research job limits must be positive")
        if timeout_sec <= 0 or poll_interval <= 0:
            raise ValueError("research timing values must be positive")
        self.worker = worker
        self.concurrency = concurrency
        self.queue_size = queue_size
        self.timeout_sec = timeout_sec
        self.terminal_limit = terminal_limit
        self.poll_interval = poll_interval
        methods = multiprocessing.get_all_start_methods()
        self._context = multiprocessing.get_context(
            "spawn" if "spawn" in methods else methods[0]
        )
        self._jobs: dict[str, ResearchJob] = {}
        self._queued: collections.deque[str] = collections.deque()
        self._running: set[str] = set()
        self._active_by_owner: dict[str, str] = {}
        self._terminal: collections.deque[str] = collections.deque()
        self._lock = asyncio.Lock()
        self._stopped = False

    async def submit(self, owner: str, kind: str, request: dict) -> str:
        async with self._lock:
            if self._stopped:
                raise ResearchJobManagerStopped(
                    "research job manager is shutting down"
                )
            active = self._active_by_owner.get(owner)
            if active is not None:
                raise ResearchJobConflict(active)
            if (
                len(self._running) >= self.concurrency
                and len(self._queued) >= self.queue_size
            ):
                raise ResearchQueueFull("research queue is full")
            job_id = uuid.uuid4().hex
            job = ResearchJob(
                id=job_id,
                owner=owner,
                kind=kind,
                request=dict(request),
                created_at=time.time(),
            )
            self._jobs[job_id] = job
            self._active_by_owner[owner] = job_id
            self._queued.append(job_id)
            self._start_ready_locked()
            return job_id

    def get(self, job_id: str, owner: str) -> dict | None:
        job = self._jobs.get(job_id)
        if job is None or job.owner != owner:
            return None
        position = None
        if job.status == "queued":
            try:
                position = list(self._queued).index(job.id) + 1
            except ValueError:
                position = None
        return job.record(position)

    def active_for_owner(self, owner: str) -> str | None:
        return self._active_by_owner.get(owner)

    async def subscribe(
        self, job_id: str, owner: str
    ) -> AsyncIterator[dict]:
        job = self._jobs.get(job_id)
        if job is None or job.owner != owner:
            return
        queue: asyncio.Queue = asyncio.Queue()
        job.listeners.add(queue)
        try:
            snapshot = self.get(job_id, owner)
            if snapshot is None:
                return
            yield {"event": "snapshot", "data": snapshot}
            if snapshot["status"] in TERMINAL_STATUSES:
                return
            while True:
                event = await queue.get()
                yield event
                if event["event"] in TERMINAL_STATUSES:
                    return
        finally:
            job.listeners.discard(queue)

    async def cancel(self, job_id: str, owner: str) -> bool:
        task = None
        async with self._lock:
            job = self._jobs.get(job_id)
            if (
                job is None
                or job.owner != owner
                or job.status in TERMINAL_STATUSES
            ):
                return False
            job.stop_reason = "research job cancelled"
            if job.status == "queued":
                try:
                    self._queued.remove(job.id)
                except ValueError:
                    pass
                self._complete_locked(job, "cancelled")
                self._start_ready_locked()
                return True
            if job.process is not None and job.process.is_alive():
                job.process.terminate()
            task = job.task
        if task is not None:
            await task
        return True

    async def shutdown(self) -> None:
        tasks = []
        async with self._lock:
            if self._stopped:
                tasks = [
                    self._jobs[job_id].task
                    for job_id in self._running
                    if self._jobs[job_id].task is not None
                ]
            else:
                self._stopped = True
                for job_id in tuple(self._queued):
                    job = self._jobs[job_id]
                    job.stop_reason = "gateway shutdown"
                    self._complete_locked(job, "cancelled")
                self._queued.clear()
                for job_id in tuple(self._running):
                    job = self._jobs[job_id]
                    job.stop_reason = "gateway shutdown"
                    if job.process is not None and job.process.is_alive():
                        job.process.terminate()
                    if job.task is not None:
                        tasks.append(job.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _start_ready_locked(self) -> None:
        while (
            not self._stopped
            and self._queued
            and len(self._running) < self.concurrency
        ):
            job_id = self._queued.popleft()
            job = self._jobs[job_id]
            if job.status != "queued":
                continue
            job.status = "running"
            job.started_at = time.time()
            self._running.add(job_id)
            job.task = asyncio.create_task(self._run(job))

    async def _run(self, job: ResearchJob) -> None:
        parent, child = self._context.Pipe(duplex=False)
        process = self._context.Process(
            target=_child_main,
            args=(self.worker, job.kind, job.request, child),
            daemon=True,
        )
        job.process = process
        terminal_kind = "error"
        terminal_data: Any = {"message": "research worker exited"}
        try:
            if job.stop_reason is not None:
                terminal_kind = "cancelled"
                return
            # `spawn` re-imports this module and pickles the worker in the
            # calling thread; `recv` unpickles a result that can be megabytes.
            # Both block, so neither may run on the event loop -- every other
            # job's SSE stream stalls for the duration.
            await asyncio.to_thread(process.start)
            child.close()
            deadline = asyncio.get_running_loop().time() + self.timeout_sec
            while True:
                while parent.poll():
                    message, data = await asyncio.to_thread(parent.recv)
                    if message == "progress":
                        self._accept_progress(job, data)
                    else:
                        terminal_kind = message
                        terminal_data = data
                        break
                if terminal_kind == "done":
                    break
                if terminal_kind == "error" and terminal_data != {
                    "message": "research worker exited"
                }:
                    break
                if job.stop_reason is not None:
                    terminal_kind = "cancelled"
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    job.stop_reason = "research job timed out"
                    terminal_kind = "error"
                    terminal_data = {"message": job.stop_reason}
                    if process.is_alive():
                        process.terminate()
                    break
                if not process.is_alive():
                    while parent.poll():
                        message, data = await asyncio.to_thread(parent.recv)
                        if message == "progress":
                            self._accept_progress(job, data)
                        else:
                            terminal_kind = message
                            terminal_data = data
                    break
                await asyncio.sleep(self.poll_interval)
        except (EOFError, OSError) as exc:
            terminal_kind = "error"
            terminal_data = {"message": f"research worker failed: {exc}"}
        finally:
            if process.pid is not None and process.is_alive():
                process.terminate()
            if process.pid is not None:
                await asyncio.to_thread(process.join, 1.0)
                if process.is_alive():
                    process.kill()
                    await asyncio.to_thread(process.join, 1.0)
            parent.close()
            child.close()
            async with self._lock:
                if job.stop_reason == "research job cancelled":
                    terminal_kind = "cancelled"
                elif (
                    job.stop_reason == "gateway shutdown"
                    and terminal_kind != "done"
                ):
                    terminal_kind = "cancelled"
                if terminal_kind == "done":
                    job.result = terminal_data
                elif terminal_kind == "error":
                    job.error = str(
                        terminal_data.get("message", terminal_data)
                        if isinstance(terminal_data, dict)
                        else terminal_data
                    )
                self._complete_locked(job, terminal_kind)
                self._start_ready_locked()

    def _accept_progress(self, job: ResearchJob, value: Any) -> None:
        data = _json_safe(value)
        if not isinstance(data, dict):
            data = {"name": "progress", "value": data}
        job.last_progress = data
        name = data.get("name")
        payload = data.get("data") if isinstance(data.get("data"), dict) else data
        if name == "cost_estimated":
            job.estimated_cost_usd = float(
                payload.get("total_with_reserve_usd", 0.0)
            )
        elif name == "cost_actual":
            job.actual_cost_usd = float(
                payload.get("actual_cost_usd", job.actual_cost_usd)
            )
            job.cost_status = str(
                payload.get("cost_status", job.cost_status)
            )
            job.cost_by_stage = dict(
                payload.get("by_stage", job.cost_by_stage)
            )
        self._emit(job, "progress", data)

    def _complete_locked(self, job: ResearchJob, status: str) -> None:
        if job.status in TERMINAL_STATUSES:
            return
        job.status = status
        job.finished_at = time.time()
        self._running.discard(job.id)
        if self._active_by_owner.get(job.owner) == job.id:
            self._active_by_owner.pop(job.owner, None)
        self._terminal.append(job.id)
        while len(self._terminal) > self.terminal_limit:
            expired = self._terminal.popleft()
            self._jobs.pop(expired, None)
        self._emit(
            job,
            status,
            job.record(),
        )

    def _emit(self, job: ResearchJob, event: str, data: dict) -> None:
        payload = {"event": event, "data": _json_safe(data)}
        for listener in tuple(job.listeners):
            listener.put_nowait(payload)

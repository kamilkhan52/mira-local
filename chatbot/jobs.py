"""Async lifecycle manager for hypothesis-generation subprocesses."""
from __future__ import annotations

import asyncio
import collections
import html
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator


ROOT = Path(__file__).resolve().parent.parent
_STAGE_RE = re.compile(r"^\s*\[(\d)/5\]\s*(.*)$")
_DOSSIER_RE = re.compile(r"^\s*Dossier written:\s*(.+?)\s*$")
_WARNING_RE = re.compile(r"^\s*(?:DEGRADED|WARNING):\s*(.+?)\s*$", re.I)
# Per-hypothesis tick printed during stage 4. Synthesis is the long stage --
# one LLM call per candidate -- so without these the UI sits on [4/5] for
# minutes with no sign of movement.
_CHECK_RE = re.compile(r"^\s*✓\s*(.+?)\s*$")
_GRAPHML_NODE_RE = re.compile(
    r"""<node\b[^>]*\bid=["']([^"']+)["'][^>]*>(.*?)</node\s*>""",
    re.I | re.S,
)
_REPORT_DATA_RE = re.compile(
    r"""<data\b[^>]*>\s*Report\s*</data\s*>""", re.I | re.S
)
_TOPIC_DATA_RE = re.compile(
    r"""<data\b[^>]*>\s*Topic\s*</data\s*>""", re.I | re.S
)
_GRAPHML_EDGE_RE = re.compile(
    r"""<edge\b[^>]*\bsource=["']([^"']+)["'][^>]*\btarget=["']([^"']+)["'][^>]*>(.*?)</edge\s*>""",
    re.I | re.S,
)
# Edge keywords written by mira/graph_ingest.py for paper->topic links.
_PAPER_TOPIC_KW_RE = re.compile(r"primary_topic topic|also_covers topic", re.I)
_REPORT_DATE_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")
_DOSSIER_DEGRADED_RE = re.compile(
    r"^- \*\*Degraded:\*\*\s*(.+?)\s*$", re.M
)
_STDERR_TAIL_BYTES = 8 * 1024


class JobConflict(Exception):
    def __init__(self, active_job_id: str):
        self.active_job_id = active_job_id
        super().__init__(f"hypothesis job already active: {active_job_id}")


class JobManagerStopped(Exception):
    """Raised when a submit arrives after shutdown has begun. Distinct from
    JobConflict: the caller must not retry, because no later submit will ever
    be accepted by this manager."""


@dataclass
class HypothesisJob:
    id: str
    params: dict
    created_at: float
    status: str = "running"
    stage: int | None = None
    stage_text: str = ""
    warnings: list[str] = field(default_factory=list)
    dossier_path: str | None = None
    error: str | None = None
    listeners: set[asyncio.Queue] = field(default_factory=set, repr=False)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)
    slot_released: bool = field(default=False, repr=False)
    reported_dossier: Path | None = field(default=None, repr=False)
    stop_reason: str | None = field(default=None, repr=False)

    def record(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "params": self.params,
            "created_at": self.created_at,
            "stage": self.stage,
            "stage_text": self.stage_text,
            "warnings": list(self.warnings),
            "dossier_path": self.dossier_path,
            "error": self.error,
        }


class JobManager:
    """Own one global hypothesis child and fan its events out to listeners."""

    def __init__(
        self,
        *,
        root: Path = ROOT,
        script_path: Path | None = None,
        hypotheses_dir: Path | None = None,
        graphml_path: Path | None = None,
        timeout_sec: float = 1200,
        kill_grace_sec: float = 10,
        # Fail closed, matching Settings.exhaustive_enabled: defaulting to
        # True made a manager built without the flag spend on the exhaustive
        # path that the gateway had not enabled.
        exhaustive_enabled: bool = False,
    ):
        self.root = Path(root)
        self.script_path = Path(script_path or self.root / "hypothesize.py")
        self.hypotheses_dir = Path(
            hypotheses_dir or self.root / "data" / "report-files" / "hypotheses"
        )
        self.graphml_path = Path(
            graphml_path
            or self.root
            / "lightrag"
            / "working_dir_combined"
            / "graph_chunk_entity_relation.graphml"
        )
        self.timeout_sec = timeout_sec
        self.kill_grace_sec = kill_grace_sec
        self.exhaustive_enabled = exhaustive_enabled
        self._jobs: dict[str, HypothesisJob] = {}
        self._finished: collections.deque[str] = collections.deque()
        self._active_job_id: str | None = None
        self._lock = asyncio.Lock()
        self._shutting_down = False
        self._profiles_mtime_ns: int | None = None
        self._profiles: tuple[str, ...] = ()
        self._topics_mtime_ns: int | None = None
        self._topics: tuple[tuple[str, int], ...] = ()

    async def submit(self, params: dict) -> str:
        async with self._lock:
            # Checked under the same lock shutdown takes, so a submit can never
            # slip between shutdown's inspection of the active job and its
            # termination of the child -- that window would leave a spawned
            # process running after the gateway believed it had stopped.
            if self._shutting_down:
                raise JobManagerStopped("hypothesis job manager is shutting down")
            if self._active_job_id is not None:
                raise JobConflict(self._active_job_id)
            # Built BEFORE the slot is claimed: a malformed parameter (an
            # embedded NUL, say) must fail without leaving the slot held.
            argv = self._argv(dict(params))
            job_id = uuid.uuid4().hex
            job = HypothesisJob(
                id=job_id,
                params=dict(params),
                created_at=time.time(),
            )
            self._jobs[job_id] = job
            self._active_job_id = job_id
            try:
                job.process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=self.root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except BaseException:
                self._release_slot_once(job)
                self._jobs.pop(job_id, None)
                raise
            job.task = asyncio.create_task(self._run(job))
            return job_id

    def _argv(self, params: dict) -> list[str]:
        # Values are attached with `--opt=value` rather than as separate argv
        # elements. A topic legitimately starting with "-" (say a chemical
        # formula or a quoted fragment) would otherwise be read by the child's
        # argparse as an option token and abort the run.
        # `topics` is the current contract; `topic` is accepted so a caller
        # holding a single string still works.
        topics = list(params.get("topics") or [params["topic"]])
        profiles = list(params["profiles"])
        for value in (*topics, *profiles):
            if "\x00" in value:
                raise ValueError("argument contains an embedded NUL byte")
        argv = [
            "python3",
            "-u",
            str(self.script_path),
            "--graph=combined",
        ]
        if self.exhaustive_enabled:
            argv.append("--exhaustive")
        argv.extend(f"--topic={topic}" for topic in topics)
        argv.extend(f"--profile={profile}" for profile in profiles)
        argv.extend(
            [
                f"--max-hypotheses={params['max_hypotheses']}",
                f"--max-candidates={params['max_candidates']}",
                # The child must write where this manager scans; otherwise the
                # fallback lookup searches a directory the run never touched.
                f"--out-dir={self.hypotheses_dir}",
            ]
        )
        if params.get("critic"):
            argv.append("--critic")
        if params.get("no_external"):
            argv.append("--no-external")
        return argv

    async def subscribe(self, job_id: str) -> AsyncIterator[dict]:
        job = self._jobs[job_id]
        queue: asyncio.Queue = asyncio.Queue()
        job.listeners.add(queue)
        try:
            # Branch on the SNAPSHOT's status, never on a re-read of
            # job.status. `yield` suspends until the consumer pulls again, and
            # the job can finish in that window: the terminal event is then
            # already sitting in `queue`, but a re-read would see the new
            # terminal status and return early, stranding it. Reading the
            # snapshot we already sent keeps the two consistent -- if it says
            # "running", every later event reaches us through the queue.
            snapshot = job.record()
            yield {"event": "snapshot", "data": snapshot}
            if snapshot["status"] != "running":
                return
            while True:
                event = await queue.get()
                yield event
                if event["event"] in {"done", "error"}:
                    return
        finally:
            job.listeners.discard(queue)

    def get(self, job_id: str) -> dict | None:
        job = self._jobs.get(job_id)
        return job.record() if job is not None else None

    def discover_profiles(self) -> list[str]:
        try:
            mtime_ns = self.graphml_path.stat().st_mtime_ns
        except OSError:
            self._profiles_mtime_ns = None
            self._profiles = ()
            return []
        if mtime_ns == self._profiles_mtime_ns:
            return list(self._profiles)
        try:
            graphml = self.graphml_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return list(self._profiles)
        profiles = {
            _REPORT_DATE_RE.sub("", html.unescape(node_id))
            for node_id, body in _GRAPHML_NODE_RE.findall(graphml)
            if _REPORT_DATA_RE.search(body)
            and _REPORT_DATE_RE.search(html.unescape(node_id))
        }
        self._profiles = tuple(sorted(profiles))
        self._profiles_mtime_ns = mtime_ns
        return list(self._profiles)

    def discover_topics(self) -> list[dict]:
        """Topic names with their paper counts, richest first.

        Regex-scanned rather than loaded through NetworkX: measured at 0.1s for
        a 17 MB graph versus tens of seconds, which is what makes a
        type-to-filter picker practical. Cached on the GraphML's mtime, the same
        invalidation `discover_profiles` uses.

        Topics with zero papers are omitted -- they cannot produce a hypothesis,
        so offering them in a picker would be a trap.
        """
        try:
            mtime_ns = self.graphml_path.stat().st_mtime_ns
        except OSError:
            self._topics_mtime_ns = None
            self._topics = ()
            return []
        if mtime_ns == self._topics_mtime_ns:
            return [{"name": n, "papers": c} for n, c in self._topics]
        try:
            graphml = self.graphml_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return [{"name": n, "papers": c} for n, c in self._topics]

        topic_ids = {
            node_id for node_id, body in _GRAPHML_NODE_RE.findall(graphml)
            if _TOPIC_DATA_RE.search(body)
        }
        counts: collections.Counter = collections.Counter()
        for src, tgt, body in _GRAPHML_EDGE_RE.findall(graphml):
            if not _PAPER_TOPIC_KW_RE.search(body):
                continue
            for endpoint in (src, tgt):
                if endpoint in topic_ids:
                    counts[endpoint] += 1
        ranked = sorted(
            ((html.unescape(name), count) for name, count in counts.items()),
            key=lambda item: (-item[1], item[0].casefold()),
        )
        self._topics = tuple(ranked)
        self._topics_mtime_ns = mtime_ns
        return [{"name": n, "papers": c} for n, c in ranked]

    async def _run(self, job: HypothesisJob) -> None:
        assert job.process is not None
        stderr_tail = bytearray()
        # Terminal events are NOT emitted where they are decided. A client that
        # reacts to `done` by submitting the next job must not race this job's
        # cleanup, so the event is held here and published by the `finally`
        # block only after the run slot has been released. Emitting inline
        # would leave an `await` (the task gather below) between publication
        # and release, and a prompt client would get a spurious 409.
        terminal: tuple[str, dict] | None = None
        stdout_task = asyncio.create_task(self._read_stdout(job))
        stderr_task = asyncio.create_task(
            self._drain_stderr(job, job.process.stderr, stderr_tail)
        )
        wait_task = asyncio.create_task(job.process.wait())
        try:
            deadline = asyncio.get_running_loop().time() + self.timeout_sec
            timed_out = False
            reader_failure: Exception | None = None

            # process.returncode -- NOT wait_task -- is the authority on
            # whether the child has exited. asyncio's subprocess transport only
            # finishes once every inherited pipe has disconnected, so a
            # grandchild that inherited stdout/stderr keeps process.wait()
            # pending long after the child itself is gone. Awaiting wait_task
            # as the exit signal would block for the grandchild's entire
            # lifetime, straight past this job's timeout. returncode, by
            # contrast, is set as soon as the child is reaped.
            await self._wait_bounded({wait_task, stdout_task}, deadline)
            if stdout_task.done():
                try:
                    stdout_task.result()
                except Exception as exc:
                    reader_failure = exc
                    await self._terminate_process(job.process, wait_task)

            if reader_failure is None and job.process.returncode is None:
                # Not finished yet: allow the rest of the deadline.
                await self._wait_bounded({wait_task}, deadline)
                if job.process.returncode is None:
                    timed_out = True
                    await self._terminate_process(job.process, wait_task)

            return_code = job.process.returncode
            if not stdout_task.done():
                if await self._drain_within_deadline(stdout_task, deadline):
                    self._add_warning(job, "stdout drain truncated at deadline")
                elif reader_failure is None:
                    try:
                        stdout_task.result()
                    except Exception as exc:
                        reader_failure = exc
            elif reader_failure is None:
                try:
                    stdout_task.result()
                except Exception as exc:
                    reader_failure = exc
            if await self._drain_within_deadline(stderr_task, deadline):
                self._add_warning(job, "stderr drain truncated at deadline")
            if job.stop_reason is not None:
                job.status = "error"
                job.error = job.stop_reason
                terminal = ("error", {"message": job.error})
            elif reader_failure is not None:
                job.status = "error"
                job.error = f"stdout reader failed: {reader_failure}"
                terminal = ("error", {"message": job.error})
            elif timed_out:
                job.status = "error"
                job.error = "hypothesis generation timed out"
                terminal = ("error", {"message": job.error})
            elif return_code != 0:
                job.status = "error"
                job.error = (
                    self._decode_stderr_tail(stderr_tail)
                    or f"hypothesis process exited {return_code}"
                )
                terminal = ("error", {"message": job.error})
            else:
                dossier = self._existing_dossier(job)
                if dossier is None:
                    job.status = "error"
                    job.error = "hypothesis process exited successfully without a dossier"
                    terminal = ("error", {"message": job.error})
                else:
                    job.status = "done"
                    job.dossier_path = str(dossier)
                    # Warnings are non-terminal, so they publish immediately and
                    # are guaranteed to precede `done`.
                    self._add_dossier_warnings(job, dossier)
                    terminal = ("done", {"dossier_path": job.dossier_path})
        except asyncio.CancelledError:
            if job.process.returncode is None:
                await self._terminate_process(job.process, wait_task)
            if job.status == "running":
                job.status = "error"
                job.error = "hypothesis job cancelled"
                terminal = ("error", {"message": job.error})
            raise
        except Exception as exc:
            if job.process.returncode is None:
                await self._terminate_process(job.process, wait_task)
            if job.status == "running":
                job.status = "error"
                job.error = f"hypothesis job failed: {exc}"
                terminal = ("error", {"message": job.error})
        finally:
            for task in (stdout_task, stderr_task, wait_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                stdout_task, stderr_task, wait_task, return_exceptions=True
            )
            self._finish(job)
            self._release_slot_once(job)
            # Publish last: every listener that sees a terminal event is now
            # guaranteed to find the slot free.
            if terminal is not None:
                self._emit(job, terminal[0], terminal[1])

    async def _wait_bounded(self, tasks: set, deadline: float) -> bool:
        """Wait for any of `tasks`, never past the run deadline."""
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        done, _ = await asyncio.wait(
            tasks, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
        )
        return bool(done)

    async def _await_returncode(
        self, process: asyncio.subprocess.Process, timeout: float
    ) -> None:
        """Poll for the child's exit for at most `timeout` seconds.

        Polling rather than awaiting process.wait() is deliberate: wait() does
        not resolve until every inherited pipe closes, which a surviving
        grandchild can defer indefinitely, and signal escalation must stay
        inside its grace window regardless.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while process.returncode is None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.02, remaining))

    async def _drain_within_deadline(
        self, task: asyncio.Task, deadline: float
    ) -> bool:
        """Wait for a reader task, bounded by the run deadline. True if it timed
        out.

        A grandchild inherits the child's stdout/stderr and can hold the pipe
        open long after the child itself exits, so an unbounded await here
        would outlive the job's timeout entirely. A truncated drain is reported
        as a warning rather than an error: the run's exit code and dossier
        still decide success, and discarding a completed run over an
        undrainable pipe would throw away minutes of finished LLM work.
        """
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if not done:
            task.cancel()
            return True
        return False

    async def _terminate_process(
        self, process: asyncio.subprocess.Process, wait_task: asyncio.Task
    ) -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        await self._await_returncode(process, self.kill_grace_sec)
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            # SIGKILL cannot be caught, so this second window only covers the
            # kernel reaping the process.
            await self._await_returncode(process, self.kill_grace_sec)

    def _decode_stderr_tail(self, tail: bytearray) -> str:
        message = bytes(tail).decode("utf-8", errors="replace").strip()
        while len(message.encode("utf-8")) > _STDERR_TAIL_BYTES:
            message = message[1:]
        return message

    async def _read_stdout(self, job: HypothesisJob) -> None:
        assert job.process is not None and job.process.stdout is not None
        while line := await job.process.stdout.readline():
            text = line.decode("utf-8", errors="replace").rstrip()
            dossier_match = _DOSSIER_RE.match(text)
            if dossier_match is not None:
                candidate = Path(dossier_match.group(1))
                job.reported_dossier = (
                    candidate if candidate.is_absolute() else self.root / candidate
                )
                continue
            warning_match = _WARNING_RE.match(text)
            if warning_match is not None:
                self._add_warning(job, warning_match.group(1))
                continue
            check_match = _CHECK_RE.match(text)
            if check_match is not None:
                self._emit(
                    job,
                    "progress",
                    {
                        "stage": job.stage,
                        "stage_text": job.stage_text,
                        "hypothesis": check_match.group(1),
                    },
                )
                continue
            marker = _STAGE_RE.match(text)
            if marker is None:
                continue
            job.stage = int(marker.group(1))
            job.stage_text = marker.group(2)
            self._emit(
                job,
                "progress",
                {"stage": job.stage, "stage_text": job.stage_text},
            )

    def _expected_dirs(self, job: HypothesisJob) -> set[Path]:
        """The directories this job is allowed to have written into."""
        profiles = job.params.get("profiles", [])
        names = ["combined"] if len(profiles) > 1 else list(profiles)
        dirs: set[Path] = set()
        for name in names:
            try:
                dirs.add((self.hypotheses_dir / name).resolve())
            except (OSError, RuntimeError):
                continue
        return dirs

    def _validated_dossier(
        self, job: HypothesisJob, candidate: Path
    ) -> Path | None:
        """Accept a dossier path only if it is a real Markdown file inside this
        job's own output directory, written during this run.

        The `Dossier written:` line is child-controlled input and the result is
        later read and served to clients, so it is validated exactly like a
        request parameter rather than trusted because the child printed it.
        """
        try:
            root = self.hypotheses_dir.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if resolved.suffix.lower() != ".md" or not resolved.is_file():
            return None
        # resolve() collapses symlinks, so a link sitting inside the root but
        # pointing outside it fails containment here rather than laundering
        # its target into a servable path.
        if not resolved.is_relative_to(root):
            return None
        if resolved.parent not in self._expected_dirs(job):
            return None
        try:
            # One second of slack absorbs coarse filesystem timestamp
            # granularity; a genuinely stale dossier predates the run by far
            # more than that.
            if resolved.stat().st_mtime < job.created_at - 1.0:
                return None
        except OSError:
            return None
        return resolved

    def _existing_dossier(self, job: HypothesisJob) -> Path | None:
        candidate = job.reported_dossier
        if candidate is not None:
            # A reported path that fails validation is a hard failure, not a
            # cue to go hunting: the child said where it wrote, and that
            # location is not servable.
            return self._validated_dossier(job, candidate)

        candidates: list[tuple[int, Path]] = []
        for output_dir in self._expected_dirs(job):
            try:
                entries = list(output_dir.iterdir())
            except (OSError, RuntimeError):
                continue
            for path in entries:
                validated = self._validated_dossier(job, path)
                if validated is None:
                    continue
                try:
                    candidates.append((validated.stat().st_mtime_ns, validated))
                except OSError:
                    continue
        return max(candidates, default=(0, None), key=lambda item: item[0])[1]

    def _add_dossier_warnings(
        self, job: HypothesisJob, dossier: Path
    ) -> None:
        try:
            markdown = dossier.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return
        for warning in _DOSSIER_DEGRADED_RE.findall(markdown):
            self._add_warning(job, warning)

    async def _drain_stderr(
        self,
        job: HypothesisJob,
        stream: asyncio.StreamReader | None,
        tail: bytearray,
    ) -> None:
        assert stream is not None
        pending = bytearray()
        while chunk := await stream.read(4096):
            tail.extend(chunk)
            if len(tail) > _STDERR_TAIL_BYTES:
                del tail[: len(tail) - _STDERR_TAIL_BYTES]
            pending.extend(chunk)
            while b"\n" in pending:
                raw_line, _, remainder = pending.partition(b"\n")
                pending = bytearray(remainder)
                self._parse_stderr_warning(job, raw_line)
            if len(pending) > _STDERR_TAIL_BYTES:
                del pending[: len(pending) - _STDERR_TAIL_BYTES]
        if pending:
            self._parse_stderr_warning(job, bytes(pending))

    def _parse_stderr_warning(self, job: HypothesisJob, line: bytes) -> None:
        match = _WARNING_RE.match(line.decode("utf-8", errors="replace"))
        if match is not None:
            self._add_warning(job, match.group(1))

    def _add_warning(self, job: HypothesisJob, warning: str) -> None:
        if warning in job.warnings:
            return
        job.warnings.append(warning)
        self._emit(job, "warnings", {"warnings": list(job.warnings)})

    def _emit(self, job: HypothesisJob, event: str, data: dict) -> None:
        payload = {"event": event, "data": data}
        for listener in tuple(job.listeners):
            listener.put_nowait(payload)

    def _finish(self, job: HypothesisJob) -> None:
        if job.status == "running":
            return
        self._finished.append(job.id)
        while len(self._finished) > 20:
            old_job_id = self._finished.popleft()
            self._jobs.pop(old_job_id, None)

    def _release_slot_once(self, job: HypothesisJob) -> None:
        if job.slot_released:
            return
        job.slot_released = True
        if self._active_job_id == job.id:
            self._active_job_id = None

    async def shutdown(self) -> None:
        # Serialize with submit's spawn window: shutdown must not observe an
        # allocated job before its process/task ownership has been installed.
        async with self._lock:
            # Latch before releasing the lock so no submit can spawn a child
            # after this point; otherwise a job started during the termination
            # below would survive the gateway.
            self._shutting_down = True
            job = (
                self._jobs.get(self._active_job_id)
                if self._active_job_id is not None
                else None
            )
        if job is None or job.task is None:
            return
        job.stop_reason = "gateway shutdown"
        process = job.process
        try:
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(
                        asyncio.shield(job.task), timeout=self.kill_grace_sec
                    )
                except TimeoutError:
                    if process.returncode is None:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
            await job.task
        finally:
            self._release_slot_once(job)

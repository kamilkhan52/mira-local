"""FastAPI gateway: the ONLY user-reachable surface for the combined graph.

The chat path proxies exactly two READ operations to LightRAG (:9623); none of
LightRAG's insert/delete/graph-management endpoints are reachable through it.

Two write paths exist on top of that read-only core, each gated by its own
fail-closed shared secret so either can be revoked without the other:

  * /api/upload      — writes to the graph (POST /documents/text), X-Upload-Token
  * /api/hypothesis  — spawns hypothesize.py, writing dossier files and spending
                       LLM budget, X-Hypothesis-Token

Both are unreachable when their token is unset. Adding a third write path is a
deliberate security decision, not a routine feature addition.
"""
import asyncio
import collections
import json
import os
import stat
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from typing import Literal

import httpx
from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import (FileResponse, JSONResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.background import BackgroundTask

from chatbot.config import (Settings, is_hypothesis_token_valid,
                            is_loopback_client)
from chatbot.jobs import JobConflict, JobManager, JobManagerStopped
from chatbot.research_jobs import (
    ResearchJobConflict,
    ResearchJobManager,
    ResearchJobManagerStopped,
    ResearchQueueFull,
)
from mira.exhaustive.providers import run_research_job

STATIC_DIR = Path(__file__).parent / "static"
MODES = ("mix", "hybrid", "global", "local", "naive")
# Leeway on the Content-Length pre-check for multipart boundary/header
# overhead, so a file right at the cap isn't rejected for its envelope.
_UPLOAD_ENVELOPE_BYTES = 8192


def _safe_filename(name: str | None) -> str:
    """Reduce an uploaded filename to a safe basename: strip directory
    components (both / and \\), drop non-printable chars, cap length. It is
    forwarded to LightRAG as metadata.file_path, so it must not carry path
    traversal or control characters."""
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    base = "".join(ch for ch in base if ch.isprintable()).strip()
    return base[:255] or "upload"


def _is_safe_dossier_component(value: str) -> bool:
    """Accept only a single, printable filename component for dossier routes."""
    return value not in ("", ".", "..") and _safe_filename(value) == value


def _inode_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


@dataclass(frozen=True)
class DossierEntry:
    root: Path
    root_identity: tuple[int, int]
    profile: str
    profile_identity: tuple[int, int]
    filename: str
    file_identity: tuple[int, int]
    mtime: float


def _dossier_listing(root: Path) -> list[DossierEntry]:
    """Return only immediate profile/markdown files contained by ``root``.

    The returned paths are server-discovered rather than built from request
    parameters.  Resolve each entry so a symlink cannot make the listing expose
    a file outside the configured dossier tree.
    """
    try:
        resolved_root = Path(root).resolve(strict=True)
    except (OSError, RuntimeError):
        return []
    if not resolved_root.is_dir():
        return []
    try:
        root_identity = _inode_identity(resolved_root.stat())
    except OSError:
        return []

    dossiers = []
    try:
        profile_dirs = resolved_root.iterdir()
        for profile_dir in profile_dirs:
            if (not _is_safe_dossier_component(profile_dir.name)
                    or profile_dir.is_symlink()):
                continue
            try:
                resolved_profile = profile_dir.resolve(strict=True)
                resolved_profile.relative_to(resolved_root)
            except (OSError, RuntimeError, ValueError):
                continue
            if not resolved_profile.is_dir() or resolved_profile.parent != resolved_root:
                continue
            try:
                profile_identity = _inode_identity(resolved_profile.stat())
            except OSError:
                continue
            for candidate in resolved_profile.iterdir():
                if (not _is_safe_dossier_component(candidate.name)
                        or candidate.suffix.lower() != ".md"
                        or candidate.is_symlink()):
                    continue
                try:
                    resolved_candidate = candidate.resolve(strict=True)
                    resolved_candidate.relative_to(resolved_profile)
                    if not resolved_candidate.is_file():
                        continue
                    candidate_info = resolved_candidate.stat()
                except (OSError, RuntimeError, ValueError):
                    continue
                dossiers.append(DossierEntry(
                    root=resolved_root,
                    root_identity=root_identity,
                    profile=profile_dir.name,
                    profile_identity=profile_identity,
                    filename=candidate.name,
                    file_identity=_inode_identity(candidate_info),
                    mtime=candidate_info.st_mtime,
                ))
    except OSError:
        return []
    return sorted(dossiers, key=lambda item: (item.profile, item.filename))


def _pin_absolute_directory(path: Path, flags: int) -> int | None:
    """Open an absolute directory one no-follow component at a time from `/`."""
    path = Path(path)
    if not path.is_absolute():
        return None
    fd = None
    try:
        fd = os.open("/", flags)
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            try:
                os.close(fd)
            except (OSError, TypeError):
                try:
                    os.close(next_fd)
                except (OSError, TypeError):
                    pass
                raise
            fd = next_fd
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            os.close(fd)
            return None
        return fd
    except (OSError, NotImplementedError, TypeError):
        if fd is not None:
            try:
                os.close(fd)
            except (OSError, TypeError):
                pass
        return None


def _read_listed_dossier(entry: DossierEntry) -> str | None:
    """Read a listed dossier through pinned, no-follow directory descriptors.

    The complete entry must come from `_dossier_listing`; its captured inode
    identities reject a rename-and-replace race even when all replacements are
    real files/directories rather than symlinks.
    `O_NONBLOCK` prevents a post-listing replacement with a FIFO from blocking
    before its `fstat` rejects anything other than a regular file.
    """
    if (not _is_safe_dossier_component(entry.profile)
            or not _is_safe_dossier_component(entry.filename)):
        return None
    try:
        directory_flags = os.O_DIRECTORY | os.O_NOFOLLOW
        nonblock_flag = os.O_NONBLOCK
    except AttributeError:
        # These flags are required to make symlink swaps fail closed.
        return None
    common_flags = os.O_RDONLY | nonblock_flag | getattr(os, "O_CLOEXEC", 0)
    root_fd = profile_fd = file_fd = None
    try:
        # Pin every captured-root ancestor from trusted `/`, since O_NOFOLLOW
        # only protects the final component of one open.
        root_fd = _pin_absolute_directory(entry.root,
                                          common_flags | directory_flags)
        if (root_fd is None
                or _inode_identity(os.fstat(root_fd)) != entry.root_identity):
            return None
        profile_fd = os.open(entry.profile, common_flags | directory_flags,
                             dir_fd=root_fd)
        profile_info = os.fstat(profile_fd)
        if (not stat.S_ISDIR(profile_info.st_mode)
                or _inode_identity(profile_info) != entry.profile_identity):
            return None
        file_fd = os.open(entry.filename, common_flags | os.O_NOFOLLOW,
                          dir_fd=profile_fd)
        size = os.fstat(file_fd)
        if (not stat.S_ISREG(size.st_mode)
                or _inode_identity(size) != entry.file_identity):
            return None
        # Bound reads to the byte count observed for this regular file, so a
        # swapped FIFO/device is never consumed and a growing file is not read
        # indefinitely.
        remaining = size.st_size
        chunks = []
        while remaining:
            try:
                chunk = os.read(file_fd, min(remaining, 64 * 1024))
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks).decode("utf-8")
    except (OSError, NotImplementedError, TypeError, UnicodeError):
        return None
    finally:
        for fd in (file_fd, profile_fd, root_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except (OSError, TypeError):
                    pass


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    mode: Literal["mix", "hybrid", "global", "local", "naive"] = "mix"
    history: list[Turn] = []
    top_k: int | None = Field(default=None, ge=1, le=100)
    chunk_top_k: int | None = Field(default=None, ge=1, le=30)


class ExhaustiveChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    history: list[Turn] = []


class HypothesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Multi-select: picking one memory topic and one optical topic aims
    # explicitly at a cross-domain pair, instead of hoping a typed sentence
    # happened to substring-match both.
    topics: list[str] = Field(min_length=1, max_length=10)
    profiles: list[str] = Field(min_length=1)
    max_hypotheses: int = Field(default=5, ge=1, le=10)
    max_candidates: int = Field(default=12, ge=1, le=30)
    critic: bool = False
    no_external: bool = False

    @field_validator("profiles")
    @classmethod
    def profiles_are_unique(cls, profiles: list[str]) -> list[str]:
        if len(profiles) != len(set(profiles)):
            raise ValueError("profiles must be unique")
        return profiles

    @field_validator("topics")
    @classmethod
    def topics_are_unique_and_bounded(cls, topics: list[str]) -> list[str]:
        if len(topics) != len(set(topics)):
            raise ValueError("topics must be unique")
        if any(not t.strip() or len(t) > 200 for t in topics):
            raise ValueError("each topic must be 1-200 characters")
        return topics


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ProvenanceStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._mtime = -1.0
        self._data = {"entities": {}, "chunks": {}, "relations": {}}
        self._maybe_reload()

    def _maybe_reload(self):
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return
        if mtime != self._mtime:
            try:
                self._data = json.loads(self.path.read_text())
                self._mtime = mtime
            except (OSError, ValueError):
                pass

    def domain(self, kind: str, key: str) -> str:
        self._maybe_reload()
        table = self._data.get(kind, {})
        # Exact match first. The provenance writer (build_provenance in
        # mira/graph_merge.py) stores relation keys in STORAGE order
        # (src<SEP>tgt as found in the KV store), NOT sorted, and the upstream
        # API may hand us endpoints in either order. So for relations, fall
        # back to the reversed key rather than assuming a sorted canonical form.
        if key in table:
            return table[key]
        if kind == "relations" and key.count("<SEP>") == 1:
            src, tgt = key.split("<SEP>")
            reversed_key = f"{tgt}<SEP>{src}"
            if reversed_key in table:
                return table[reversed_key]
        return "unknown"


class RateLimiter:
    def __init__(self, per_min: int, concurrent: int):
        self.per_min = per_min
        self.concurrent = concurrent
        self._hits: dict[str, collections.deque] = collections.defaultdict(
            collections.deque)
        self._active: collections.Counter = collections.Counter()

    def acquire(self, ip: str) -> bool:
        now = time.monotonic()
        hits = self._hits[ip]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_min or self._active[ip] >= self.concurrent:
            return False
        hits.append(now)
        self._active[ip] += 1
        return True

    def release(self, ip: str) -> None:
        if self._active[ip] > 0:
            self._active[ip] -= 1


def create_app(
    settings: Settings,
    *,
    research_jobs: ResearchJobManager | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(live_app: FastAPI):
        try:
            yield
        finally:
            await live_app.state.jobs.shutdown()
            await live_app.state.research_jobs.shutdown()
            await live_app.state.client.aclose()

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.provenance = ProvenanceStore(settings.provenance_path)
    app.state.limiter = RateLimiter(settings.rate_per_min,
                                    settings.rate_concurrent)
    app.state.client = httpx.AsyncClient(
        base_url=settings.lightrag_url,
        headers={"X-API-Key": settings.lightrag_api_key},
        timeout=httpx.Timeout(15.0, read=180.0),
    )
    app.state.jobs = JobManager(
        hypotheses_dir=settings.hypotheses_dir,
        timeout_sec=settings.hypothesis_timeout_sec,
        exhaustive_enabled=settings.exhaustive_enabled,
    )
    app.state.research_jobs = research_jobs or ResearchJobManager(
        partial(run_research_job, settings),
        concurrency=settings.exhaustive_concurrency,
        queue_size=settings.exhaustive_queue_size,
        timeout_sec=settings.exhaustive_job_timeout_sec,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        if (request.url.path.startswith("/api/hypothesis")
                and not _hypothesis_authorized(request)):
            return JSONResponse(
                {"detail": "hypothesis not authorized"}, status_code=403
            )
        if request.url.path.startswith("/api/research"):
            if not app.state.settings.exhaustive_enabled:
                return JSONResponse(
                    {"detail": "exhaustive research is disabled"},
                    status_code=503,
                )
            if not _research_authorized(request):
                return JSONResponse(
                    {"detail": "research not authorized"}, status_code=403
                )
        # Reject oversized uploads by declared length BEFORE Starlette parses
        # and spools the multipart body. The handler still enforces the exact
        # cap for requests that omit or understate Content-Length.
        if request.url.path == "/api/upload" and request.method == "POST":
            declared = request.headers.get("content-length")
            if declared and declared.isdigit():
                s: Settings = app.state.settings
                if int(declared) > s.max_upload_bytes + _UPLOAD_ENVELOPE_BYTES:
                    return JSONResponse(
                        {"detail": f"file too large (max {s.max_upload_bytes} bytes)"},
                        status_code=413)
        resp: Response = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        path = request.url.path
        if path.startswith("/api/") or path.endswith((".html", ".css", ".js")) or path == "/":
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return resp

    @app.get("/api/health")
    async def health():
        try:
            r = await app.state.client.get("/health")
            up = r.status_code == 200 and r.json().get("status") == "healthy"
        except httpx.HTTPError:
            up = False
        return JSONResponse({"ok": up, "upstream": settings.lightrag_url})

    def _hypothesis_authorized(request: Request) -> bool:
        # A caller on the loopback interface already has local access, which is
        # the same trust level as running hypothesize.py directly -- so the
        # token buys nothing against them. It remains mandatory for every remote
        # peer, who could otherwise spend LLM budget.
        #
        # request.client.host is the real socket peer. Deliberately NOT
        # X-Forwarded-For: that is attacker-controlled and would let any remote
        # caller claim to be local.
        if (
            is_loopback_client(request.client.host if request.client else None)
            and not request.headers.get("Tailscale-User-Login")
        ):
            return True
        return is_hypothesis_token_valid(
            request.headers.get("X-Hypothesis-Token", ""),
            app.state.settings.hypothesis_token)

    def _research_authorized(request: Request) -> bool:
        if (
            is_loopback_client(request.client.host if request.client else None)
            and not request.headers.get("Tailscale-User-Login")
        ):
            return True
        return is_hypothesis_token_valid(
            request.headers.get("X-Research-Token", ""),
            app.state.settings.research_token,
        )

    def _client_ip(request: Request) -> str:
        return request.client.host if request.client else "unknown"

    def _research_owner(request: Request) -> str:
        peer = request.client.host if request.client else "unknown"
        tailscale_login = (
            request.headers.get("Tailscale-User-Login", "").strip()[:256]
            if is_loopback_client(peer)
            else ""
        )
        return f"{peer}|{tailscale_login}" if tailscale_login else peer

    def _validate_research_chat(req: ExhaustiveChatRequest) -> str | None:
        s: Settings = app.state.settings
        history_chars = sum(len(turn.content) for turn in req.history)
        if (
            len(req.query) > s.max_query_chars
            or len(req.history) > s.max_history_turns
            or history_chars > s.max_history_chars
            or any(
                turn.role == "user"
                and len(turn.content) > s.max_query_chars
                for turn in req.history
            )
        ):
            return "input too large"
        return None

    async def _submit_research(
        owner: str, kind: str, payload: dict, ip: str
    ) -> JSONResponse:
        if not app.state.settings.exhaustive_enabled:
            return JSONResponse(
                {"detail": "exhaustive research is disabled"},
                status_code=503,
            )
        # Same per-IP budget as chat and upload. A research job is the most
        # expensive thing this gateway can start, so submission is the one
        # research call that must be rate limited; the slot is released as
        # soon as the job is queued, since the job itself is bounded by the
        # manager's own concurrency and FIFO.
        if not app.state.limiter.acquire(ip):
            return JSONResponse(
                {"detail": "rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        try:
            job_id = await app.state.research_jobs.submit(
                owner, kind, payload
            )
        except ResearchJobConflict as exc:
            return JSONResponse(
                {
                    "detail": "a research job is already active",
                    "active_job_id": exc.active_job_id,
                },
                status_code=409,
            )
        except ResearchQueueFull:
            return JSONResponse(
                {"detail": "research queue is full"},
                status_code=503,
                headers={"Retry-After": "30"},
            )
        except ResearchJobManagerStopped:
            return JSONResponse(
                {"detail": "gateway is shutting down"}, status_code=503
            )
        finally:
            app.state.limiter.release(ip)
        record = app.state.research_jobs.get(job_id, owner)
        return JSONResponse(
            {
                "job_id": job_id,
                "queue_position": (
                    record.get("queue_position") if record else None
                ),
            },
            status_code=202,
        )

    @app.post("/api/research/chat")
    async def submit_research_chat(
        req: ExhaustiveChatRequest, request: Request
    ):
        error = _validate_research_chat(req)
        if error:
            return JSONResponse({"detail": error}, status_code=422)
        return await _submit_research(
            _research_owner(request),
            "chat",
            req.model_dump(),
            _client_ip(request),
        )

    @app.post("/api/research/hypotheses")
    async def submit_research_hypotheses(
        req: HypothesisRequest, request: Request
    ):
        return await _submit_research(
            _research_owner(request),
            "hypotheses",
            req.model_dump(),
            _client_ip(request),
        )

    @app.get("/api/research/active")
    async def active_research(request: Request):
        owner = _research_owner(request)
        job_id = app.state.research_jobs.active_for_owner(owner)
        return JSONResponse({
            "active_job_id": job_id,
            "job": (
                app.state.research_jobs.get(job_id, owner)
                if job_id else None
            ),
        })

    @app.get("/api/research/{job_id}/events")
    async def research_events(job_id: str, request: Request):
        owner = _research_owner(request)
        if app.state.research_jobs.get(job_id, owner) is None:
            return JSONResponse(
                {"detail": "research job not found"}, status_code=404
            )

        async def event_stream():
            async for event in app.state.research_jobs.subscribe(
                job_id, owner
            ):
                yield sse(event["event"], event["data"])

        return StreamingResponse(
            event_stream(), media_type="text/event-stream"
        )

    @app.get("/api/research/{job_id}")
    async def get_research_job(job_id: str, request: Request):
        record = app.state.research_jobs.get(
            job_id, _research_owner(request)
        )
        if record is None:
            return JSONResponse(
                {"detail": "research job not found"}, status_code=404
            )
        return JSONResponse(record)

    @app.delete("/api/research/{job_id}")
    async def cancel_research_job(job_id: str, request: Request):
        owner = _research_owner(request)
        if app.state.research_jobs.get(job_id, owner) is None:
            return JSONResponse(
                {"detail": "research job not found"}, status_code=404
            )
        await app.state.research_jobs.cancel(job_id, owner)
        return JSONResponse(app.state.research_jobs.get(job_id, owner))

    @app.get("/api/hypothesis/profiles")
    async def hypothesis_profiles(request: Request):
        if not _hypothesis_authorized(request):
            return JSONResponse({"detail": "hypothesis not authorized"},
                                status_code=403)
        return JSONResponse({"profiles": app.state.jobs.discover_profiles()})

    @app.get("/api/hypothesis/topics")
    async def hypothesis_topics(request: Request):
        """Feeds the picker. Cheap enough (0.1s, cached on mtime) to send the
        whole list so the client can filter locally without per-keystroke
        round-trips."""
        if not _hypothesis_authorized(request):
            return JSONResponse({"detail": "hypothesis not authorized"},
                                status_code=403)
        return JSONResponse({"topics": app.state.jobs.discover_topics()})

    @app.post("/api/hypothesis")
    async def submit_hypothesis(req: HypothesisRequest, request: Request):
        if not _hypothesis_authorized(request):
            return JSONResponse({"detail": "hypothesis not authorized"},
                                status_code=403)
        # No exhaustive-mode 409 here: this endpoint runs the CLI, which the
        # job manager invokes with --exhaustive when the feature is on, so the
        # request is served rather than refused. /api/research/hypotheses is
        # the queued alternative with progress and cost events, not a
        # replacement for the synchronous path.
        # Topics come from the picker, but validate anyway -- a client is not a
        # trust boundary, and an unvetted value would reach the CLI as an argv
        # element that simply matches nothing.
        known_topics = {t["name"] for t in app.state.jobs.discover_topics()}
        unknown_topics = sorted({t for t in req.topics if t not in known_topics})
        if unknown_topics:
            return JSONResponse(
                {
                    "detail": "topics are not present in the combined graph",
                    "invalid_topics": unknown_topics,
                    "available_topics": sorted(known_topics)[:50],
                },
                status_code=422,
            )
        available = app.state.jobs.discover_profiles()
        available_set = set(available)
        invalid = sorted({
            profile for profile in req.profiles if profile not in available_set
        })
        if invalid:
            return JSONResponse(
                {
                    "detail": "profiles are not available in the combined graph",
                    "invalid_profiles": invalid,
                    "available_profiles": available,
                },
                status_code=422,
            )
        try:
            job_id = await app.state.jobs.submit(req.model_dump())
        except JobConflict as exc:
            return JSONResponse(
                {
                    "detail": "a hypothesis job is already running",
                    "active_job_id": exc.active_job_id,
                },
                status_code=409,
            )
        except JobManagerStopped:
            # The gateway is shutting down. 503, not 409: retrying against this
            # process will never succeed.
            return JSONResponse(
                {"detail": "gateway is shutting down"}, status_code=503
            )
        except ValueError as exc:
            # Malformed argv input (an embedded NUL, say) rejected before spawn.
            return JSONResponse({"detail": str(exc)}, status_code=422)
        return JSONResponse({"job_id": job_id}, status_code=202)

    @app.get("/api/hypothesis/dossiers")
    async def list_dossiers(request: Request):
        if not _hypothesis_authorized(request):
            return JSONResponse({"detail": "hypothesis not authorized"},
                                status_code=403)
        return JSONResponse([
            {"profile": entry.profile, "filename": entry.filename,
             "mtime": entry.mtime}
            for entry in
            _dossier_listing(app.state.settings.hypotheses_dir)
        ])

    @app.get("/api/hypothesis/dossiers/{profile}/{name}")
    async def get_dossier(profile: str, name: str, request: Request):
        if not _hypothesis_authorized(request):
            return JSONResponse({"detail": "hypothesis not authorized"},
                                status_code=403)
        if (not _is_safe_dossier_component(profile)
                or not _is_safe_dossier_component(name)):
            return JSONResponse({"detail": "dossier not found"}, status_code=404)

        # Request components must select an entry the server listed; they never
        # participate in constructing a filesystem path.
        allowed = {
            (entry.profile, entry.filename): entry
            for entry in
            _dossier_listing(app.state.settings.hypotheses_dir)
        }
        entry = allowed.get((profile, name))
        if entry is None:
            return JSONResponse({"detail": "dossier not found"}, status_code=404)
        content = _read_listed_dossier(entry)
        if content is None:
            return JSONResponse({"detail": "dossier not found"}, status_code=404)
        return Response(content=content, media_type="text/markdown")

    @app.get("/api/hypothesis/{job_id}/events")
    async def hypothesis_events(job_id: str, request: Request):
        if not _hypothesis_authorized(request):
            return JSONResponse({"detail": "hypothesis not authorized"},
                                status_code=403)
        if app.state.jobs.get(job_id) is None:
            return JSONResponse({"detail": "hypothesis job not found"},
                                status_code=404)

        async def event_stream():
            async for event in app.state.jobs.subscribe(job_id):
                yield sse(event["event"], event["data"])

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/hypothesis/{job_id}")
    async def get_hypothesis_job(job_id: str, request: Request):
        if not _hypothesis_authorized(request):
            return JSONResponse({"detail": "hypothesis not authorized"},
                                status_code=403)
        record = app.state.jobs.get(job_id)
        if record is None:
            return JSONResponse({"detail": "hypothesis job not found"},
                                status_code=404)
        dossier_markdown = None
        if record["status"] == "done" and record["dossier_path"]:
            try:
                job_path = Path(record["dossier_path"]).resolve(strict=True)
            except (OSError, RuntimeError):
                job_path = None
            if job_path is not None:
                for entry in _dossier_listing(
                    app.state.settings.hypotheses_dir
                ):
                    listed_path = (
                        entry.root / entry.profile / entry.filename
                    )
                    if listed_path == job_path:
                        dossier_markdown = _read_listed_dossier(entry)
                        break
        return JSONResponse({
            **record,
            "dossier_markdown": dossier_markdown,
        })

    def _label_sources(payload: dict) -> dict:
        prov = app.state.provenance
        data = payload.get("data", {})
        meta = payload.get("metadata", {})
        entities = [
            {"name": e.get("entity_name"), "type": e.get("entity_type"),
             "description": e.get("description"),
             "domain": prov.domain("entities", e.get("entity_name", ""))}
            for e in data.get("entities", [])]
        relationships = [
            {"src": r.get("src_id"), "tgt": r.get("tgt_id"),
             "description": r.get("description"),
             "domain": prov.domain(
                 "relations", f'{r.get("src_id")}<SEP>{r.get("tgt_id")}')}
            for r in data.get("relationships", [])]
        references = [
            {"id": ref.get("reference_id"), "file_path": ref.get("file_path"),
             "domain": "unknown"}
            for ref in data.get("references", [])]
        return {"entities": entities, "relationships": relationships,
                "references": references, "mode": meta.get("query_mode"),
                "keywords": meta.get("keywords", {})}

    @app.post("/api/chat")
    async def chat(req: ChatRequest, request: Request):
        s: Settings = app.state.settings
        if s.exhaustive_enabled:
            return JSONResponse(
                {
                    "detail": (
                        "bounded chat is disabled while exhaustive research "
                        "is enabled"
                    )
                },
                status_code=409,
            )
        history_chars = sum(len(t.content) for t in req.history)
        if (len(req.query) > s.max_query_chars
                or len(req.history) > s.max_history_turns
                or history_chars > s.max_history_chars
                or any(t.role == "user" and len(t.content) > s.max_query_chars for t in req.history)):
            return JSONResponse({"detail": "input too large"}, status_code=422)
        ip = request.client.host if request.client else "unknown"
        if not app.state.limiter.acquire(ip):
            return JSONResponse({"detail": "rate limit exceeded"},
                                status_code=429,
                                headers={"Retry-After": "60"})

        # One-shot releaser: the slot must be freed exactly once per request.
        # It runs in the generator's `finally` on normal paths, but Starlette
        # pulls the generator lazily -- if the client disconnects before the
        # first item is drawn, `gen()`'s body (and its `finally`) never runs.
        # The BackgroundTask below covers that window. The guard makes the two
        # release sites idempotent so a completed request cannot free a slot
        # belonging to a DIFFERENT in-flight request from the same IP.
        released = False

        def release_once() -> None:
            nonlocal released
            if not released:
                released = True
                app.state.limiter.release(ip)

        async def gen():
            client: httpx.AsyncClient = app.state.client
            stream_body = {
                "query": req.query, "mode": req.mode, "stream": True,
                "include_references": True,
                "conversation_history": [t.model_dump() for t in req.history],
            }
            data_body = {"query": req.query, "mode": req.mode,
                         "include_chunk_content": False}
            for k in ("top_k", "chunk_top_k"):
                v = getattr(req, k)
                if v is not None:
                    stream_body[k] = v
                    data_body[k] = v

            async def fetch_sources():
                try:
                    r = await client.post("/query/data", json=data_body,
                                          timeout=60.0)
                    r.raise_for_status()
                    return _label_sources(r.json())
                except (httpx.HTTPError, ValueError) as e:
                    return {"entities": [], "relationships": [],
                            "references": [], "mode": req.mode, "keywords": {},
                            "error": f"sources unavailable: {type(e).__name__}"}

            sources_task = asyncio.create_task(fetch_sources())
            got_tokens = False
            try:
                async with client.stream("POST", "/query/stream",
                                         json=stream_body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            continue
                        if "response" in obj:
                            got_tokens = True
                            yield sse("token", {"text": obj["response"]})
                        if sources_task is not None and sources_task.done():
                            yield sse("sources", sources_task.result())
                            sources_task = None
                if sources_task is not None:
                    yield sse("sources", await sources_task)
                    sources_task = None
                yield sse("done", {})
            except httpx.HTTPError as e:
                msg = ("Knowledge graph service is unavailable"
                       if not got_tokens else
                       f"stream interrupted: {type(e).__name__}")
                yield sse("error", {"message": msg, "retryable": True})
            finally:
                if sources_task is not None:
                    sources_task.cancel()
                release_once()

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 background=BackgroundTask(release_once))

    @app.post("/api/upload")
    async def upload(request: Request, file: UploadFile = File(...)):
        import secrets

        s: Settings = app.state.settings

        # Auth: /api/upload WRITES to the graph (POST /documents/text), so it is
        # gated by a shared token on top of the tailnet, unlike the read-only
        # chat path. Fail closed -- an unset token disables uploads entirely --
        # and compare in constant time so the check can't be timing-probed.
        token = request.headers.get("X-Upload-Token", "")
        if not s.upload_token or not secrets.compare_digest(token, s.upload_token):
            return JSONResponse({"detail": "upload not authorized"},
                                status_code=403)

        # Rate limit the same way as chat: one shared per-IP budget + a
        # concurrency slot, released in the finally below.
        ip = request.client.host if request.client else "unknown"
        if not app.state.limiter.acquire(ip):
            return JSONResponse({"detail": "rate limit exceeded"},
                                status_code=429, headers={"Retry-After": "60"})
        try:
            return await _ingest_upload(file, s)
        finally:
            app.state.limiter.release(ip)

    async def _ingest_upload(file: UploadFile, s: Settings):
        import io
        import pypdf

        filename = _safe_filename(file.filename)
        # Bounded read: never pull more than the cap (+1 to detect overflow)
        # into memory, even when Content-Length was absent or understated and
        # the middleware pre-check let the request through.
        content = await file.read(s.max_upload_bytes + 1)

        if len(content) > s.max_upload_bytes:
            return JSONResponse(
                {"detail": f"file too large (max {s.max_upload_bytes} bytes)"},
                status_code=413)

        if filename.endswith(".pdf"):
            try:
                pdf_file = io.BytesIO(content)
                reader = pypdf.PdfReader(pdf_file)
                text_parts = []
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
                text = "\n".join(text_parts)
            except Exception as e:
                return JSONResponse({"detail": f"Failed to parse PDF: {str(e)}"}, status_code=400)
        elif filename.endswith((".txt", ".md", ".json")):
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = content.decode("latin-1")
                except Exception:
                    return JSONResponse({"detail": "Failed to decode text file"}, status_code=400)
        else:
            return JSONResponse({"detail": "Unsupported file format. Use PDF, TXT or MD."}, status_code=400)

        if not text.strip():
            return JSONResponse({"detail": "No text content found in file"}, status_code=400)

        client = app.state.client
        try:
            r = await client.post("/documents/text", json={
                "text": text,
                "metadata": {"file_path": filename}
            }, timeout=90.0)
            r.raise_for_status()
        except Exception as e:
            return JSONResponse({"detail": f"Upstream ingestion failed: {str(e)}"}, status_code=500)

        return JSONResponse({"ok": True, "filename": filename, "chars": len(text)})

    @app.get("/")
    async def index():
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Content-Security-Policy": "default-src 'self'"})

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def app():
    """Uvicorn factory entrypoint: uvicorn chatbot.app:app --factory"""
    from chatbot.config import Settings
    return create_app(Settings.from_env())

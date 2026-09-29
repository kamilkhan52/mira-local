"""Gateway settings, read from environment. See chatbot/.env.example."""
import ipaddress
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from mira.config import shared_llm_model

_DEFAULT_PROVENANCE = ("/Users/Eddie/Documents/n8n_memory_research_agent/"
                       "lightrag/working_dir_combined/provenance.json")
_DEFAULT_HYPOTHESES_DIR = "data/report-files/hypotheses"
_DEFAULT_MEMORY_DIR = "data/lightrag/working_dir"
_DEFAULT_OPTICAL_DIR = "data/lightrag/working_dir_optical"
_DEFAULT_STORAGE_DIR = "data/lightrag/working_dir_storage"
# Resolved from the shared `llm_models` block; the literals are the shipped
# fallbacks for a config that predates the optional exhaustive keys.
_DEFAULT_MAP_MODEL = shared_llm_model(
    "exhaustive_map", "google/gemini-2.5-flash")
_DEFAULT_SYNTHESIS_MODEL = shared_llm_model(
    "exhaustive_synthesis", "anthropic/claude-sonnet-5")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def is_hypothesis_token_valid(presented_token: str, configured_token: str) -> bool:
    """Return whether a presented hypothesis token matches a configured secret."""
    return bool(configured_token) and secrets.compare_digest(
        presented_token, configured_token)


def is_loopback_client(host: str | None) -> bool:
    """Whether a request's PEER ADDRESS is the local machine.

    A loopback caller already has local access, which is the same trust level as
    running hypothesize.py directly -- the token exists to stop *remote* peers
    spending LLM budget, and it still does for them.

    This must be called with the real socket peer (``request.client.host``),
    never a forwarded header: ``X-Forwarded-For`` is attacker-controlled and
    would let any remote peer claim to be local. Keying off the socket also
    cannot drift from reality the way a bind-address setting can, since the
    configured bind and the actual ``--host`` may disagree.
    """
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class Settings:
    bind: str = "127.0.0.1"
    port: int = 8090
    lightrag_url: str = "http://127.0.0.1:9623"
    lightrag_api_key: str = ""
    provenance_path: Path = field(default_factory=lambda: Path(_DEFAULT_PROVENANCE))
    rate_per_min: int = 10
    rate_concurrent: int = 3
    max_query_chars: int = 4000
    max_history_turns: int = 12
    # Aggregate cap across all history turns. Per-turn caps only bound user
    # turns (assistant replies may legitimately exceed max_query_chars), so this
    # bounds the total conversation_history forwarded upstream.
    max_history_chars: int = 50000
    # Shared secret gating the write-path /api/upload endpoint. Empty = uploads
    # disabled (fail closed): the endpoint is a graph mutation, so unlike the
    # read-only chat path it is not exposed on the tailnet gate alone.
    upload_token: str = ""
    # Separate secret gating the hypothesis-generation endpoint. Empty disables
    # that endpoint so it fails closed.
    hypothesis_token: str = ""
    hypothesis_timeout_sec: int = 1200
    hypotheses_dir: Path = field(
        default_factory=lambda: Path(_DEFAULT_HYPOTHESES_DIR))
    # Cap on decoded upload size; the whole file is read into memory and parsed.
    max_upload_bytes: int = 10 * 1024 * 1024
    exhaustive_enabled: bool = False
    research_token: str = ""
    exhaustive_memory_dir: Path = field(
        default_factory=lambda: Path(_DEFAULT_MEMORY_DIR))
    exhaustive_optical_dir: Path = field(
        default_factory=lambda: Path(_DEFAULT_OPTICAL_DIR))
    exhaustive_storage_dir: Path = field(
        default_factory=lambda: Path(_DEFAULT_STORAGE_DIR))
    exhaustive_node_threshold: float = 0.42
    exhaustive_edge_threshold: float = 0.38
    exhaustive_batch_tokens: int = 80_000
    exhaustive_queue_size: int = 20
    exhaustive_concurrency: int = 1
    exhaustive_job_timeout_sec: int = 3600
    exhaustive_map_model: str = _DEFAULT_MAP_MODEL
    exhaustive_synthesis_model: str = _DEFAULT_SYNTHESIS_MODEL
    # Preflight spend ceiling. A run whose estimate exceeds it is refused
    # before the first billable call, so a pathological query cannot spend an
    # unbounded amount while nobody is watching the job.
    max_estimated_cost_usd: float = 25.0
    # Ceilings on evidence fan-out. Exhaustive scoring has no result-count
    # limit by design, but the compiled corpus it feeds to the map stage is
    # billed per token, so the expansion is bounded highest-scored-first.
    exhaustive_max_papers: int = 400
    exhaustive_max_chunks_per_paper: int = 40

    def __post_init__(self) -> None:
        for name in (
            "exhaustive_node_threshold",
            "exhaustive_edge_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in (
            "exhaustive_batch_tokens",
            "exhaustive_queue_size",
            "exhaustive_concurrency",
            "exhaustive_job_timeout_sec",
            "exhaustive_max_papers",
            "exhaustive_max_chunks_per_paper",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_estimated_cost_usd <= 0:
            raise ValueError("max_estimated_cost_usd must be positive")

    @classmethod
    def from_env(cls, *, require_lightrag: bool = True) -> "Settings":
        """Read settings from the environment.

        `require_lightrag=False` is for callers that never reach the LightRAG
        upstream -- the exhaustive research path reads the graph directories
        directly, so demanding its API key would block a run that cannot use
        it.
        """
        key = os.environ.get("LIGHTRAG_API_KEY", "")
        if not key and require_lightrag:
            raise RuntimeError("LIGHTRAG_API_KEY is required")
        return cls(
            bind=os.environ.get("CHATBOT_BIND", "127.0.0.1"),
            port=int(os.environ.get("CHATBOT_PORT", "8090")),
            lightrag_url=os.environ.get("LIGHTRAG_URL", "http://127.0.0.1:9623"),
            lightrag_api_key=key,
            provenance_path=Path(os.environ.get("PROVENANCE_PATH",
                                                _DEFAULT_PROVENANCE)),
            rate_per_min=int(os.environ.get("RATE_PER_MIN", "10")),
            rate_concurrent=int(os.environ.get("RATE_CONCURRENT", "3")),
            max_query_chars=int(os.environ.get("MAX_QUERY_CHARS", "4000")),
            max_history_turns=int(os.environ.get("MAX_HISTORY_TURNS", "12")),
            max_history_chars=int(os.environ.get("MAX_HISTORY_CHARS", "50000")),
            upload_token=os.environ.get("UPLOAD_TOKEN", ""),
            hypothesis_token=os.environ.get("HYPOTHESIS_TOKEN", ""),
            hypothesis_timeout_sec=int(os.environ.get("HYPOTHESIS_TIMEOUT_SEC",
                                                      "1200")),
            hypotheses_dir=Path(os.environ.get("HYPOTHESES_DIR",
                                                _DEFAULT_HYPOTHESES_DIR)),
            max_upload_bytes=int(os.environ.get("MAX_UPLOAD_BYTES",
                                                str(10 * 1024 * 1024))),
            exhaustive_enabled=_env_bool("EXHAUSTIVE_RETRIEVAL_ENABLED"),
            research_token=os.environ.get("EXHAUSTIVE_RESEARCH_TOKEN", ""),
            exhaustive_memory_dir=Path(os.environ.get(
                "EXHAUSTIVE_MEMORY_DIR", _DEFAULT_MEMORY_DIR)),
            exhaustive_optical_dir=Path(os.environ.get(
                "EXHAUSTIVE_OPTICAL_DIR", _DEFAULT_OPTICAL_DIR)),
            exhaustive_storage_dir=Path(os.environ.get(
                "EXHAUSTIVE_STORAGE_DIR", _DEFAULT_STORAGE_DIR)),
            exhaustive_node_threshold=float(os.environ.get(
                "EXHAUSTIVE_NODE_THRESHOLD", "0.42")),
            exhaustive_edge_threshold=float(os.environ.get(
                "EXHAUSTIVE_EDGE_THRESHOLD", "0.38")),
            exhaustive_batch_tokens=int(os.environ.get(
                "EXHAUSTIVE_BATCH_TOKENS", "80000")),
            exhaustive_queue_size=int(os.environ.get(
                "EXHAUSTIVE_QUEUE_SIZE", "20")),
            exhaustive_concurrency=int(os.environ.get(
                "EXHAUSTIVE_CONCURRENCY", "1")),
            exhaustive_job_timeout_sec=int(os.environ.get(
                "EXHAUSTIVE_JOB_TIMEOUT_SEC", "3600")),
            exhaustive_map_model=os.environ.get(
                "EXHAUSTIVE_MAP_MODEL", _DEFAULT_MAP_MODEL),
            exhaustive_synthesis_model=os.environ.get(
                "EXHAUSTIVE_SYNTHESIS_MODEL", _DEFAULT_SYNTHESIS_MODEL),
            max_estimated_cost_usd=float(os.environ.get(
                "MIRA_MAX_ESTIMATED_COST_USD", "25.0")),
            exhaustive_max_papers=int(os.environ.get(
                "EXHAUSTIVE_MAX_PAPERS", "400")),
            exhaustive_max_chunks_per_paper=int(os.environ.get(
                "EXHAUSTIVE_MAX_CHUNKS_PER_PAPER", "40")),
        )

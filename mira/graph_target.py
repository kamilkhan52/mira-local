"""Which graph a hypothesis/discovery run reads and writes.

`hypothesize.py` and `discover.py` were hardcoded to the memory graph. Each now
takes `--graph {memory,optical}`, resolved here into the paths and URL that used
to be module-level constants.

Targets are fully isolated: separate working_dir, separate LightRAG server,
separate ledger, separate novelty cache. Nothing is shared, and no run reads
across graphs.

Spec: docs/superpowers/specs/2026-07-16-optical-graphrag-parity-design.md
"""
from __future__ import annotations
from mira.paths import REPORT_FILES, LOCAL_CACHE, LIGHTRAG_DIR

from dataclasses import dataclass
from pathlib import Path

from mira.config import ROOT


@dataclass(frozen=True)
class GraphTarget:
    name: str
    working_dir: Path
    base_url: str
    ledger_path: Path
    # A Path on every target, never None: callers test `venue_db.exists()` to
    # decide whether to degrade, and a None would raise there instead. A target
    # with no corpus yet points at a path that simply does not exist.
    venue_db: Path
    novelty_cache: Path
    venue_dbs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.venue_dbs:
            object.__setattr__(self, "venue_dbs", (self.venue_db,))

    @property
    def graphml(self) -> Path:
        return self.working_dir / "graph_chunk_entity_relation.graphml"

    @property
    def vdb_entities(self) -> Path:
        return self.working_dir / "vdb_entities.json"


_TARGETS = {
    "memory": GraphTarget(
        name="memory",
        working_dir=LIGHTRAG_DIR / "working_dir",
        base_url="http://localhost:9621",
        ledger_path=REPORT_FILES / "hypotheses" / "discovery-ledger.json",
        venue_db=LOCAL_CACHE / "venue_corpus.sqlite",
        novelty_cache=LOCAL_CACHE / "venue_novelty.json",
    ),
    # The optical venue corpus is deferred (spec §4), so venue_db points at a
    # path that does not exist yet — novelty degrades loudly through the
    # existing "venue corpus missing" branch until Eddie supplies the venue list.
    "optical": GraphTarget(
        name="optical",
        working_dir=LIGHTRAG_DIR / "working_dir_optical",
        base_url="http://localhost:9622",
        ledger_path=REPORT_FILES / "hypotheses" / "discovery-ledger-optical.json",
        venue_db=LOCAL_CACHE / "venue_corpus_optical.sqlite",
        novelty_cache=LOCAL_CACHE / "venue_novelty_optical.json",
    ),
    "storage": GraphTarget(
        name="storage",
        working_dir=LIGHTRAG_DIR / "working_dir_storage",
        base_url="http://localhost:9624",
        ledger_path=REPORT_FILES / "hypotheses" / "discovery-ledger-storage.json",
        venue_db=LOCAL_CACHE / "venue_corpus_storage.sqlite",
        novelty_cache=LOCAL_CACHE / "venue_novelty_storage.json",
    ),
    "combined": GraphTarget(
        name="combined",
        working_dir=LIGHTRAG_DIR / "working_dir_combined",
        base_url="http://localhost:9623",
        ledger_path=REPORT_FILES / "hypotheses" / "discovery-ledger-combined.json",
        venue_db=LOCAL_CACHE / "venue_corpus.sqlite",
        novelty_cache=LOCAL_CACHE / "venue_novelty_combined.json",
        venue_dbs=(
            LOCAL_CACHE / "venue_corpus.sqlite",
            LOCAL_CACHE / "venue_corpus_optical.sqlite",
        ),
    ),
}

GRAPH_NAMES = tuple(_TARGETS)
DEFAULT_GRAPH = "memory"


def resolve_target(name: str) -> GraphTarget:
    try:
        return _TARGETS[name]
    except KeyError:
        raise ValueError(
            f"unknown graph {name!r}; choose from {', '.join(GRAPH_NAMES)}"
        ) from None

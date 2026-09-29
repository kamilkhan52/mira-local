"""The hypothesis/discovery CLIs target one of several isolated graphs.

Two properties matter most here:

1. `memory` must reproduce exactly the constants the CLIs hardcoded before the
   seam existed — that is what makes every existing invocation regression-free.
2. Targets must not share mutable state. A shared novelty cache would serve one
   graph's corpus hit counts to another's pairs; a shared ledger would mix their
   pair statuses.
"""
from pathlib import Path

import pytest

from mira.config import ROOT
from mira.graph_target import DEFAULT_GRAPH, GRAPH_NAMES, resolve_target


def test_default_graph_is_memory():
    assert DEFAULT_GRAPH == "memory"


def test_memory_target_reproduces_the_original_constants():
    # These are the literal values hypothesize.py/discover.py held before the
    # seam. If this test fails, existing runs have silently changed behaviour.
    t = resolve_target("memory")
    assert t.working_dir == ROOT / "lightrag" / "working_dir"
    assert t.graphml == ROOT / "lightrag" / "working_dir" / "graph_chunk_entity_relation.graphml"
    assert t.vdb_entities == ROOT / "lightrag" / "working_dir" / "vdb_entities.json"
    assert t.base_url == "http://localhost:9621"
    assert t.ledger_path == ROOT / "report-files" / "hypotheses" / "discovery-ledger.json"
    assert t.venue_db == ROOT / "cache" / "venue_corpus.sqlite"
    assert t.novelty_cache == ROOT / "cache" / "venue_novelty.json"


def test_optical_target_resolves_to_its_own_graph():
    t = resolve_target("optical")
    assert t.working_dir == ROOT / "lightrag" / "working_dir_optical"
    assert t.graphml == ROOT / "lightrag" / "working_dir_optical" / "graph_chunk_entity_relation.graphml"
    assert t.base_url == "http://localhost:9622"


def test_storage_target_resolves_to_its_own_graph():
    t = resolve_target("storage")
    assert t.working_dir == ROOT / "lightrag" / "working_dir_storage"
    assert t.graphml == ROOT / "lightrag" / "working_dir_storage" / "graph_chunk_entity_relation.graphml"
    assert t.vdb_entities == ROOT / "lightrag" / "working_dir_storage" / "vdb_entities.json"
    assert t.base_url == "http://localhost:9624"
    assert t.ledger_path == ROOT / "report-files" / "hypotheses" / "discovery-ledger-storage.json"
    assert t.venue_db == ROOT / "cache" / "venue_corpus_storage.sqlite"
    assert t.novelty_cache == ROOT / "cache" / "venue_novelty_storage.json"


def test_combined_target_resolves_to_the_merged_graph_and_own_state():
    """The merged graph writes only to its own graph and discovery state."""
    t = resolve_target("combined")

    assert t.working_dir == ROOT / "lightrag" / "working_dir_combined"
    assert t.graphml == (
        ROOT / "lightrag" / "working_dir_combined" /
        "graph_chunk_entity_relation.graphml"
    )
    assert t.vdb_entities == (
        ROOT / "lightrag" / "working_dir_combined" / "vdb_entities.json"
    )
    assert t.base_url == "http://localhost:9623"
    assert t.ledger_path == (
        ROOT / "report-files" / "hypotheses" / "discovery-ledger-combined.json"
    )
    assert t.novelty_cache == ROOT / "cache" / "venue_novelty_combined.json"
    assert t.venue_db == resolve_target("memory").venue_db


def test_each_target_uses_the_venue_corpora_for_its_domain():
    memory = resolve_target("memory")
    optical = resolve_target("optical")
    storage = resolve_target("storage")
    combined = resolve_target("combined")

    assert memory.venue_dbs == (memory.venue_db,)
    assert optical.venue_dbs == (optical.venue_db,)
    assert storage.venue_dbs == (storage.venue_db,)
    assert combined.venue_dbs == (memory.venue_db, optical.venue_db)


def test_unknown_graph_names_error():
    with pytest.raises(ValueError, match="unknown graph"):
        resolve_target("nope")


@pytest.mark.parametrize("field", ["working_dir", "base_url", "ledger_path",
                                   "novelty_cache"])
def test_targets_share_no_state(field):
    values = {getattr(resolve_target(name), field) for name in GRAPH_NAMES}
    assert len(values) == len(GRAPH_NAMES)


def test_venue_db_is_always_a_path_even_when_absent():
    # hypothesize.py/discover.py call `target.venue_db.exists()` to decide
    # whether to degrade; a None here would raise AttributeError instead.
    #
    # This asserts the TYPE contract only. It deliberately does not assert that
    # the file is absent: that held only in a data-less worktree, and the
    # optical corpus does exist in the main checkout, so an absence assertion
    # fails there for reasons that have nothing to do with this code.
    t = resolve_target("optical")
    assert t.venue_db is not None
    assert isinstance(t.venue_db, Path)
    assert t.venue_db.exists() in (True, False)  # callable, never None


def test_storage_venue_db_is_a_deferred_path():
    t = resolve_target("storage")
    assert t.venue_db is not None
    assert t.venue_db.exists() is False


def test_every_named_target_resolves():
    for name in GRAPH_NAMES:
        assert resolve_target(name).name == name


def test_storage_compose_service_is_localhost_only_and_isolated():
    compose = (ROOT / "lightrag" / "docker-compose.lightrag.yml").read_text()
    assert "lightrag-storage:" in compose
    assert "127.0.0.1:9624:9621" in compose
    assert "./working_dir_storage:/app/working_dir" in compose
    assert "./.env.lightrag.storage" in compose

"""reconstruct_papers merges arXiv metadata over the per-paper caches.

Neither cache stage stores a title, so before this the fallback chain ended at
the bare versioned ID and every optical Paper node was named "2511.03432v1".
arXiv metadata supplies the title, and the authors that the Cached_Data CSV
does not cover for this graph.
"""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "backfill_graph", Path(__file__).parent.parent / "scripts" / "backfill_graph.py"
)
backfill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backfill)

_META = {
    "2601.1": {
        "title": "Ultrafast Topological Photonics",
        "authors": ["Wenfeng Zhou", "Lian Zhou"],
        "raw_id": "http://arxiv.org/abs/2601.1v1",
    }
}


def test_arxiv_title_names_the_paper_node():
    papers = backfill.reconstruct_papers(["2601.1v1"], {}, {}, arxiv_meta=_META)
    assert papers[0]["title"] == "Ultrafast Topological Photonics"


def test_without_metadata_title_still_falls_back_to_the_id():
    # Unchanged legacy behaviour — this is the bug that produced the current
    # optical graph, kept as the documented fallback rather than an exception.
    papers = backfill.reconstruct_papers(["2601.1v1"], {}, {})
    assert papers[0]["title"] == "2601.1v1"


def test_arxiv_title_wins_over_cache_title():
    cls = {"2601.1": {"title": "Stale Cached Title", "primary_topic": "X"}}
    papers = backfill.reconstruct_papers(["2601.1v1"], cls, {}, arxiv_meta=_META)
    assert papers[0]["title"] == "Ultrafast Topological Photonics"


def test_authors_become_author_affiliations_with_no_institutions():
    # graph_ingest._build_payload reads this shape to emit Author nodes and
    # authored_by edges. arXiv has no per-author affiliation, so the institution
    # list is empty and no affiliated_with edge is created.
    papers = backfill.reconstruct_papers(["2601.1v1"], {}, {}, arxiv_meta=_META)
    assert papers[0]["author_affiliations"] == {"Wenfeng Zhou": [], "Lian Zhou": []}


def test_paper_without_arxiv_authors_has_no_author_key():
    # An absent key, not an empty dict: _build_payload iterates it either way,
    # but this keeps "no author data" distinguishable from "no authors".
    papers = backfill.reconstruct_papers(["2601.9v1"], {}, {}, arxiv_meta=_META)
    assert "author_affiliations" not in papers[0]


def test_metadata_does_not_override_cached_topics():
    # arXiv supplies title/authors only; topics remain the classifier's job.
    cls = {"2601.1": {"primary_topic": "Silicon Photonics",
                      "secondary_topics": ["Optical Switching"]}}
    papers = backfill.reconstruct_papers(["2601.1v1"], cls, {}, arxiv_meta=_META)
    assert papers[0]["primary_topic"] == "Silicon Photonics"
    assert papers[0]["secondary_topics"] == ["Optical Switching"]


def test_affiliations_still_come_from_the_affiliation_cache():
    aff = {"2601.1": {"affiliations": ["MIT"], "credibility_tier": 9}}
    papers = backfill.reconstruct_papers(["2601.1v1"], {}, aff, arxiv_meta=_META)
    assert papers[0]["affiliations"] == ["MIT"]
    assert papers[0]["credibility_tier"] == 9


def test_metadata_is_matched_on_base_id_across_versions():
    # The graph holds versioned ids; metadata is keyed by base id.
    papers = backfill.reconstruct_papers(["2601.1v3"], {}, {}, arxiv_meta=_META)
    assert papers[0]["title"] == "Ultrafast Topological Photonics"

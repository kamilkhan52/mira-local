"""arXiv metadata backfill: parsing, author filtering, and coverage reporting.

The network is stubbed via the injectable `fetch` seam, mirroring
mira/hypothesis/novelty.py's.
"""
import requests

from scripts.backfill_arxiv_metadata import (
    COVERAGE_FLOOR,
    base_id,
    fetch_metadata,
    is_valid_author,
)


def _paper(arxiv_id, title="A Title", authors=("Alice Smith",), summary="An abstract."):
    return {"id": arxiv_id, "title": title, "authors": list(authors), "summary": summary,
            "raw_id": f"http://arxiv.org/abs/{arxiv_id}"}


def _quiet(_msg):
    pass


def test_base_id_strips_version():
    assert base_id("2511.03432v1") == "2511.03432"
    assert base_id("2511.03432v12") == "2511.03432"
    assert base_id("2511.03432") == "2511.03432"


def test_fetch_metadata_maps_titles_and_authors():
    meta = fetch_metadata(
        ["2511.03432v1"],
        fetch=lambda ids: [_paper("2511.03432v1", "Topological Photonics", ["Wenfeng Zhou", "Lian Zhou"])],
        delay=0, log=_quiet,
    )
    assert meta["2511.03432"]["title"] == "Topological Photonics"
    assert meta["2511.03432"]["authors"] == ["Wenfeng Zhou", "Lian Zhou"]


def test_summary_is_captured_for_the_classifier():
    # The classification stage prompts on paper["summary"]; without it the 421
    # uncached papers cannot be reclassified.
    meta = fetch_metadata(
        ["2511.03432v1"],
        fetch=lambda ids: [_paper("2511.03432v1", summary="We demonstrate a photonic switch.")],
        delay=0, log=_quiet,
    )
    assert meta["2511.03432"]["summary"] == "We demonstrate a photonic switch."


def test_response_version_may_differ_from_request():
    # arXiv resolves an unversioned id to the latest version, so the response
    # can carry a version the request did not. Matching is on base id, so this
    # must count as resolved rather than as a miss.
    meta = fetch_metadata(
        ["2604.18496"],
        fetch=lambda ids: [_paper("2604.18496v2")],
        delay=0, log=_quiet,
    )
    assert "2604.18496" in meta
    assert meta["2604.18496"]["arxiv_id"] == "2604.18496v2"


def test_junk_author_names_are_filtered_but_recorded():
    # "Osama0020Yousuf" is escape residue; the rest are real people.
    meta = fetch_metadata(
        ["2511.00001v1"],
        fetch=lambda ids: [_paper("2511.00001v1", authors=["Osama0020Yousuf", "Alice Smith"])],
        delay=0, log=_quiet,
    )
    entry = meta["2511.00001"]
    assert entry["authors"] == ["Alice Smith"]
    assert entry["rejected_authors"] == ["Osama0020Yousuf"]


def test_placeholder_author_is_filtered():
    meta = fetch_metadata(
        ["2511.00002v1"],
        fetch=lambda ids: [_paper("2511.00002v1", authors=["Corresponding Author", "Bob Jones"])],
        delay=0, log=_quiet,
    )
    assert meta["2511.00002"]["authors"] == ["Bob Jones"]


def test_metadata_author_filter_is_conservative_and_syntax_only():
    assert is_valid_author("Alice Smith")
    assert is_valid_author("José O'Connor III")
    assert not is_valid_author("Osama0020Yousuf")
    assert not is_valid_author("Corresponding Author")
    assert not is_valid_author("Unknown")


def test_unresolved_ids_are_absent_not_invented():
    # A withdrawn paper simply is not in the response. It must not appear with
    # an empty title, which would silently look like a resolved paper.
    meta = fetch_metadata(
        ["2511.00001v1", "2511.99999v1"],
        fetch=lambda ids: [_paper("2511.00001v1")],
        delay=0, log=_quiet,
    )
    assert set(meta) == {"2511.00001"}


def test_a_failed_batch_is_reported_and_does_not_abort_the_run():
    logs = []

    def boom(ids):
        raise requests.RequestException("503")

    meta = fetch_metadata(["2511.00001v1"], fetch=boom, delay=0, log=logs.append)
    assert meta == {}
    assert any("WARNING" in m for m in logs)


def test_batching_covers_every_id_across_multiple_requests():
    ids = [f"2511.{i:05d}v1" for i in range(120)]
    seen = []

    def fake(batch):
        seen.append(len(batch))
        return [_paper(i) for i in batch]

    meta = fetch_metadata(ids, fetch=fake, delay=0, log=_quiet)
    assert len(meta) == 120
    assert sum(seen) == 120
    assert len(seen) == 3  # 50 + 50 + 20


def test_coverage_floor_is_a_real_threshold():
    assert 0 < COVERAGE_FLOOR <= 1


def test_raw_id_is_stored_verbatim_not_reconstructed():
    # The classification cache fingerprint hashes this exact string. Rebuilding
    # it as "https://arxiv.org/abs/<id>" makes every lookup miss and re-bills
    # the LLM for papers already classified — silently, since a miss just looks
    # like work to do.
    meta = fetch_metadata(
        ["2604.14521v1"],
        fetch=lambda ids: [_paper("2604.14521v1")],
        delay=0, log=_quiet,
    )
    assert meta["2604.14521"]["raw_id"] == "http://arxiv.org/abs/2604.14521v1"


def test_raw_id_survives_a_version_mismatch():
    meta = fetch_metadata(
        ["2604.18496"],
        fetch=lambda ids: [_paper("2604.18496v2")],
        delay=0, log=_quiet,
    )
    assert meta["2604.18496"]["raw_id"] == "http://arxiv.org/abs/2604.18496v2"

"""Coverage is the only way to answer "does this graph have full text".

The subtlety worth testing: a Paper node is covered when a *full-text document*
exists for its arXiv id, not merely when some chunk mentions it. Bulk metadata
chunks carry a `{profile}-{date}-bulk` source_id and must never count as
coverage — mistaking them for full text is exactly what made the storage graph
look partially populated when it had none at all.
"""
import json
from pathlib import Path

import pytest

from mira.fulltext_coverage import (
    Coverage, coverage, fulltext_ids, paper_arxiv_ids, recover_arxiv_id,
)

GRAPHML = """<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d1" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d2" for="node" attr.name="description" attr.type="string"/>
  <graph edgedefault="undirected">
    <node id="Covered Paper">
      <data key="d1">Paper</data>
      <data key="d2">https://arxiv.org/abs/2511.06249v1 · a covered paper</data>
    </node>
    <node id="Bulk Only Paper">
      <data key="d1">Paper</data>
      <data key="d2">https://arxiv.org/abs/2511.06010v1 · metadata only</data>
    </node>
    <node id="Untraceable Paper">
      <data key="d1">Paper</data>
      <data key="d2">no url here</data>
    </node>
    <node id="Some Author">
      <data key="d1">Person</data>
      <data key="d2">https://arxiv.org/abs/2511.99999v1</data>
    </node>
  </graph>
</graphml>
"""

CHUNKS = {
    "chunk-a": {"source_id": "fulltext-2511.06249v1", "tokens": 1200},
    "chunk-b": {"source_id": "fulltext-2511.06249v1", "tokens": 900},
    "chunk-c": {"source_id": "storage-innovation-2026-07-20-bulk", "tokens": 240},
}


@pytest.fixture
def wd(tmp_path):
    (tmp_path / "graph_chunk_entity_relation.graphml").write_text(GRAPHML)
    (tmp_path / "kv_store_text_chunks.json").write_text(json.dumps(CHUNKS))
    return tmp_path


def test_recover_arxiv_id_from_stub_name():
    assert recover_arxiv_id("arxiv:2511.06249v1", "") == "2511.06249v1"


def test_recover_arxiv_id_from_description_url():
    assert recover_arxiv_id(
        "Some Title", "https://arxiv.org/abs/2511.06010v1 · blah") == "2511.06010v1"


def test_recover_arxiv_id_returns_empty_when_absent():
    assert recover_arxiv_id("Some Title", "no url here") == ""


def test_paper_arxiv_ids_covers_only_paper_nodes(wd):
    ids = paper_arxiv_ids(wd / "graph_chunk_entity_relation.graphml")
    assert set(ids) == {"Covered Paper", "Bulk Only Paper", "Untraceable Paper"}
    assert ids["Covered Paper"] == "2511.06249v1"
    assert ids["Untraceable Paper"] == ""


def test_fulltext_ids_ignores_bulk_metadata_chunks(wd):
    assert fulltext_ids(wd / "kv_store_text_chunks.json") == {"2511.06249v1"}


def test_coverage_counts_only_fulltext_backed_papers(wd):
    c = coverage(wd)
    assert isinstance(c, Coverage)
    assert c.total == 3
    assert c.covered == 1
    assert c.missing == ["Bulk Only Paper"]
    assert c.no_id == ["Untraceable Paper"]


def test_coverage_pct_is_share_of_all_paper_nodes(wd):
    assert coverage(wd).pct == pytest.approx(100 / 3)


def test_coverage_pct_is_zero_when_there_are_no_papers(tmp_path):
    (tmp_path / "graph_chunk_entity_relation.graphml").write_text(
        "<?xml version='1.0' encoding='utf-8'?>"
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        '<graph edgedefault="undirected"></graph></graphml>')
    (tmp_path / "kv_store_text_chunks.json").write_text("{}")
    assert coverage(tmp_path).pct == 0.0

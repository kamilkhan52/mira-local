import json, lzma, csv as csv_mod, io
from pathlib import Path
import pytest
from repair_author_edges import (
    papers_missing_authored_by, title_to_arxiv_key,
    cache_author_lookup, resolve_authors,
)

MINIMAL_GRAPHML = """\
<?xml version='1.0' encoding='utf-8'?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
<key id="d0" for="node" attr.name="entity_type" attr.type="string"/>
<key id="d1" for="edge" attr.name="keywords" attr.type="string"/>
<graph id="G" edgedefault="directed">
  <node id="Paper One"><data key="d0">Paper</data></node>
  <node id="Paper Two"><data key="d0">Paper</data></node>
  <node id="Alice Smith"><data key="d0">Author</data></node>
  <edge id="e1" source="Paper One" target="Alice Smith">
    <data key="d1">authored_by author</data>
  </edge>
</graph>
</graphml>"""

def test_papers_missing_authored_by(tmp_path):
    p = tmp_path / "graph.graphml"
    p.write_text(MINIMAL_GRAPHML)
    missing = papers_missing_authored_by(p)
    assert "Paper Two" in missing
    assert "Paper One" not in missing
    assert "Alice Smith" not in missing  # Author, not Paper

def test_papers_missing_authored_by_all_linked(tmp_path):
    graphml = MINIMAL_GRAPHML.replace(
        '<node id="Paper Two"><data key="d0">Paper</data></node>',
        '<node id="Paper Two"><data key="d0">Paper</data></node>\n'
        '  <edge id="e2" source="Paper Two" target="Alice Smith">'
        '<data key="d1">authored_by author</data></edge>'
    )
    p = tmp_path / "graph.graphml"
    p.write_text(graphml)
    assert papers_missing_authored_by(p) == set()

def test_title_to_arxiv_key(tmp_path):
    data = [{"body_markdown": "[Great Paper](https://arxiv.org/abs/2604.05285v1) — MIT"}]
    (tmp_path / "r.json").write_text(json.dumps(data))
    m = title_to_arxiv_key(tmp_path)
    assert m["Great Paper"] == "2604.05285v1"

def test_title_to_arxiv_key_ignores_bad_json(tmp_path):
    (tmp_path / "bad.json").write_text("not json")
    assert title_to_arxiv_key(tmp_path) == {}

def test_cache_author_lookup(tmp_path):
    data = {"2604.05285": {"author_affiliations": {"Alice": ["MIT"]}}}
    p = tmp_path / "affiliations.json"
    p.write_text(json.dumps(data))
    m = cache_author_lookup(p)
    assert m["2604.05285"] == {"Alice": ["MIT"]}

def test_cache_author_lookup_missing_file(tmp_path):
    assert cache_author_lookup(tmp_path / "missing.json") == {}

def test_resolve_authors_arxiv_key_from_csv():
    csv_map = {"2604.05285v1": {"Alice": ["MIT"]}}
    result = resolve_authors("arxiv:2604.05285v1", {}, csv_map, {})
    assert result == {"Alice": ["MIT"]}

def test_resolve_authors_title_from_csv():
    title_map = {"Great Paper": "2604.05285v1"}
    csv_map = {"2604.05285v1": {"Alice": ["MIT"]}}
    result = resolve_authors("Great Paper", title_map, csv_map, {})
    assert result == {"Alice": ["MIT"]}

def test_resolve_authors_fallback_to_cache():
    cache_map = {"2604.05285": {"Bob": ["Stanford"]}}
    result = resolve_authors("arxiv:2604.05285v1", {}, {}, cache_map)
    assert result == {"Bob": ["Stanford"]}

def test_resolve_authors_not_found():
    assert resolve_authors("arxiv:9999.99999v1", {}, {}, {}) == {}

def test_resolve_authors_unknown_title():
    assert resolve_authors("Some Unknown Title", {}, {}, {}) == {}

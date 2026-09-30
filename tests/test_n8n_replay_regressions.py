"""Regressions found by replaying real n8n execution 2085 (tools/n8n_replay.py)."""
import json

from _paths import repoint


def test_arxiv_parse_keeps_internal_newlines_like_n8n():
    from mira.fetch import _parse_xml
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2609.32197v1</id><title>  A Title  </title>
<summary>  First paragraph.
  Second paragraph.  </summary><published>2026-09-26T03:49:52Z</published>
<author><name>A</name></author></entry></feed>"""
    p = _parse_xml(xml)[0]
    assert p["title"] == "A Title"
    assert p["summary"] == "First paragraph.\n  Second paragraph."


def test_affiliation_prompt_renders_authors_and_categories_as_json_arrays(monkeypatch, tmp_path):
    import mira.pipeline as mp
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    seen = {}

    def fake_llm(client, model, system, user, **kw):
        seen["user"] = user
        return json.dumps({"affiliations": [], "author_affiliations": {}, "credibility_tier": 5,
                           "credibility_reasoning": "x"})
    monkeypatch.setattr("mira.config.llm_call", fake_llm)
    config = {"profile_id": "memory-innovation", "topic": {"focus": "memory"},
              "llm_models": {"affiliation": "m"},
              "prompts": {"affiliation": {"system": "", "user": "authors: {{authors}}\ncategory: {{category}}"}}}
    paper = {"id": "2609.1", "raw_id": "http://arxiv.org/abs/2609.1v1", "title": "t", "summary": "s",
             "authors": ["Desen Sun", "Valérie Castin"], "categories": ["cs.DC", "cs.LG"],
             "first_page_text": "x" * 60}
    mp._get_affiliation(paper, config, None, {})
    assert seen["user"] == 'authors: ["Desen Sun","Valérie Castin"]\ncategory: ["cs.DC","cs.LG"]'


def test_classify_ignores_legacy_id_keyed_cache(monkeypatch, tmp_path):
    """The old CLI's cache/classifications.json (keyed by bare id only) must
    not leak another profile's result into this run."""
    import mira.pipeline as mp
    repoint(monkeypatch, "mira.pipeline", tmp_path)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "classifications.json").write_text(json.dumps(
        {"2609.1": {"relevance_score": 3, "primary_topic": "Other CXL-related systems"}}))
    calls = []

    def fake_llm(client, model, system, user, **kw):
        calls.append(user)
        if "credibility" in user:
            return json.dumps({"affiliations": [], "author_affiliations": {}, "credibility_tier": 8,
                               "credibility_reasoning": "x"})
        return json.dumps({"relevance_score": 8, "primary_topic": "HBM"})
    monkeypatch.setattr("mira.config.llm_call", fake_llm)
    monkeypatch.setenv("MIRA_LLM_CACHE_BYPASS", "1")
    config = {"profile_id": "memory-innovation", "topic": {"focus": "memory"},
              "llm_models": {"affiliation": "m", "classification": "m"},
              "prompts": {"affiliation": {"system": "", "user": "credibility {{title}}"},
                          "classification": {"system": "", "user": "classify {{title}}"}}}
    paper = {"id": "2609.1", "raw_id": "http://arxiv.org/abs/2609.1v1", "title": "t", "summary": "s",
             "authors": [], "categories": [], "first_page_text": "x" * 60}
    out = mp.classify_papers([paper], config, None)
    assert out[0]["relevance_score"] == 8 and out[0]["primary_topic"] == "HBM"


def test_structured_llm_call_uses_n8n_tool_protocol():
    """Structured stages answer via n8n's format_final_json_response tool, with
    n8n's instruction appended to the system message (execution-verified)."""
    import json
    from types import SimpleNamespace
    from mira.config import llm_call
    from mira import structured
    sent = {}

    class Completions:
        def create(self, **kw):
            sent.update(kw)
            call = SimpleNamespace(function=SimpleNamespace(
                arguments=json.dumps({"output": {"selected_indices": [0, 2]}})))
            msg = SimpleNamespace(content=None, tool_calls=[call])
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    out = llm_call(client, "m", "Pick articles.\n", "articles...", schema="media_selection")
    assert json.loads(out) == {"selected_indices": [0, 2]}
    assert sent["messages"][0]["content"] == "Pick articles.\n\n\n" + structured.SYSTEM_SUFFIX
    tool = sent["tools"][0]["function"]
    assert tool["name"] == "format_final_json_response"
    assert tool["parameters"]["properties"]["output"] == structured.SCHEMAS["media_selection"]
    llm_call(client, "m", "", "x", schema="trend")
    assert sent["messages"][0]["content"] == structured.SYSTEM_SUFFIX  # empty system: instruction only

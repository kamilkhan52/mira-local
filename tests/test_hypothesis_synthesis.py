from types import SimpleNamespace

from mira.hypothesis.gaps import GapCandidate
from mira.hypothesis.synthesis import (
    apply_critic,
    build_synthesis_prompt,
    parse_critic_verdict,
)


def _client_replying(text: str):
    """Minimal OpenAI-shaped stub: every completion returns `text`."""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: resp)))


def _cand() -> GapCandidate:
    c = GapCandidate("CXL", "PIM", ["HBM4"], ["Samsung"], ["Kim"], 2.0, 10.0)
    c.novelty_hits = 1
    c.novelty_label = "open gap"
    c.external_titles = ["Some adjacent external paper"]
    return c


def test_synthesis_prompt_contains_gap_facts_and_evidence():
    prompt = build_synthesis_prompt(_cand(), "EVIDENCE-SIDE-A", "EVIDENCE-SIDE-B")
    assert "CXL" in prompt and "PIM" in prompt
    assert "Samsung" in prompt                     # gap facts included
    assert "EVIDENCE-SIDE-A" in prompt and "EVIDENCE-SIDE-B" in prompt
    assert "Some adjacent external paper" in prompt  # novelty findings included
    # Required output structure is spelled out for the model
    for heading in ("Claim", "Supporting evidence", "Why plausibly unexplored",
                    "Suggested validation"):
        assert heading in prompt


def test_synthesis_prompt_marks_missing_retrieval():
    prompt = build_synthesis_prompt(_cand(), None, None)
    assert "retrieval unavailable" in prompt.lower()


def test_unverified_novelty_prompt_names_venue_corpus_not_semantic_scholar():
    c = _cand()
    c.novelty_hits = None
    c.external_titles = []
    prompt = build_synthesis_prompt(c, "A", "B")
    assert "Semantic Scholar" not in prompt      # S2 was replaced by the venue corpus
    assert "venue corpus" in prompt
    assert "unverified" in prompt


def test_parse_critic_verdict():
    v, body = parse_critic_verdict("VERDICT: STRENGTHEN\nRevised hypothesis...")
    assert v == "strengthen" and body == "Revised hypothesis..."
    v, body = parse_critic_verdict("VERDICT: KILL\nThe claim contradicts P3.")
    assert v == "kill"
    v, _ = parse_critic_verdict("I think it's fine.")
    assert v == "unknown"


def test_parse_critic_verdict_accept():
    v, body = parse_critic_verdict("VERDICT: ACCEPT\nClaims match the evidence.")
    assert v == "accept" and body == "Claims match the evidence."


def test_apply_critic_accept_keeps_text_unlabelled():
    c = _cand()
    verdict, text = apply_critic(
        _client_replying("VERDICT: ACCEPT\nAll claims check out."),
        c, "original text", None, None)
    assert verdict == "accept" and text == "original text"
    assert "killed" not in c.novelty_label and c.critic_objection is None


def test_critic_system_offers_three_verdicts_and_a_kill_bar():
    from mira.hypothesis.synthesis import CRITIC_SYSTEM
    for verdict in ("VERDICT: ACCEPT", "VERDICT: STRENGTHEN", "VERDICT: KILL"):
        assert verdict in CRITIC_SYSTEM
    # Kill must be reserved for decisive, evidence-based defects — the
    # hypothesis is speculative by design, so skepticism alone can't kill it.
    assert "NOT" in CRITIC_SYSTEM and "speculative" in CRITIC_SYSTEM


def test_apply_critic_strengthen_replaces_text():
    c = _cand()
    verdict, text = apply_critic(
        _client_replying("VERDICT: STRENGTHEN\nRevised hypothesis."),
        c, "original text", None, None)
    assert verdict == "strengthen" and text == "Revised hypothesis."
    assert c.critic_objection is None


def test_apply_critic_strengthen_strips_reviewer_preamble():
    reply = ("VERDICT: STRENGTHEN\nThe core idea is sound but one claim "
             "overreaches.\n\n---\n\n**Revised Hypothesis**\n\n"
             "**Claim** — the corrected claim.\n**Supporting evidence** — ...")
    verdict, text = apply_critic(_client_replying(reply), _cand(),
                                 "original text", None, None)
    assert verdict == "strengthen"
    assert text.startswith("**Claim**")
    assert "Revised Hypothesis" not in text


def test_apply_critic_kill_labels_candidate_and_keeps_text():
    c = _cand()
    verdict, text = apply_critic(
        _client_replying("VERDICT: KILL\nDecisive objection."),
        c, "original text", None, None)
    assert verdict == "kill" and text == "original text"
    assert c.novelty_label.endswith("killed by critic")
    assert c.critic_objection == "Decisive objection."


def test_apply_critic_unknown_verdict_keeps_text_unlabelled():
    c = _cand()
    verdict, text = apply_critic(
        _client_replying("Looks fine to me."), c, "original text", None, None)
    assert verdict == "unknown" and text == "original text"
    assert "killed" not in c.novelty_label and c.critic_objection is None


def test_retrieve_context_returns_none_on_non_dict_json(monkeypatch):
    from mira.hypothesis import synthesis

    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return ["not", "a", "dict"]

    monkeypatch.setattr(synthesis.requests, "post",
                        lambda *a, **kw: FakeResponse())
    assert synthesis.retrieve_context("q", base_url="http://x") is None


def test_retrieve_context_sends_api_key_header_when_configured(monkeypatch):
    """LightRAG rejects unauthenticated /query with 'API Key required'. Without
    this header every run silently degrades grounding to graph metadata."""
    from mira.hypothesis import synthesis

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": "context"}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(synthesis.requests, "post", fake_post)
    monkeypatch.setenv("LIGHTRAG_API_KEY", "secret-key")

    assert synthesis.retrieve_context("q", base_url="http://x") == "context"
    assert captured.get("headers", {}).get("X-API-Key") == "secret-key"


def test_retrieve_context_omits_api_key_header_when_unset(monkeypatch):
    """Unauthenticated instances must keep working unchanged."""
    from mira.hypothesis import synthesis

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"response": "context"}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(synthesis.requests, "post", fake_post)
    monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)

    assert synthesis.retrieve_context("q", base_url="http://x") == "context"
    assert "X-API-Key" not in (captured.get("headers") or {})

import json
from types import MappingProxyType, SimpleNamespace

import networkx as nx
import pytest

from mira.exhaustive.compiler import (
    CitationRecord,
    CompiledEvidence,
    EvidenceRecord,
    ModelResponse,
)
from mira.exhaustive.costs import CostLedger
from mira.exhaustive.engine import PreparedResearch
from mira.exhaustive.types import (
    DOMAIN_NAMES,
    ResearchEvidenceBundle,
    UsageRecord,
)
from mira.hypothesis.exhaustive import ExhaustiveHypothesisRunner
from tests.test_exhaustive_compiler import prompt_payload


def _domain_snapshot(domain):
    graph = nx.Graph()
    report = f"{domain}-profile-2026-07-01"
    topic_a = "HBM" if domain == "memory" else f"{domain} bridge"
    topic_c = "Silicon Photonics" if domain == "optical" else f"{domain} edge"
    paper_a = f"{domain} paper A"
    paper_c = f"{domain} paper C"
    graph.add_node(report, entity_type="Report")
    for paper, topic in ((paper_a, topic_a), (paper_c, topic_c)):
        graph.add_node(paper, entity_type="Paper")
        graph.add_node(topic, entity_type="Topic")
        graph.add_edge(report, paper, keywords="selected_in report")
        graph.add_edge(paper, topic, keywords="primary_topic topic")
    graph.add_node("Shared Lab", entity_type="Institution")
    graph.add_edge(
        "Shared Lab", topic_a, keywords="researches topic"
    )
    graph.add_edge(
        "Shared Lab", topic_c, keywords="researches topic"
    )
    graph.add_node(f"{domain} common", entity_type="Topic")
    graph.add_edge(
        f"{domain} common", topic_a, keywords="related_to topic"
    )
    graph.add_edge(
        f"{domain} common", topic_c, keywords="related_to topic"
    )
    names = tuple(graph.nodes)
    return SimpleNamespace(
        domain=domain,
        graph=graph,
        entity_names=names,
        entity_matrix=__import__("numpy").eye(len(names), dtype="float32"),
    )


def _prepared():
    bundle = ResearchEvidenceBundle.empty(
        "Generate hypotheses connecting HBM and Silicon Photonics"
    )
    bundle = ResearchEvidenceBundle(
        query=bundle.query,
        snapshot_fingerprints=MappingProxyType({}),
        nodes_scanned=MappingProxyType({
            "memory": 8, "optical": 8, "storage": 8
        }),
        edges_scanned=MappingProxyType({
            "memory": 8, "optical": 8, "storage": 8
        }),
        selected_regions=(),
        paper_names=tuple(
            f"{domain} paper {suffix}"
            for domain in DOMAIN_NAMES
            for suffix in ("A", "C")
        ),
        chunks=(),
        threshold_version="test",
        exhaustive=True,
    )
    estimate = SimpleNamespace(
        subtotal_usd=0.1,
        total_with_reserve_usd=0.11,
        reserve=0.1,
        by_stage={},
    )
    return PreparedResearch(
        query=bundle.query,
        history=(),
        bundle=bundle,
        estimated_cost=estimate,
        snapshots=MappingProxyType({
            domain: _domain_snapshot(domain) for domain in DOMAIN_NAMES
        }),
    )


def _compiled():
    evidence_citation = CitationRecord(
        chunk_id="chunk-1",
        file_path="paper.md",
        title="Evidence Paper",
        domains=("memory", "optical"),
    )
    paper_citations = tuple(
        CitationRecord(
            chunk_id=f"chunk-{domain}-{suffix.lower()}",
            file_path=f"{domain}-{suffix.lower()}.md",
            title=f"{domain} paper {suffix}",
            domains=(domain,),
        )
        for domain in DOMAIN_NAMES
        for suffix in ("A", "C")
    )
    citations = (evidence_citation, *paper_citations)
    record = EvidenceRecord(
        claim="HBM and photonics share packaging constraints.",
        measurements=("10 ns",),
        mechanisms=("co-packaging",),
        assumptions=(),
        limitations=(),
        contradictions=(),
        citations=citations,
        relevance="direct",
        contributing_chunk_ids=("chunk-1",),
    )
    return CompiledEvidence(
        query="Generate hypotheses",
        records=(record,),
        contributing_chunk_ids=("chunk-1",),
        measurements=("10 ns",),
        contradictions=(),
        citations=citations,
    )


class HypothesisModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        request = prompt_payload(kwargs["user"])
        hypotheses = [{
            "topic_a": item["topic_a"],
            "topic_c": item["topic_c"],
            "text": (
                "**Claim** — measurable claim.\n"
                "**Supporting evidence** — Evidence Paper.\n"
                "**Why plausibly unexplored** — structural gap.\n"
                "**Suggested validation** — controlled experiment."
            ),
        } for item in request["candidates"]]
        return ModelResponse(
            text=json.dumps({"hypotheses": hypotheses}),
            usage=UsageRecord(
                stage=kwargs["stage"],
                model=kwargs["model"],
                input_tokens=100,
                output_tokens=50,
                total_cost=0.01,
            ),
        )


class _RateLimitError(RuntimeError):
    """Stands in for a provider SDK rate-limit exception."""

    status_code = 429


def _request(**overrides):
    request = {
        "topics": ["HBM", "Silicon Photonics"],
        "profiles": [
            "memory-profile", "optical-profile", "storage-profile"
        ],
        "max_hypotheses": 5,
        "critic": False,
        "no_external": True,
    }
    request.update(overrides)
    return request


def test_runner_uses_all_three_snapshots_and_one_shared_sonnet_call():
    model = HypothesisModel()
    ledger = CostLedger()
    runner = ExhaustiveHypothesisRunner(model, ledger)

    output = runner(
        _prepared(),
        _compiled(),
        _request(),
        lambda _event: None,
    )

    assert "# Hypothesis Dossier" in output.markdown
    assert output.candidates_evaluated >= output.hypotheses_generated
    assert output.hypotheses_generated > 0
    assert len(model.calls) == 1
    assert model.calls[0]["stage"] == "hypothesis_synthesis"
    assert ledger.summary().actual_cost_usd == pytest.approx(0.01)
    assert "Evidence Paper" in model.calls[0]["user"]
    synthesis = json.loads(model.calls[0]["user"])
    assert any(
        candidate["domains_a"] and candidate["domains_c"]
        for candidate in synthesis["candidates"]
    )


def test_runner_rejects_candidate_without_compiled_citations_for_both_sides():
    model = HypothesisModel()
    compiled = _compiled()
    memory_only = tuple(
        citation
        for citation in compiled.citations
        if citation.title in {"Evidence Paper", "memory paper A"}
    )
    compiled = CompiledEvidence(
        query=compiled.query,
        records=compiled.records,
        contributing_chunk_ids=compiled.contributing_chunk_ids,
        measurements=compiled.measurements,
        contradictions=compiled.contradictions,
        citations=memory_only,
    )

    output = ExhaustiveHypothesisRunner(model, CostLedger())(
        _prepared(),
        compiled,
        _request(),
        lambda _event: None,
    )

    assert output.hypotheses_generated == 0
    assert model.calls == []


def test_runner_rejects_incomplete_coverage_before_model_call():
    model = HypothesisModel()
    runner = ExhaustiveHypothesisRunner(model, CostLedger())
    prepared = _prepared()
    incomplete = PreparedResearch(
        query=prepared.query,
        history=(),
        bundle=ResearchEvidenceBundle.empty(prepared.query),
        estimated_cost=prepared.estimated_cost,
        snapshots=prepared.snapshots,
    )

    with pytest.raises(ValueError, match="exhaustive"):
        runner(
            incomplete,
            _compiled(),
            {
                "topics": ["HBM", "Silicon Photonics"],
                "profiles": ["memory-profile", "optical-profile"],
                "max_hypotheses": 5,
            },
            lambda _event: None,
        )

    assert model.calls == []


HYPOTHESIS_TEXT = (
    "**Claim** — measurable claim.\n"
    "**Supporting evidence** — Evidence Paper.\n"
    "**Why plausibly unexplored** — structural gap.\n"
    "**Suggested validation** — controlled experiment."
)


class CriticModel(HypothesisModel):
    """Drafts one blank hypothesis, then reviews only what was drafted."""

    def __init__(self, verdict="strengthen", objections=("unsupported claim",)):
        super().__init__()
        self.verdict = verdict
        self.objections = list(objections)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        request = prompt_payload(kwargs["user"])
        if kwargs["stage"] == "hypothesis_critic":
            body = {"reviews": [{
                "topic_a": item["topic_a"],
                "topic_c": item["topic_c"],
                "verdict": self.verdict,
                "text": HYPOTHESIS_TEXT + "\nRevised.",
                "objections": self.objections,
            } for item in request["candidates"]]}
        else:
            body = {"hypotheses": [{
                "topic_a": item["topic_a"],
                "topic_c": item["topic_c"],
                "text": HYPOTHESIS_TEXT if index == 0 else "",
            } for index, item in enumerate(request["candidates"])]}
        return ModelResponse(
            text=json.dumps(body),
            usage=UsageRecord(
                stage=kwargs["stage"],
                model=kwargs["model"],
                input_tokens=100,
                output_tokens=50,
                total_cost=0.01,
            ),
        )


def test_critic_skips_blank_drafts_instead_of_looking_them_up():
    model = CriticModel()
    output = ExhaustiveHypothesisRunner(model, CostLedger())(
        _prepared(),
        _compiled(),
        _request(max_hypotheses=3, critic=True),
        lambda _event: None,
    )

    assert output.hypotheses_generated == 1
    assert [call["stage"] for call in model.calls] == [
        "hypothesis_synthesis",
        "hypothesis_critic",
    ]
    critic_candidates = json.loads(model.calls[1]["user"])["candidates"]
    assert len(critic_candidates) == 1
    assert critic_candidates[0]["draft"].startswith("**Claim**")


def test_critic_verdict_and_objections_reach_the_dossier():
    model = CriticModel(objections=["cites a paper that is not in evidence"])

    output = ExhaustiveHypothesisRunner(model, CostLedger())(
        _prepared(),
        _compiled(),
        _request(max_hypotheses=3, critic=True),
        lambda _event: None,
    )

    assert "**Critic verdict:** strengthen" in output.markdown
    assert "cites a paper that is not in evidence" in output.markdown
    assert "Revised." in output.markdown


def test_killed_hypothesis_records_the_objection_it_was_killed_for():
    model = CriticModel(verdict="kill", objections=["contradicted by 10 ns"])

    output = ExhaustiveHypothesisRunner(model, CostLedger())(
        _prepared(),
        _compiled(),
        _request(max_hypotheses=3, critic=True),
        lambda _event: None,
    )

    assert "killed by critic" in output.markdown
    assert "contradicted by 10 ns" in output.markdown


def test_critic_uses_the_configured_synthesis_model():
    model = CriticModel()

    ExhaustiveHypothesisRunner(model, CostLedger(), model="vendor/pinned")(
        _prepared(),
        _compiled(),
        _request(max_hypotheses=3, critic=True),
        lambda _event: None,
    )

    assert {call["model"] for call in model.calls} == {"vendor/pinned"}


def test_synthesis_retry_tells_the_model_what_failed_and_backs_off():
    class BrokenThenValidModel(HypothesisModel):
        def __call__(self, **kwargs):
            if len(self.calls) < 2:
                self.calls.append(kwargs)
                return ModelResponse(
                    text='{"hypotheses": "not a list"}',
                    usage=UsageRecord(
                        stage=kwargs["stage"],
                        model=kwargs["model"],
                        input_tokens=1,
                        output_tokens=1,
                        total_cost=0.0001,
                    ),
                )
            return super().__call__(**kwargs)

    model = BrokenThenValidModel()
    delays = []

    ExhaustiveHypothesisRunner(model, CostLedger(), sleep=delays.append)(
        _prepared(),
        _compiled(),
        _request(),
        lambda _event: None,
    )

    prompts = [call["user"] for call in model.calls]
    assert len(prompts) == 3
    assert prompts[1] != prompts[0] and prompts[2] != prompts[0]
    assert all(
        "Previous attempt failed schema validation" in prompt
        for prompt in prompts[1:]
    )
    assert delays == [1.0, 2.0]


def test_provider_rejection_is_not_retried():
    class RateLimitedModel(HypothesisModel):
        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            raise _RateLimitError("429 slow down")

    model = RateLimitedModel()
    delays = []

    with pytest.raises(RuntimeError, match="provider rejected the call"):
        ExhaustiveHypothesisRunner(model, CostLedger(), sleep=delays.append)(
            _prepared(),
            _compiled(),
            _request(),
            lambda _event: None,
        )

    assert len(model.calls) == 1
    assert delays == []


def test_similarity_builds_each_snapshot_entity_index_only_once():
    class CountingNames:
        def __init__(self, names):
            self.names = tuple(names)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.names)

    prepared = _prepared()
    counters = []
    for snapshot in prepared.snapshots.values():
        counter = CountingNames(snapshot.entity_names)
        snapshot.entity_names = counter
        counters.append(counter)

    ExhaustiveHypothesisRunner(HypothesisModel(), CostLedger())(
        prepared,
        _compiled(),
        _request(max_hypotheses=3),
        lambda _event: None,
    )

    assert [counter.iterations for counter in counters] == [1, 1, 1]

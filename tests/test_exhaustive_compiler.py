import json
from dataclasses import replace

import pytest

from mira.exhaustive.compiler import (
    MAP_SYSTEM_PROMPT,
    EvidenceBatchError,
    EvidenceCompiler,
    ModelResponse,
    batch_response_schema,
)
from mira.exhaustive.costs import CostLedger
from mira.exhaustive.types import (
    EvidenceChunk,
    EvidenceProvenance,
    ResearchEvidenceBundle,
    UsageRecord,
)


RETRY_MARKER = "\n\nPrevious attempt failed"


def prompt_payload(user: str) -> dict:
    """Parse a JSON user prompt, ignoring any appended retry feedback."""
    return json.loads(user.split(RETRY_MARKER, 1)[0])


def _bundle(count=3):
    base = ResearchEvidenceBundle.empty("How do the systems interact?")
    chunks = tuple(
        EvidenceChunk(
            chunk_id=f"chunk-{index:02d}",
            content_hash=f"sha256:{index:02d}",
            content=f"Measured claim {index}",
            domains=("memory",),
            file_paths=(f"paper-{index}.md",),
            paper_names=(f"Paper {index}",),
            source_chunk_ids=(f"chunk-{index:02d}",),
            provenance=(EvidenceProvenance(
                source_chunk_id=f"chunk-{index:02d}",
                domain="memory",
                file_path=f"paper-{index}.md",
                paper_name=f"Paper {index}",
            ),),
        )
        for index in range(count)
    )
    return replace(
        base,
        chunks=chunks,
        paper_names=tuple(f"Paper {index}" for index in range(count)),
        exhaustive=True,
    )


class ValidModel:
    def __init__(self):
        self.calls = []

    def __call__(self, *, model, system, user, stage, max_output_tokens):
        payload = prompt_payload(user)
        self.calls.append((stage, payload))
        if stage == "evidence_map":
            chunk_ids = [item["chunk_id"] for item in payload["chunks"]]
            chunks = {
                item["chunk_id"]: item for item in payload["chunks"]
            }
            records = [{
                "claim": f"Claim from {chunk_id}",
                "measurements": [f"{chunk_id}: 10 ns"],
                "mechanisms": ["mechanism"],
                "assumptions": [],
                "limitations": ["small sample"],
                "contradictions": [f"{chunk_id}: conflicting baseline"],
                "citations": [{
                    "chunk_id": chunk_id,
                    "source_chunk_id": item["source_chunk_id"],
                    "file_path": item["file_path"],
                    "title": item["paper_name"],
                    "domains": [item["domain"]],
                } for item in chunks[chunk_id]["provenance"]],
                "relevance": "direct",
                "contributing_chunk_ids": [chunk_id],
            } for chunk_id in chunk_ids]
            text = json.dumps({
                "processed_chunk_ids": chunk_ids,
                "records": records,
            })
        elif stage == "evidence_reduce":
            chunk_ids = payload["expected_chunk_ids"]
            records = payload["records"]
            text = json.dumps({
                "processed_chunk_ids": chunk_ids,
                "records": [{
                    "claim": "Reduced claims",
                    "measurements": [
                        measurement
                        for record in records
                        for measurement in record["measurements"]
                    ],
                    "mechanisms": ["mechanism"],
                    "assumptions": [],
                    "limitations": [
                        limitation
                        for record in records
                        for limitation in record["limitations"]
                    ],
                    "contradictions": [
                        contradiction
                        for record in records
                        for contradiction in record["contradictions"]
                    ],
                    "citations": [
                        citation
                        for record in records
                        for citation in record["citations"]
                    ],
                    "relevance": "reduced",
                    "contributing_chunk_ids": chunk_ids,
                }],
            })
        else:
            text = "# Exhaustive answer\n\nGrounded result."
        return ModelResponse(
            text=text,
            usage=UsageRecord(
                stage=stage,
                model=model,
                input_tokens=100,
                output_tokens=20,
                total_cost=0.001,
                generation_id=f"gen-{len(self.calls)}",
            ),
        )


def test_batch_partition_processes_every_chunk_exactly_once():
    model = ValidModel()
    compiler = EvidenceCompiler(
        model,
        CostLedger(),
        batch_tokens=45,
        token_counter=len,
        prompt_overhead_tokens=0,
    )

    compiled = compiler.compile(_bundle(17), lambda _event: None)

    mapped = [
        chunk_id
        for stage, payload in model.calls
        if stage == "evidence_map"
        for chunk_id in (item["chunk_id"] for item in payload["chunks"])
    ]
    assert sorted(mapped) == [f"chunk-{index:02d}" for index in range(17)]
    assert len(mapped) == len(set(mapped))
    assert compiled.contributing_chunk_ids == tuple(sorted(mapped))


def test_public_batch_count_matches_compilation_partitioning():
    compiler = EvidenceCompiler(
        ValidModel(),
        CostLedger(),
        batch_tokens=45,
        token_counter=len,
        prompt_overhead_tokens=0,
    )

    assert compiler.batch_count(_bundle(17).chunks) == 9


def test_permanent_batch_schema_failure_retries_twice_then_aborts():
    prompts = []

    def invalid_model(**kwargs):
        prompts.append(kwargs["user"])
        return ModelResponse(
            text='{"records": []}',
            usage=UsageRecord(
                stage="evidence_map",
                model="google/gemini-2.5-flash",
                input_tokens=1,
                output_tokens=1,
                total_cost=0.0001,
            ),
        )

    delays = []
    compiler = EvidenceCompiler(
        invalid_model,
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
        sleep=delays.append,
    )

    with pytest.raises(EvidenceBatchError) as exc:
        compiler.compile(_bundle(1), lambda _event: None)

    assert exc.value.chunk_ids == ("chunk-00",)
    assert len(prompts) == 3
    # An identical retry is spend with no new information in it.
    assert prompts[1] != prompts[0] and prompts[2] != prompts[0]
    assert all(
        "Previous attempt failed schema validation" in prompt
        for prompt in prompts[1:]
    )
    assert delays == [1.0, 2.0]


def test_model_cannot_acknowledge_chunks_without_returning_evidence():
    def empty_model(**kwargs):
        payload = prompt_payload(kwargs["user"])
        chunk_ids = [item["chunk_id"] for item in payload["chunks"]]
        return ModelResponse(
            text=json.dumps({
                "processed_chunk_ids": chunk_ids,
                "records": [],
            }),
            usage=UsageRecord(
                stage=kwargs["stage"],
                model=kwargs["model"],
                input_tokens=1,
                output_tokens=1,
                total_cost=0.0001,
            ),
        )

    compiler = EvidenceCompiler(
        empty_model,
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(EvidenceBatchError, match="every expected chunk"):
        compiler.compile(_bundle(1), lambda _event: None)


def test_model_cannot_forge_citation_provenance():
    class ForgedCitationModel(ValidModel):
        def __call__(self, **kwargs):
            response = super().__call__(**kwargs)
            if kwargs["stage"] != "evidence_map":
                return response
            payload = json.loads(response.text)
            payload["records"][0]["citations"][0]["file_path"] = "forged.md"
            return replace(response, text=json.dumps(payload))

    compiler = EvidenceCompiler(
        ForgedCitationModel(),
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(EvidenceBatchError, match="citation provenance"):
        compiler.compile(_bundle(1), lambda _event: None)


def test_deduplicated_chunk_rejects_crossed_provenance_tuple():
    bundle = _bundle(1)
    chunk = replace(
        bundle.chunks[0],
        domains=("memory", "optical"),
        file_paths=("memory.md", "optical.md"),
        paper_names=("Memory Paper", "Optical Paper"),
        source_chunk_ids=("memory-chunk", "optical-chunk"),
        provenance=(
            EvidenceProvenance(
                "memory-chunk", "memory", "memory.md", "Memory Paper"
            ),
            EvidenceProvenance(
                "optical-chunk", "optical", "optical.md", "Optical Paper"
            ),
        ),
    )
    bundle = replace(bundle, chunks=(chunk,))

    class CrossedCitationModel(ValidModel):
        def __call__(self, **kwargs):
            response = super().__call__(**kwargs)
            if kwargs["stage"] != "evidence_map":
                return response
            payload = json.loads(response.text)
            citation = payload["records"][0]["citations"][0]
            citation.update({
                "source_chunk_id": "memory-chunk",
                "file_path": "memory.md",
                "title": "Optical Paper",
                "domains": ["memory"],
            })
            return replace(response, text=json.dumps(payload))

    compiler = EvidenceCompiler(
        CrossedCitationModel(),
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(EvidenceBatchError, match="citation provenance"):
        compiler.compile(bundle, lambda _event: None)


def test_deduplicated_chunk_requires_every_source_provenance_citation():
    bundle = _bundle(1)
    chunk = replace(
        bundle.chunks[0],
        domains=("memory", "optical"),
        file_paths=("memory.md", "optical.md"),
        paper_names=("Memory Paper", "Optical Paper"),
        source_chunk_ids=("memory-chunk", "optical-chunk"),
        provenance=(
            EvidenceProvenance(
                "memory-chunk", "memory", "memory.md", "Memory Paper"
            ),
            EvidenceProvenance(
                "optical-chunk", "optical", "optical.md", "Optical Paper"
            ),
        ),
    )
    bundle = replace(bundle, chunks=(chunk,))

    class OmittingCitationModel(ValidModel):
        def __call__(self, **kwargs):
            response = super().__call__(**kwargs)
            if kwargs["stage"] != "evidence_map":
                return response
            payload = json.loads(response.text)
            payload["records"][0]["citations"] = (
                payload["records"][0]["citations"][:1]
            )
            return replace(response, text=json.dumps(payload))

    compiler = EvidenceCompiler(
        OmittingCitationModel(),
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(EvidenceBatchError, match="every exact provenance"):
        compiler.compile(bundle, lambda _event: None)


def test_compiler_rejects_evidence_without_exact_provenance():
    bundle = _bundle(1)
    bundle = replace(
        bundle,
        chunks=(replace(bundle.chunks[0], provenance=()),),
    )
    model = ValidModel()
    compiler = EvidenceCompiler(
        model,
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
    )

    with pytest.raises(EvidenceBatchError, match="exact provenance"):
        compiler.compile(bundle, lambda _event: None)

    assert model.calls == []


def test_reduction_preserves_measurements_contradictions_and_citations():
    model = ValidModel()
    compiler = EvidenceCompiler(
        model,
        CostLedger(),
        batch_tokens=600,
        compiled_target_tokens=100,
        token_counter=len,
        prompt_overhead_tokens=0,
    )

    compiled = compiler.compile(_bundle(4), lambda _event: None)

    assert any("10 ns" in value for value in compiled.measurements)
    assert len(compiled.contradictions) == 4
    assert {citation.chunk_id for citation in compiled.citations} == {
        "chunk-00", "chunk-01", "chunk-02", "chunk-03"
    }
    assert compiled.contributing_chunk_ids == (
        "chunk-00", "chunk-01", "chunk-02", "chunk-03"
    )
    assert any(stage == "evidence_reduce" for stage, _ in model.calls)


def test_model_usage_is_recorded_by_stage():
    model = ValidModel()
    ledger = CostLedger()
    compiler = EvidenceCompiler(
        model,
        ledger,
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
    )

    compiled = compiler.compile(_bundle(2), lambda _event: None)
    answer = compiler.synthesize_chat(compiled, lambda _event: None)

    assert answer.markdown.startswith("# Exhaustive answer")
    assert ledger.summary().actual_cost_usd == pytest.approx(0.002)
    assert set(ledger.summary().by_stage) == {"evidence_map", "synthesis"}


def test_chat_synthesis_includes_bounded_conversation_history():
    model = ValidModel()
    compiler = EvidenceCompiler(
        model,
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
    )
    compiled = compiler.compile(_bundle(1), lambda _event: None)

    compiler.synthesize_chat(
        compiled,
        lambda _event: None,
        history=("user: earlier question", "assistant: earlier answer"),
    )

    synthesis = next(
        payload for stage, payload in model.calls if stage == "synthesis"
    )
    assert synthesis["conversation_history"] == [
        "user: earlier question",
        "assistant: earlier answer",
    ]


def test_reduction_aborts_when_a_model_does_not_reduce_the_record_count():
    class NonReducingModel(ValidModel):
        def __init__(self):
            super().__init__()
            self.reduce_calls = 0

        def __call__(self, **kwargs):
            if kwargs["stage"] != "evidence_reduce":
                return super().__call__(**kwargs)
            self.reduce_calls += 1
            if self.reduce_calls > 1:
                raise AssertionError("reducer was called after making no progress")
            payload = prompt_payload(kwargs["user"])
            self.calls.append((kwargs["stage"], payload))
            return ModelResponse(
                text=json.dumps({
                    "processed_chunk_ids": payload["expected_chunk_ids"],
                    "records": payload["records"],
                }),
                usage=UsageRecord(
                    stage=kwargs["stage"],
                    model=kwargs["model"],
                    input_tokens=100,
                    output_tokens=20,
                    total_cost=0.001,
                ),
            )

    model = NonReducingModel()
    compiler = EvidenceCompiler(
        model,
        CostLedger(),
        batch_tokens=1_000,
        compiled_target_tokens=100,
        token_counter=len,
        prompt_overhead_tokens=0,
    )

    with pytest.raises(EvidenceBatchError, match="did not reduce"):
        compiler.compile(_bundle(2), lambda _event: None)

    assert model.reduce_calls == 1


def test_input_shaped_citation_payload_needs_no_rename_to_validate():
    """The model may echo the field names it was given.

    A deduplicated chunk carries one provenance entry per domain for the same
    source; a citation naming both domains at once is the input's own shape,
    not a forgery, and requiring `title` instead of `paper_name` would make
    the first attempt fail for a purely cosmetic reason."""
    bundle = _bundle(1)
    chunk = replace(
        bundle.chunks[0],
        domains=("memory", "optical"),
        provenance=(
            EvidenceProvenance(
                "chunk-00", "memory", "paper-0.md", "Paper 0"
            ),
            EvidenceProvenance(
                "chunk-00", "optical", "paper-0.md", "Paper 0"
            ),
        ),
    )
    bundle = replace(bundle, chunks=(chunk,))

    class InputShapedModel:
        def __init__(self):
            self.calls = []

        def __call__(self, *, model, system, user, stage, max_output_tokens):
            payload = prompt_payload(user)
            self.calls.append((stage, system))
            item = payload["chunks"][0]
            return ModelResponse(
                text=json.dumps({
                    "processed_chunk_ids": [item["chunk_id"]],
                    "records": [{
                        "claim": "Claim",
                        "measurements": [],
                        "mechanisms": [],
                        "assumptions": [],
                        "limitations": [],
                        "contradictions": [],
                        "citations": [{
                            "chunk_id": item["chunk_id"],
                            "source_chunk_id": "chunk-00",
                            "file_path": "paper-0.md",
                            "paper_name": "Paper 0",
                            "domains": ["memory", "optical"],
                        }],
                        "relevance": "direct",
                        "contributing_chunk_ids": [item["chunk_id"]],
                    }],
                }),
                usage=UsageRecord(
                    stage=stage,
                    model=model,
                    input_tokens=1,
                    output_tokens=1,
                    total_cost=0.0001,
                ),
            )

    model = InputShapedModel()
    compiler = EvidenceCompiler(
        model,
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
        sleep=lambda _seconds: None,
    )

    compiled = compiler.compile(bundle, lambda _event: None)

    assert len(model.calls) == 1
    assert compiled.citations[0].title == "Paper 0"
    assert compiled.citations[0].domains == ("memory", "optical")


def test_map_prompt_states_every_required_field_name():
    for field in (
        "processed_chunk_ids",
        "records",
        "claim",
        "measurements",
        "mechanisms",
        "assumptions",
        "limitations",
        "contradictions",
        "citations",
        "relevance",
        "contributing_chunk_ids",
        "chunk_id",
        "source_chunk_id",
        "file_path",
        "paper_name",
        "domains",
    ):
        assert field in MAP_SYSTEM_PROMPT, field
    assert "LIST of domain strings" in MAP_SYSTEM_PROMPT
    for domain in ("memory", "optical", "storage"):
        assert domain in MAP_SYSTEM_PROMPT


def test_response_schema_matches_the_validated_payload_models():
    schema = batch_response_schema()

    citation = schema["$defs"]["_CitationPayload"]

    assert set(schema["properties"]) == {"processed_chunk_ids", "records"}
    assert set(citation["properties"]) == {
        "chunk_id", "source_chunk_id", "file_path", "paper_name", "domains",
    }
    assert citation["additionalProperties"] is False
    assert citation["properties"]["domains"]["type"] == "array"


def test_provider_rejection_aborts_without_burning_the_retry_budget():
    class RateLimited(RuntimeError):
        status_code = 429

    calls = 0

    def rate_limited_model(**_kwargs):
        nonlocal calls
        calls += 1
        raise RateLimited("429 too many requests")

    delays = []
    compiler = EvidenceCompiler(
        rate_limited_model,
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
        sleep=delays.append,
    )

    with pytest.raises(EvidenceBatchError, match="provider rejected"):
        compiler.compile(_bundle(1), lambda _event: None)

    assert calls == 1
    assert delays == []


def test_oversized_chunk_is_split_rather_than_aborting_the_compilation():
    bundle = _bundle(2)
    bundle = replace(
        bundle,
        chunks=(
            replace(bundle.chunks[0], content="x" * 90),
            bundle.chunks[1],
        ),
    )
    model = ValidModel()
    compiler = EvidenceCompiler(
        model,
        CostLedger(),
        batch_tokens=40,
        token_counter=len,
        prompt_overhead_tokens=0,
    )

    compiled = compiler.compile(bundle, lambda _event: None)

    transported = [
        item["chunk_id"]
        for stage, payload in model.calls
        if stage == "evidence_map"
        for item in payload["chunks"]
    ]
    assert any("#part" in chunk_id for chunk_id in transported)
    # Coverage is still proven against the selected chunks, not the parts.
    assert compiled.contributing_chunk_ids == ("chunk-00", "chunk-01")
    assert {citation.chunk_id for citation in compiled.citations} == {
        "chunk-00", "chunk-01"
    }


def test_unique_citations_keep_distinct_source_chunks_of_one_paper():
    bundle = _bundle(1)
    chunk = replace(
        bundle.chunks[0],
        source_chunk_ids=("chunk-00", "chunk-99"),
        provenance=(
            EvidenceProvenance(
                "chunk-00", "memory", "paper-0.md", "Paper 0"
            ),
            EvidenceProvenance(
                "chunk-99", "memory", "paper-0.md", "Paper 0"
            ),
        ),
    )
    bundle = replace(bundle, chunks=(chunk,))

    compiled = EvidenceCompiler(
        ValidModel(),
        CostLedger(),
        batch_tokens=1_000,
        token_counter=len,
        prompt_overhead_tokens=0,
    ).compile(bundle, lambda _event: None)

    assert {
        citation.source_chunk_id for citation in compiled.citations
    } == {"chunk-00", "chunk-99"}

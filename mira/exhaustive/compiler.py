"""Tiered Gemini evidence extraction and Sonnet final synthesis."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Protocol

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from mira.config import shared_llm_model

from .costs import CostLedger
from .types import (
    DOMAIN_NAMES,
    DomainName,
    EvidenceChunk,
    EvidenceProvenance,
    ResearchEvidenceBundle,
    UsageRecord,
)


# Resolved from the shared `llm_models` block so a model swap is a config edit,
# not a code edit. The literals are the shipped fallbacks for a config that
# predates these optional keys.
MAP_MODEL = shared_llm_model("exhaustive_map", "google/gemini-2.5-flash")
SYNTHESIS_MODEL = shared_llm_model(
    "exhaustive_synthesis", "anthropic/claude-sonnet-5"
)
# Retry ceiling for the JSON-returning synthesis stages. Lives here rather than
# in mira.hypothesis so the cost estimate and the retry loop cannot drift.
MAX_SYNTHESIS_ATTEMPTS = 3

# The validated payload models forbid unknown fields, so an undocumented
# schema is a guaranteed retry loop. Every required name and shape is stated
# here verbatim, matching _BatchPayload / _EvidenceRecordPayload /
# _CitationPayload exactly.
_BATCH_SCHEMA_PROMPT = (
    "Return ONLY a JSON object with exactly these two fields:\n"
    '  "processed_chunk_ids": list[str] — every chunk_id you were given, '
    "each exactly once, copied verbatim.\n"
    '  "records": list of evidence record objects.\n'
    "Each evidence record object has exactly these fields:\n"
    '  "claim": str (required)\n'
    '  "measurements": list[str]\n'
    '  "mechanisms": list[str]\n'
    '  "assumptions": list[str]\n'
    '  "limitations": list[str]\n'
    '  "contradictions": list[str]\n'
    '  "citations": list of citation objects\n'
    '  "relevance": str (required)\n'
    '  "contributing_chunk_ids": list[str] (required) — the chunk_ids this '
    "record was drawn from, copied verbatim.\n"
    "Each citation object has exactly these fields:\n"
    '  "chunk_id": str — the chunk_id of the supplied chunk.\n'
    '  "source_chunk_id": str — copied verbatim from that chunk\'s '
    "provenance entry.\n"
    '  "file_path": str — copied verbatim from the same provenance entry.\n'
    '  "paper_name": str — copied verbatim from the same provenance entry.\n'
    '  "domains": LIST of domain strings — every domain the same provenance '
    "entry appears under, from "
    + ", ".join(DOMAIN_NAMES)
    + ". Always a list, never a bare string, even for one domain.\n"
    "Add no other fields anywhere. Invent no values: every citation field "
    "except chunk_id is copied verbatim from a supplied provenance entry."
)

MAP_SYSTEM_PROMPT = (
    "Extract structured scientific evidence from every supplied chunk. "
    "Across your records, acknowledge every chunk_id exactly once and cite "
    "every exact provenance entry of every chunk exactly once.\n"
    + _BATCH_SCHEMA_PROMPT
)

REDUCE_SYSTEM_PROMPT = (
    "Reduce the evidence records without dropping numerical results, "
    "contradictions, citations, or contributing chunk IDs. Keep every "
    "expected_chunk_id and every citation from the input.\n"
    + _BATCH_SCHEMA_PROMPT
)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    usage: UsageRecord


class ModelCall(Protocol):
    def __call__(
        self,
        *,
        model: str,
        system: str,
        user: str,
        stage: str,
        max_output_tokens: int,
    ) -> ModelResponse: ...


@dataclass(frozen=True, slots=True)
class CitationRecord:
    chunk_id: str
    file_path: str
    title: str
    domains: tuple[DomainName, ...]
    source_chunk_id: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    claim: str
    measurements: tuple[str, ...]
    mechanisms: tuple[str, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    contradictions: tuple[str, ...]
    citations: tuple[CitationRecord, ...]
    relevance: str
    contributing_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledEvidence:
    query: str
    records: tuple[EvidenceRecord, ...]
    contributing_chunk_ids: tuple[str, ...]
    measurements: tuple[str, ...]
    contradictions: tuple[str, ...]
    citations: tuple[CitationRecord, ...]


@dataclass(frozen=True, slots=True)
class ResearchAnswer:
    markdown: str
    citations: tuple[CitationRecord, ...]
    contributing_chunk_ids: tuple[str, ...]


class _CitationPayload(BaseModel):
    # `paper_name` is the field name the map input actually carries, so the
    # model can copy provenance through verbatim. `title` stays accepted as an
    # alias: the reduce stage re-reads records this compiler serialized, and a
    # rename the model has to invent is a retry loop, not a safeguard.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    chunk_id: str
    source_chunk_id: str
    file_path: str = ""
    paper_name: str = Field(
        default="",
        validation_alias=AliasChoices("paper_name", "title"),
    )
    domains: list[DomainName] = Field(min_length=1)


class _EvidenceRecordPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    measurements: list[str] = Field(default_factory=list)
    mechanisms: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    citations: list[_CitationPayload] = Field(default_factory=list)
    relevance: str
    contributing_chunk_ids: list[str]


class _BatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    processed_chunk_ids: list[str]
    records: list[_EvidenceRecordPayload]


class EvidenceBatchError(RuntimeError):
    def __init__(self, message: str, chunk_ids: Iterable[str]):
        self.chunk_ids = tuple(sorted(chunk_ids))
        super().__init__(message)


class ProviderCallError(RuntimeError):
    """A provider rejected the call itself (auth, quota, rate limit).

    Retrying an identical request cannot fix any of these, so the compiler
    surfaces them immediately instead of burning its validation budget.
    """


_NON_RETRYABLE_STATUS = frozenset({401, 402, 403, 404, 429})
_NON_RETRYABLE_NAMES = (
    "authentication",
    "permission",
    "ratelimit",
    "rate_limit",
    "notfound",
    "unprocessable",
)


def is_provider_failure(exc: BaseException) -> bool:
    """Whether an exception is a provider rejection rather than bad output."""
    if isinstance(exc, ProviderCallError):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _NON_RETRYABLE_STATUS:
        return True
    name = type(exc).__name__.casefold()
    return any(marker in name for marker in _NON_RETRYABLE_NAMES)


def batch_response_schema() -> dict:
    """JSON schema for a map/reduce batch, derived from the validated models."""
    return _BatchPayload.model_json_schema()


# Suffix appended to the transport id of an oversized chunk's parts. Chunk ids
# are LightRAG content digests, which never contain it; `compile` rejects a
# bundle that proves otherwise rather than silently merging two chunks.
_PART_SEPARATOR = "#part"


def _parent_chunk_id(chunk_id: str) -> str:
    return chunk_id.split(_PART_SEPARATOR, 1)[0]


def _retry_prompt(user: str, error: Exception) -> str:
    """Tell the model what was wrong; an identical retry is pure spend."""
    return (
        f"{user}\n\nPrevious attempt failed schema validation: {error}. "
        "Return ONLY corrected JSON."
    )


def _default_token_counter(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _ordered_domains(domains: Iterable[str]) -> tuple[DomainName, ...]:
    """Normalize a domain list to canonical order so comparison is stable."""
    present = set(domains)
    return tuple(domain for domain in DOMAIN_NAMES if domain in present)


def _provenance_registry(
    provenance: Iterable[EvidenceProvenance],
) -> set[tuple[str, str, str, tuple[DomainName, ...]]]:
    """Group exact provenance by source, collecting every domain it came from.

    A chunk deduplicated across graphs keeps one entry per (source chunk, file,
    paper) with all of its domains, so a citation may legitimately name several
    domains at once.
    """
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for item in provenance:
        grouped.setdefault(
            (item.source_chunk_id, item.file_path, item.paper_name), set()
        ).add(item.domain)
    return {
        (*key, _ordered_domains(domains))
        for key, domains in grouped.items()
    }


def _citation_to_dict(citation: CitationRecord) -> dict:
    return {
        "chunk_id": citation.chunk_id,
        "source_chunk_id": citation.source_chunk_id,
        "file_path": citation.file_path,
        "paper_name": citation.title,
        "domains": list(citation.domains),
    }


def _record_to_dict(record: EvidenceRecord) -> dict:
    return {
        "claim": record.claim,
        "measurements": list(record.measurements),
        "mechanisms": list(record.mechanisms),
        "assumptions": list(record.assumptions),
        "limitations": list(record.limitations),
        "contradictions": list(record.contradictions),
        "citations": [_citation_to_dict(item) for item in record.citations],
        "relevance": record.relevance,
        "contributing_chunk_ids": list(record.contributing_chunk_ids),
    }


class EvidenceCompiler:
    def __init__(
        self,
        model_call: ModelCall,
        ledger: CostLedger,
        *,
        batch_tokens: int = 80_000,
        compiled_target_tokens: int = 120_000,
        token_counter: Callable[[str], int] = _default_token_counter,
        prompt_overhead_tokens: int = 512,
        map_model: str = MAP_MODEL,
        synthesis_model: str = SYNTHESIS_MODEL,
        max_reduction_rounds: int = 8,
        max_validation_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if batch_tokens <= 0 or compiled_target_tokens <= 0:
            raise ValueError("compiler token limits must be positive")
        if prompt_overhead_tokens < 0:
            raise ValueError("prompt_overhead_tokens must be non-negative")
        if max_reduction_rounds <= 0:
            raise ValueError("max_reduction_rounds must be positive")
        if max_validation_attempts <= 0:
            raise ValueError("max_validation_attempts must be positive")
        self.model_call = model_call
        self.ledger = ledger
        self.batch_tokens = batch_tokens
        self.compiled_target_tokens = compiled_target_tokens
        self.token_counter = token_counter
        self.prompt_overhead_tokens = prompt_overhead_tokens
        self.map_model = map_model
        self.synthesis_model = synthesis_model
        self.max_reduction_rounds = max_reduction_rounds
        self.max_validation_attempts = max_validation_attempts
        self.sleep = sleep

    def compile(
        self,
        bundle: ResearchEvidenceBundle,
        emit: Callable[[dict], None],
    ) -> CompiledEvidence:
        if not bundle.exhaustive:
            raise EvidenceBatchError(
                "cannot compile an incomplete evidence bundle",
                (chunk.chunk_id for chunk in bundle.chunks),
            )
        missing_provenance = tuple(
            chunk.chunk_id
            for chunk in bundle.chunks
            if not chunk.provenance
        )
        if missing_provenance:
            raise EvidenceBatchError(
                "selected evidence is missing exact provenance",
                missing_provenance,
            )
        reserved = tuple(
            chunk.chunk_id
            for chunk in bundle.chunks
            if _PART_SEPARATOR in chunk.chunk_id
        )
        if reserved:
            raise EvidenceBatchError(
                f"chunk ids may not contain {_PART_SEPARATOR!r}",
                reserved,
            )
        batches = self._partition_chunks(bundle.chunks)
        # An oversized chunk is split into parts for transport only; every
        # part carries its parent's id back so coverage is still proven against
        # the chunks the collector selected.
        parents = {
            chunk.chunk_id: _parent_chunk_id(chunk.chunk_id)
            for batch in batches
            for chunk in batch
        }
        records: list[EvidenceRecord] = []
        processed: list[str] = []
        for index, batch in enumerate(batches, 1):
            emit({
                "name": "evidence_batch_started",
                "batch": index,
                "total": len(batches),
            })
            expected = tuple(chunk.chunk_id for chunk in batch)
            citation_registry = {
                chunk.chunk_id: _provenance_registry(chunk.provenance)
                for chunk in batch
            }
            payload = {
                "question": bundle.query,
                "chunks": [{
                    "chunk_id": chunk.chunk_id,
                    "source_chunk_ids": list(chunk.source_chunk_ids),
                    "domains": list(chunk.domains),
                    "file_paths": list(chunk.file_paths),
                    "paper_names": list(chunk.paper_names),
                    "provenance": [{
                        "source_chunk_id": item.source_chunk_id,
                        "domain": item.domain,
                        "file_path": item.file_path,
                        "paper_name": item.paper_name,
                    } for item in chunk.provenance],
                    "content": chunk.content,
                } for chunk in batch],
            }
            parsed = self._validated_call(
                model=self.map_model,
                system=MAP_SYSTEM_PROMPT,
                user=json.dumps(payload, ensure_ascii=False),
                stage="evidence_map",
                expected_chunk_ids=expected,
                citation_registry=citation_registry,
                emit=emit,
            )
            records.extend(self._records(parsed, parents))
            processed.extend(expected)
            emit({
                "name": "evidence_batch_completed",
                "batch": index,
                "total": len(batches),
            })

        if len(processed) != len(set(processed)):
            raise EvidenceBatchError(
                "a chunk was processed more than once",
                (parents[chunk_id] for chunk_id in processed),
            )
        expected_all = tuple(sorted(chunk.chunk_id for chunk in bundle.chunks))
        covered = {parents[chunk_id] for chunk_id in processed}
        if sorted(covered) != list(expected_all) or sorted(processed) != sorted(
            parents
        ):
            raise EvidenceBatchError(
                "compiled evidence does not cover every selected chunk",
                set(expected_all) - covered,
            )

        records = self._reduce_records(bundle.query, records, emit)
        citations = self._unique_citations(
            citation for record in records for citation in record.citations
        )
        return CompiledEvidence(
            query=bundle.query,
            records=tuple(records),
            contributing_chunk_ids=expected_all,
            measurements=tuple(
                value for record in records for value in record.measurements
            ),
            contradictions=tuple(
                value for record in records for value in record.contradictions
            ),
            citations=citations,
        )

    def synthesize_chat(
        self,
        compiled: CompiledEvidence,
        emit: Callable[[dict], None],
        history: Iterable[str] = (),
    ) -> ResearchAnswer:
        emit({"name": "synthesis_started"})
        response = self.model_call(
            model=self.synthesis_model,
            system=(
                "You are MIRA, a cross-domain research strategist. Answer only "
                "from the compiled evidence. Preserve contradictions and cite "
                "the supplied paper titles and paths. Never invent a citation."
            ),
            user=json.dumps({
                "question": compiled.query,
                "conversation_history": list(history),
                "coverage": {
                    "contributing_chunk_ids": list(
                        compiled.contributing_chunk_ids
                    ),
                },
                "records": [
                    _record_to_dict(record) for record in compiled.records
                ],
            }, ensure_ascii=False),
            stage="synthesis",
            max_output_tokens=8_000,
        )
        self.ledger.record_usage(replace(
            response.usage,
            stage="synthesis",
            model=self.synthesis_model,
        ))
        self._emit_cost(emit)
        if not response.text.strip():
            raise EvidenceBatchError(
                "synthesis returned empty content",
                compiled.contributing_chunk_ids,
            )
        emit({"name": "synthesis_completed"})
        return ResearchAnswer(
            markdown=response.text.strip(),
            citations=compiled.citations,
            contributing_chunk_ids=compiled.contributing_chunk_ids,
        )

    def _split_oversized(
        self, chunk: EvidenceChunk
    ) -> tuple[EvidenceChunk, ...]:
        """Halve a chunk that cannot fit one batch until every part fits.

        Aborting the whole compilation because one source document is large
        loses every other chunk's evidence; the parts keep identical
        provenance, so citation validation and coverage are unaffected.
        """
        budget = self.batch_tokens - self.prompt_overhead_tokens
        if self.token_counter(chunk.content) <= budget:
            return (chunk,)
        if len(chunk.content) < 2:
            raise EvidenceBatchError(
                f"chunk {chunk.chunk_id} cannot be split below the evidence "
                "batch size",
                (chunk.chunk_id,),
            )
        middle = len(chunk.content) // 2
        halves = (chunk.content[:middle], chunk.content[middle:])
        parts: list[EvidenceChunk] = []
        for half in halves:
            parts.extend(self._split_oversized(replace(chunk, content=half)))
        total = len(parts)
        return tuple(
            replace(
                part,
                chunk_id=f"{_parent_chunk_id(part.chunk_id)}"
                f"{_PART_SEPARATOR}{index}/{total}",
            )
            for index, part in enumerate(parts, 1)
        )

    def _partition_chunks(
        self, chunks: tuple[EvidenceChunk, ...]
    ) -> tuple[tuple[EvidenceChunk, ...], ...]:
        batches = []
        current = []
        used = self.prompt_overhead_tokens
        transported = tuple(
            part for chunk in chunks for part in self._split_oversized(chunk)
        )
        for chunk in transported:
            size = self.token_counter(chunk.content)
            if current and used + size > self.batch_tokens:
                batches.append(tuple(current))
                current = []
                used = self.prompt_overhead_tokens
            current.append(chunk)
            used += size
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    def batch_count(self, chunks: tuple[EvidenceChunk, ...]) -> int:
        """Return the number of map batches using the compiler's real policy."""
        return len(self._partition_chunks(chunks))

    def _validated_call(
        self,
        *,
        model: str,
        system: str,
        user: str,
        stage: str,
        expected_chunk_ids: tuple[str, ...],
        citation_registry: dict[
            str,
            set[tuple[str, str, str, tuple[DomainName, ...]]],
        ],
        emit: Callable[[dict], None],
    ) -> _BatchPayload:
        last_error: Exception | None = None
        for attempt in range(self.max_validation_attempts):
            try:
                response = self.model_call(
                    model=model,
                    system=system,
                    user=user if last_error is None else _retry_prompt(
                        user, last_error
                    ),
                    stage=stage,
                    max_output_tokens=16_000,
                )
                self.ledger.record_usage(replace(
                    response.usage, stage=stage, model=model
                ))
                self._emit_cost(emit)
                parsed = _BatchPayload.model_validate_json(response.text)
                if sorted(parsed.processed_chunk_ids) != sorted(
                    expected_chunk_ids
                ) or len(parsed.processed_chunk_ids) != len(
                    expected_chunk_ids
                ):
                    raise ValueError(
                        "model did not acknowledge every expected chunk"
                    )
                contributing = {
                    chunk_id
                    for record in parsed.records
                    for chunk_id in record.contributing_chunk_ids
                }
                if contributing != set(expected_chunk_ids):
                    raise ValueError(
                        "model evidence did not cover every expected chunk"
                    )
                cited_provenance = set()
                for record in parsed.records:
                    for citation in record.citations:
                        allowed = citation_registry.get(citation.chunk_id)
                        actual = (
                            citation.source_chunk_id,
                            citation.file_path,
                            citation.paper_name,
                            _ordered_domains(citation.domains),
                        )
                        if allowed is None or actual not in allowed:
                            raise ValueError(
                                "model returned invalid citation provenance"
                            )
                        cited_provenance.add(
                            (citation.chunk_id, *actual)
                        )
                expected_provenance = {
                    (chunk_id, *provenance)
                    for chunk_id, records in citation_registry.items()
                    for provenance in records
                }
                if cited_provenance != expected_provenance:
                    raise ValueError(
                        "model citations did not cover every exact provenance"
                    )
                return parsed
            except Exception as exc:
                # Only malformed output is worth another identical-cost call.
                # An auth or quota rejection would fail the same way twice more
                # and delay the real error the operator has to act on.
                if is_provider_failure(exc):
                    raise EvidenceBatchError(
                        f"{stage} failed: provider rejected the call: {exc}",
                        expected_chunk_ids,
                    ) from exc
                last_error = exc
                if attempt + 1 < self.max_validation_attempts:
                    self.sleep(2.0 ** attempt)
        raise EvidenceBatchError(
            f"{stage} failed after {self.max_validation_attempts} "
            f"attempts: {last_error}",
            expected_chunk_ids,
        )

    def _records(
        self,
        payload: _BatchPayload,
        parents: dict[str, str] | None = None,
    ) -> list[EvidenceRecord]:
        resolve = (
            (lambda chunk_id: parents.get(chunk_id, chunk_id))
            if parents else (lambda chunk_id: chunk_id)
        )
        return [
            EvidenceRecord(
                claim=record.claim,
                measurements=tuple(record.measurements),
                mechanisms=tuple(record.mechanisms),
                assumptions=tuple(record.assumptions),
                limitations=tuple(record.limitations),
                contradictions=tuple(record.contradictions),
                citations=tuple(CitationRecord(
                    chunk_id=resolve(citation.chunk_id),
                    source_chunk_id=citation.source_chunk_id,
                    file_path=citation.file_path,
                    title=citation.paper_name,
                    domains=_ordered_domains(citation.domains),
                ) for citation in record.citations),
                relevance=record.relevance,
                contributing_chunk_ids=tuple(sorted({
                    resolve(chunk_id)
                    for chunk_id in record.contributing_chunk_ids
                })),
            )
            for record in payload.records
        ]

    def _reduce_records(
        self,
        query: str,
        records: list[EvidenceRecord],
        emit: Callable[[dict], None],
    ) -> list[EvidenceRecord]:
        reduction_round = 0
        while len(records) > 1:
            input_tokens = self.token_counter(json.dumps([
                _record_to_dict(record) for record in records
            ], ensure_ascii=False))
            if input_tokens <= self.compiled_target_tokens:
                break
            reduction_round += 1
            if reduction_round > self.max_reduction_rounds:
                raise EvidenceBatchError(
                    "evidence reduction exceeded its configured round limit",
                    (
                        chunk_id
                        for record in records
                        for chunk_id in record.contributing_chunk_ids
                    ),
                )
            groups = self._partition_records(records)
            reduced: list[EvidenceRecord] = []
            for index, group in enumerate(groups, 1):
                expected = tuple(sorted({
                    chunk_id
                    for record in group
                    for chunk_id in record.contributing_chunk_ids
                }))
                citation_registry: dict[
                    str,
                    set[tuple[str, str, str, tuple[DomainName, ...]]],
                ] = {}
                for record in group:
                    for citation in record.citations:
                        citation_registry.setdefault(
                            citation.chunk_id, set()
                        ).add((
                            citation.source_chunk_id,
                            citation.file_path,
                            citation.title,
                            citation.domains,
                        ))
                emit({
                    "name": "evidence_reduce_started",
                    "batch": index,
                    "total": len(groups),
                })
                parsed = self._validated_call(
                    model=self.map_model,
                    system=REDUCE_SYSTEM_PROMPT,
                    user=json.dumps({
                        "question": query,
                        "expected_chunk_ids": list(expected),
                        "records": [
                            _record_to_dict(record) for record in group
                        ],
                    }, ensure_ascii=False),
                    stage="evidence_reduce",
                    expected_chunk_ids=expected,
                    citation_registry=citation_registry,
                    emit=emit,
                )
                reduced.extend(self._records(parsed))
            reduced_tokens = self.token_counter(json.dumps([
                _record_to_dict(record) for record in reduced
            ], ensure_ascii=False))
            if reduced_tokens >= input_tokens:
                raise EvidenceBatchError(
                    "evidence reduction did not reduce the token volume",
                    (
                        chunk_id
                        for record in records
                        for chunk_id in record.contributing_chunk_ids
                    ),
                )
            records = reduced
        return records

    def _partition_records(
        self, records: list[EvidenceRecord]
    ) -> tuple[tuple[EvidenceRecord, ...], ...]:
        groups = []
        current = []
        used = self.prompt_overhead_tokens
        for record in records:
            size = self.token_counter(json.dumps(
                _record_to_dict(record), ensure_ascii=False
            ))
            # A reduction group must contain at least two records to make
            # progress. The reducer budget is therefore a soft target when
            # individual structured records are unusually large.
            if len(current) >= 2 and used + size > self.batch_tokens:
                groups.append(tuple(current))
                current = []
                used = self.prompt_overhead_tokens
            current.append(record)
            used += size
        if current:
            groups.append(tuple(current))
        return tuple(groups)

    def _unique_citations(
        self, citations: Iterable[CitationRecord]
    ) -> tuple[CitationRecord, ...]:
        unique = {}
        for citation in citations:
            # source_chunk_id is part of the identity: two source chunks of the
            # same paper are distinct provenance, and collapsing them would
            # drop one of the exact references the answer is checked against.
            key = (
                citation.chunk_id,
                citation.source_chunk_id,
                citation.file_path,
                citation.title,
                citation.domains,
            )
            unique.setdefault(key, citation)
        return tuple(unique[key] for key in sorted(unique))

    def _emit_cost(self, emit: Callable[[dict], None]) -> None:
        cost = self.ledger.summary()
        emit({
            "name": "cost_actual",
            "actual_cost_usd": cost.actual_cost_usd,
            "cost_status": cost.cost_status,
            "by_stage": dict(cost.by_stage),
        })

#!/usr/bin/env python3
"""Bulk-load citation-preserving full-text chunks into a stopped LightRAG service.

Run this inside a service whose graph directory is mounted at ``/app/working_dir``.
The loader deliberately updates only the chunk stores: metadata entities and
relationships remain untouched.  It batches embedding writes in memory and calls
LightRAG's final persistence hook exactly once after all batches succeed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any


DEFAULT_WORKING_DIR = Path("/app/working_dir")
PAYLOAD_FILENAME = "_fulltext_payload.json"
DEFAULT_BATCH_SIZE = 150


def stable_chunk_id(content: str) -> str:
    """Return LightRAG's deterministic ``chunk-<md5>`` identifier."""
    return "chunk-" + hashlib.md5(content.encode()).hexdigest()


def batch_items(items: Sequence[dict[str, Any]], size: int) -> Iterator[Sequence[dict[str, Any]]]:
    """Yield bounded batches, rejecting values that could loop indefinitely."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def chunk_entry(chunk: dict[str, Any], tokenizer: Any, processed_status: Any = "processed") -> tuple[str, dict[str, Any]]:
    """Build the text/vector-store row shape used by LightRAG custom inserts."""
    content = chunk["content"]
    source_id = chunk["source_id"]
    return stable_chunk_id(content), {
        "content": content,
        "source_id": source_id,
        "tokens": len(tokenizer.encode(content)),
        "chunk_order_index": chunk.get("chunk_order_index", 0),
        "full_doc_id": source_id,
        "file_path": chunk.get("file_path", "custom_kg"),
        "status": processed_status,
    }


async def load_chunks(rag: Any, chunks: Sequence[dict[str, Any]], batch_size: int = DEFAULT_BATCH_SIZE, *, processed_status: Any = "processed") -> None:
    """Upsert full-text chunks into both LightRAG chunk stores without persisting."""
    for batch in batch_items(chunks, batch_size):
        records = dict(
            chunk_entry(chunk, rag.tokenizer, processed_status)
            for chunk in batch
        )
        await asyncio.gather(
            rag.chunks_vdb.upsert(records),
            rag.text_chunks.upsert(records),
        )


async def load_and_persist(rag: Any, chunks: Sequence[dict[str, Any]], batch_size: int = DEFAULT_BATCH_SIZE, *, processed_status: Any = "processed") -> None:
    """Load all chunks, persist once, then close the initialized LightRAG stores."""
    try:
        await load_chunks(rag, chunks, batch_size, processed_status=processed_status)
        await rag._insert_done()
    finally:
        await rag.finalize_storages()


def read_payload(path: Path) -> list[dict[str, Any]]:
    """Read the host builder's full-text payload, failing clearly on bad shape."""
    payload = json.loads(path.read_text())
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"{path} must contain a list under 'chunks'")
    return chunks


async def build_rag(working_dir: Path) -> tuple[Any, Any]:
    """Create and initialize the LightRAG instance only in the service image."""
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from lightrag.base import DocStatus

    async def embed(texts: list[str]) -> Any:
        return await openai_embed(
            texts,
            model=os.environ["EMBEDDING_MODEL"],
            api_key=os.environ["EMBEDDING_BINDING_API_KEY"],
            base_url=os.environ["EMBEDDING_BINDING_HOST"],
        )

    async def llm(prompt: str, system_prompt: str | None = None, history_messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        return await openai_complete_if_cache(
            os.environ["LLM_MODEL"],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=os.environ["LLM_BINDING_API_KEY"],
            base_url=os.environ["LLM_BINDING_HOST"],
            **kwargs,
        )

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=llm,
        llm_model_name=os.environ.get("LLM_MODEL"),
        default_embedding_timeout=300,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.environ["EMBEDDING_DIM"]),
            max_token_size=8192,
            func=embed,
        ),
    )
    await rag.initialize_storages()
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status
        await initialize_pipeline_status()
    except Exception as exc:  # LightRAG 1.4.16 deployments differ here.
        print(f"(pipeline status init skipped: {exc})")
    return rag, DocStatus.PROCESSED


async def run(working_dir: Path, payload_path: Path, batch_size: int) -> int:
    """Load one mounted-service payload and report its deterministic chunk count."""
    chunks = read_payload(payload_path)
    rag, processed_status = await build_rag(working_dir)
    await load_and_persist(rag, chunks, batch_size, processed_status=processed_status)
    print(f"Loaded {len(chunks)} full-text chunks from {payload_path}")
    return len(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk-load full-text chunks into a stopped LightRAG service")
    parser.add_argument("--working-dir", type=Path, default=DEFAULT_WORKING_DIR, help="Mounted LightRAG working directory (default: /app/working_dir)")
    parser.add_argument("--payload", type=Path, default=None, help="Payload path (default: <working-dir>/_fulltext_payload.json)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Maximum chunks embedded per upsert batch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = args.payload or args.working_dir / PAYLOAD_FILENAME
    asyncio.run(run(args.working_dir, payload, args.batch_size))


if __name__ == "__main__":
    main()

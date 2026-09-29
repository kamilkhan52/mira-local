import asyncio
import hashlib

from scripts import bulk_load_fulltext as loader


class FakeStore:
    def __init__(self):
        self.rows = {}
        self.upserts = []

    async def upsert(self, rows):
        self.upserts.append(dict(rows))
        self.rows.update(rows)


class FakeTokenizer:
    def encode(self, text):
        return text.split()


class FakeRag:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.chunks_vdb = FakeStore()
        self.text_chunks = FakeStore()
        self.metadata_entities = {"Storage Paper": {"entity_type": "Paper"}}
        self.metadata_relationships = {("Storage Paper", "Storage Topic")}
        self.persist_calls = 0
        self.finalize_calls = 0

    async def _insert_done(self):
        self.persist_calls += 1

    async def finalize_storages(self):
        self.finalize_calls += 1


def _chunk(title, arxiv_id, body, order=0):
    url = f"https://arxiv.org/abs/{arxiv_id}"
    content = f"Paper: {title}\nArXiv: {url}\n\n{body}"
    return {
        "content": content,
        "source_id": f"fulltext-{arxiv_id}",
        "file_path": url,
        "chunk_order_index": order,
    }


def test_fulltext_loader_uses_stable_ids_and_preserves_storage_arxiv_urls_on_rerun():
    rag = FakeRag()
    original_entities = dict(rag.metadata_entities)
    original_relationships = set(rag.metadata_relationships)
    chunks = [
        _chunk("Storage Paper", "2401.00001v1", "first page", 0),
        _chunk("Storage Paper", "2401.00001v1", "second page", 1),
    ]

    asyncio.run(loader.load_and_persist(rag, chunks, batch_size=1))
    asyncio.run(loader.load_and_persist(rag, chunks, batch_size=1))

    expected_ids = {
        "chunk-" + hashlib.md5(chunk["content"].encode()).hexdigest()
        for chunk in chunks
    }
    assert set(rag.text_chunks.rows) == expected_ids
    assert set(rag.chunks_vdb.rows) == expected_ids
    assert len(rag.text_chunks.rows) == len(chunks)
    assert {entry["file_path"] for entry in rag.text_chunks.rows.values()} == {
        "https://arxiv.org/abs/2401.00001v1"
    }
    assert {entry["source_id"] for entry in rag.text_chunks.rows.values()} == {
        "fulltext-2401.00001v1"
    }
    assert rag.metadata_entities == original_entities
    assert rag.metadata_relationships == original_relationships
    assert rag.persist_calls == 2
    assert rag.finalize_calls == 2


def test_fulltext_loader_batches_chunk_upserts_before_one_final_persist():
    rag = FakeRag()
    chunks = [
        _chunk("Storage Paper", "2401.00001v1", f"page {number}", number)
        for number in range(3)
    ]

    asyncio.run(loader.load_and_persist(rag, chunks, batch_size=2))

    assert [len(batch) for batch in rag.chunks_vdb.upserts] == [2, 1]
    assert [len(batch) for batch in rag.text_chunks.upserts] == [2, 1]
    assert rag.persist_calls == 1
    assert rag.finalize_calls == 1

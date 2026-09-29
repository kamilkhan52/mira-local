"""Read-only view over NanoVectorDB's vdb_entities.json.

The file stores a base64 float32 matrix, row-aligned with the `data` records
and unit-normalized, so cosine similarity is a plain dot product. Vectors were
produced by LightRAG's embedding model at ingest time — nothing is re-embedded
here (spec §10: reuse existing vectors; SPECTER2 is a deferred upgrade).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable

import numpy as np


class EntityVectors:
    def __init__(self, names: list[str], matrix: np.ndarray):
        self._index = {n: i for i, n in enumerate(names)}
        self._matrix = matrix

    @classmethod
    def load(cls, path: Path) -> "EntityVectors":
        raw = json.loads(Path(path).read_text())
        matrix = np.frombuffer(base64.b64decode(raw["matrix"]), dtype=np.float32)
        matrix = matrix.reshape(len(raw["data"]), raw["embedding_dim"])
        return cls([r["entity_name"] for r in raw["data"]], matrix)

    def centroid(self, names: Iterable[str]) -> np.ndarray | None:
        """Compute the mean vector of entity embeddings.

        Duplicates are ignored; each entity contributes once to the mean.
        """
        rows = sorted({self._index[n] for n in names if n in self._index})
        if not rows:
            return None
        c = self._matrix[rows].mean(axis=0)
        norm = float(np.linalg.norm(c))
        return c / norm if norm else None

    def side_similarity(
        self,
        papers_a: Iterable[str],
        papers_b: Iterable[str],
        fallback_a: str | None = None,
        fallback_b: str | None = None,
    ) -> float:
        ca = self.centroid(papers_a)
        if ca is None and fallback_a is not None:
            ca = self.centroid([fallback_a])
        cb = self.centroid(papers_b)
        if cb is None and fallback_b is not None:
            cb = self.centroid([fallback_b])
        if ca is None or cb is None:
            return 0.0
        return max(float(np.dot(ca, cb)), 0.0)

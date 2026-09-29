"""CLI-side adapters for the graph merge: env-file parsing and the OpenRouter
embedder. Pure merge logic lives in mira/graph_merge.py."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mira.graph_merge import MergeError


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def make_openai_embedder(api_key: str, host: str, model: str,
                         batch_size: int = 100, max_retries: int = 3,
                         transport=None):
    import httpx

    client = httpx.Client(transport=transport, timeout=60.0)
    url = host.rstrip("/") + "/embeddings"

    def embed(texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            for attempt in range(max_retries + 1):
                resp = client.post(url, json={"model": model, "input": batch},
                                   headers={"Authorization": f"Bearer {api_key}"})
                if resp.status_code == 200:
                    data = sorted(resp.json()["data"], key=lambda d: d["index"])
                    rows.extend(d["embedding"] for d in data)
                    break
                if resp.status_code in (429,) or resp.status_code >= 500:
                    if attempt == max_retries:
                        raise MergeError(
                            f"embeddings API failed after retries: "
                            f"{resp.status_code} {resp.text[:200]}")
                    time.sleep(2 ** attempt)
                    continue
                raise MergeError(f"embeddings API error: "
                                 f"{resp.status_code} {resp.text[:200]}")
        return np.asarray(rows, dtype=np.float32)

    return embed

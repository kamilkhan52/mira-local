import base64
import json

import numpy as np

from mira.hypothesis.vectors import EntityVectors


def _write_vdb(path, names, vectors):
    arr = np.asarray(vectors, dtype=np.float32)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    path.write_text(json.dumps({
        "embedding_dim": arr.shape[1],
        "data": [{"entity_name": n} for n in names],
        "matrix": base64.b64encode(arr.tobytes()).decode(),
    }))


def test_load_and_identical_sides_score_one(tmp_path):
    p = tmp_path / "vdb_entities.json"
    _write_vdb(p, ["A", "B"], [[1.0, 0.0], [1.0, 0.0]])
    ev = EntityVectors.load(p)
    assert abs(ev.side_similarity(["A"], ["B"]) - 1.0) < 1e-6


def test_orthogonal_sides_score_zero(tmp_path):
    p = tmp_path / "vdb_entities.json"
    _write_vdb(p, ["A", "B"], [[1.0, 0.0], [0.0, 1.0]])
    ev = EntityVectors.load(p)
    assert abs(ev.side_similarity(["A"], ["B"])) < 1e-6


def test_fallback_entity_used_when_papers_missing(tmp_path):
    p = tmp_path / "vdb_entities.json"
    _write_vdb(p, ["TopicA", "B"], [[1.0, 0.0], [1.0, 0.0]])
    ev = EntityVectors.load(p)
    # No paper named "NoSuchPaper"; falls back to the topic node's own vector.
    sim = ev.side_similarity(["NoSuchPaper"], ["B"], fallback_a="TopicA")
    assert abs(sim - 1.0) < 1e-6


def test_missing_everything_scores_zero(tmp_path):
    p = tmp_path / "vdb_entities.json"
    _write_vdb(p, ["A"], [[1.0, 0.0]])
    ev = EntityVectors.load(p)
    assert ev.side_similarity(["nope"], ["A"]) == 0.0


def test_negative_cosine_clamped_to_zero(tmp_path):
    p = tmp_path / "vdb_entities.json"
    _write_vdb(p, ["A", "B"], [[1.0, 0.0], [-1.0, 0.0]])
    ev = EntityVectors.load(p)
    assert ev.side_similarity(["A"], ["B"]) == 0.0


def test_centroid_averages_multiple_names_and_ignores_duplicates(tmp_path):
    p = tmp_path / "vdb_entities.json"
    _write_vdb(p, ["A", "B", "Q"], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ev = EntityVectors.load(p)
    # Centroid of A and B is the 45° diagonal — identical to Q's direction.
    assert abs(ev.side_similarity(["A", "B"], ["Q"]) - 1.0) < 1e-6
    # Duplicates must not skew the centroid toward the repeated vector.
    assert abs(ev.side_similarity(["A", "A", "A", "B"], ["Q"]) - 1.0) < 1e-6

"""retrieval.py'deki saf matematik mantığı için hızlı birim testleri.

Foundry Local modeli gerektirmez, DB'ye dokunmaz; sabit vektörlerle
cosine similarity hesabının doğruluğunu kontrol eder.
"""

import numpy as np
import pytest

from src.retrieval import _cosine_similarities


def test_identical_vector_has_similarity_one():
    query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    matrix = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    scores = _cosine_similarities(query, matrix)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_vectors_have_similarity_zero():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[0.0, 1.0]], dtype=np.float32)
    scores = _cosine_similarities(query, matrix)
    assert scores[0] == pytest.approx(0.0, abs=1e-5)


def test_opposite_vectors_have_similarity_negative_one():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[-1.0, 0.0]], dtype=np.float32)
    scores = _cosine_similarities(query, matrix)
    assert scores[0] == pytest.approx(-1.0, abs=1e-5)


def test_similarity_is_scale_invariant():
    """Cosine similarity vektörün büyüklüğünden (magnitude) etkilenmemeli."""
    query = np.array([1.0, 1.0], dtype=np.float32)
    matrix_small = np.array([[2.0, 2.0]], dtype=np.float32)
    matrix_large = np.array([[200.0, 200.0]], dtype=np.float32)

    score_small = _cosine_similarities(query, matrix_small)[0]
    score_large = _cosine_similarities(query, matrix_large)[0]
    assert score_small == pytest.approx(score_large, abs=1e-5)
    assert score_small == pytest.approx(1.0, abs=1e-5)


def test_ranks_multiple_vectors_correctly():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array(
        [
            [0.0, 1.0],   # ortogonal -> 0
            [1.0, 0.0],   # aynı yön -> 1
            [-1.0, 0.0],  # zıt yön -> -1
        ],
        dtype=np.float32,
    )
    scores = _cosine_similarities(query, matrix)
    top_index = int(np.argmax(scores))
    assert top_index == 1
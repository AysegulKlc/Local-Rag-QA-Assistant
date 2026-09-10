"""Sorgu -> ilgili chunk'ları bulma (retrieval).

Kullanıcı sorgusunu embedding modeliyle vektörleştirir, SQLite'daki tüm
chunk embedding'leriyle vectorized (numpy) cosine similarity hesaplar
ve en alakalı top-k chunk'ı skorlarıyla birlikte döner.

CLI test modu:
    python -m src.retrieval "soru"
"""

import sys
from dataclasses import dataclass

import numpy as np

from src import config, storage
from src.foundry_client import get_embedding_client

_embedding_client = None


def _get_embedding_client():
    """Embedding client'ı bir kez oluşturup önbellekte tutar. rag.py'deki
    _get_chat_client() ile aynı desen: her sorguda model yeniden
    hazırlanmasın diye (bkz. foundry_client._get_ready_model), tek bir
    process içinde client tekrar tekrar istenmez."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = get_embedding_client()
    return _embedding_client


@dataclass
class Chunk:
    id: int
    source_file: str
    chunk_index: int
    content: str
    score: float


def _fetch_all_chunks() -> tuple[list["storage.ChunkRow"], np.ndarray]:
    conn = storage.connect()
    try:
        rows = storage.fetch_all_chunks(conn)
    finally:
        conn.close()

    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)

    matrix = np.vstack([row.embedding for row in rows])
    return rows, matrix


def _cosine_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_unit = query_vec / np.linalg.norm(query_vec)
    matrix_unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_unit @ query_unit


def get_top_chunks(query: str, k: int = config.TOP_K) -> list[Chunk]:
    """Sorguyu embed eder, DB'deki tüm chunk embedding'leriyle vectorized
    cosine similarity hesaplar ve en alakalı k chunk'ı skorlarıyla
    birlikte döner (büyükten küçüğe sıralı).

    config.MIN_SCORE_THRESHOLD altında kalan chunk'lar sonuca dahil
    edilmez: bunlar sorguyla yalnızca yüzeysel kelime örtüşmesi taşıyan,
    gerçekte alakasız parçalar olma ihtimali yüksek olduğundan, context'e
    girip modelin halüsinasyon yapmasına zemin hazırlamalarını önlemek
    için elenir. Hiçbir chunk eşiği geçemezse boş liste döner (rag.py bu
    durumda NO_ANSWER_TEXT ile cevap verir).
    """
    embedding_client = _get_embedding_client()
    response = embedding_client.generate_embedding(query)
    query_vec = np.asarray(response.data[0].embedding, dtype=np.float32)

    rows, matrix = _fetch_all_chunks()

    if not rows:
        return []

    scores = _cosine_similarities(query_vec, matrix)
    top_indices = np.argsort(-scores)[:k]

    return [
        Chunk(
            id=rows[i].id,
            source_file=rows[i].source_file,
            chunk_index=rows[i].chunk_index,
            content=rows[i].content,
            score=float(scores[i]),
        )
        for i in top_indices
        if scores[i] >= config.MIN_SCORE_THRESHOLD
    ]


def main() -> None:
    if len(sys.argv) < 2:
        print('Kullanim: python -m src.retrieval "soru"')
        return

    query = sys.argv[1]
    chunks = get_top_chunks(query)

    if not chunks:
        print("Sonuc bulunamadi. Once 'python -m src.ingest' calistirin.")
        return

    for rank, chunk in enumerate(chunks, start=1):
        print(f"[{rank}] score={chunk.score:.4f}  {chunk.source_file} (chunk #{chunk.chunk_index})")
        print(f"    {chunk.content[:200]}")
        print()


if __name__ == "__main__":
    main()

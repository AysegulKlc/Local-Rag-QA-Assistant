"""SQLite depolama katmanı.

Sorumluluk:
- Şema oluşturma (chunks tablosu).
- Chunk + embedding kaydetme (replace_document_chunks).
- Tüm chunk'ları ve embedding'lerini okuma (fetch_all_chunks).
- Embedding vektörünün BLOB <-> numpy array dönüşümü.

ingest.py ve retrieval.py artık kendi _connect()/CREATE TABLE kopyalarını
tutmuyor; şema ve erişim tek buradan yönetilir.
"""

import sqlite3
from dataclasses import dataclass

import numpy as np

from src import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL
)
"""


@dataclass
class ChunkRow:
    id: int
    source_file: str
    chunk_index: int
    content: str
    embedding: np.ndarray


def connect() -> sqlite3.Connection:
    """DB dosyasının bulunduğu dizini oluşturur, bağlantıyı açar ve
    şemanın var olduğundan emin olur."""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(_SCHEMA)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_file ON chunks(source_file)")
    return conn


def replace_document_chunks(
    conn: sqlite3.Connection,
    source_file: str,
    chunks: list[str],
    vectors: list[np.ndarray],
) -> None:
    """source_file'a ait eski kayıtları silip yenilerini yazar (idempotent
    ingestion). Çağıran commit etmekten sorumludur."""
    conn.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
    conn.executemany(
        "INSERT INTO chunks (source_file, chunk_index, content, embedding) "
        "VALUES (?, ?, ?, ?)",
        [
            (source_file, i, chunk, vector.astype(np.float32).tobytes())
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ],
    )


def fetch_all_chunks(conn: sqlite3.Connection) -> list[ChunkRow]:
    """Tüm chunk'ları embedding'leriyle birlikte döner (retrieval.py'nin
    cosine similarity karşılaştırması için)."""
    rows = conn.execute(
        "SELECT id, source_file, chunk_index, content, embedding FROM chunks"
    ).fetchall()
    return [
        ChunkRow(
            id=row[0],
            source_file=row[1],
            chunk_index=row[2],
            content=row[3],
            embedding=np.frombuffer(row[4], dtype=np.float32),
        )
        for row in rows
    ]

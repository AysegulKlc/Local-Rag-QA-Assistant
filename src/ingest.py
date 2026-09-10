"""Doküman ingestion pipeline.

data/docs altındaki .txt ve .md dosyalarını okur, paragraf bazlı
chunk'lara böler (~config.CHUNK_SIZE karakter, config.CHUNK_OVERLAP
örtüşme), her chunk için Foundry Local ile embedding üretir ve
SQLite'a (chunks tablosu) yazar.

Idempotent: bir dosya tekrar işlendiğinde o dosyaya ait eski kayıtlar
silinip yeniden yazılır, aynı dosyadan çift kayıt oluşmaz.

Çalıştırma:
    python -m src.ingest
"""

import re
from pathlib import Path

import numpy as np

from src import config, storage
from src.foundry_client import get_embedding_client

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _slide_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({overlap}) CHUNK_SIZE'dan ({chunk_size}) küçük olmalı, "
            "aksi halde adım boyu 1'e düşer ve chunk sayısı patlar."
        )
    step = max(chunk_size - overlap, 1)
    pieces = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return pieces


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Paragraf sınırlarını koruyarak ~chunk_size karakterlik, overlap kadar
    örtüşen parçalara böler. Tek bir paragraf chunk_size'dan uzunsa, o
    paragraf kayan pencereyle karakter bazlı alt parçalara bölünür.
    """
    chunks: list[str] = []
    current = ""

    for para in _split_paragraphs(text):
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_slide_window(para, chunk_size, overlap))
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    return chunks


def _iter_documents(docs_dir: Path):
    for path in sorted(docs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def ingest_documents(docs_dir: Path = config.DOCUMENTS_DIR) -> int:
    """docs_dir altındaki desteklenen dosyaları chunk'lar, embedding üretir
    ve SQLite'a yazar. Yazılan toplam chunk sayısını döner."""
    conn = storage.connect()
    embedding_client = get_embedding_client()
    total_chunks = 0

    try:
        for path in _iter_documents(docs_dir):
            text = path.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            if not chunks:
                print(f"{path.name}: chunk üretilemedi (boş dosya), atlandı")
                continue

            response = embedding_client.generate_embeddings(chunks)
            vectors = [
                np.asarray(item.embedding, dtype=np.float32) for item in response.data
            ]

            source_file = path.name
            storage.replace_document_chunks(conn, source_file, chunks, vectors)
            conn.commit()

            total_chunks += len(chunks)
            print(f"{source_file}: {len(chunks)} chunk yazıldı")
    finally:
        conn.close()

    return total_chunks


def main() -> None:
    total = ingest_documents()
    print(f"\nTamamlandı: toplam {total} chunk SQLite'a yazıldı ({config.DB_PATH})")


if __name__ == "__main__":
    main()

"""Uçtan uca RAG cevaplama.

retrieval.py'den gelen top-K chunk'ları [Kaynak: dosya] etiketleriyle
bağlam olarak biçimlendirir, Foundry Local chat modeline gönderir ve
cevabı, kullanılan kaynak dosya adlarını ve yanıt süresini döner.

Çok turlu (multi-turn) sohbet, çağıranın tuttuğu `history` listesini
(sırayla {"role": "user"/"assistant", "content": ...} mesajları) her
çağrıya iletmesiyle desteklenir; fonksiyonun kendisi durum tutmaz.
"""

import time

from src import config
from src.foundry_client import get_chat_client
from src.retrieval import Chunk, get_top_chunks

_chat_client = None


def _get_chat_client():
    global _chat_client
    if _chat_client is None:
        _chat_client = get_chat_client()
    return _chat_client


def _format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[Kaynak: {c.source_file}]\n{c.content}" for c in chunks)


def answer_query(question: str, history: list[dict] | None = None) -> dict:
    """Soruyu retrieval + generation zinciriyle cevaplar.

    history: önceki turların düz {"role", "content"} mesaj listesi (varsa).
    Sadece son config.MAX_HISTORY_TURNS tur kullanılır; retrieval her
    zaman güncel soruya göre yeniden yapılır.

    Döner: {"answer": str, "sources": list[str], "chunks": list[Chunk],
            "latency": float (saniye)}
    "chunks", cevabı üretmek için kullanılan top-K parçalardır; çağıran
    bunları (örn. debug gösterimi için) retrieval'ı tekrar çalıştırmadan
    kullanabilir.
    """
    start = time.perf_counter()

    question = question.strip()
    if not question:
        return {
            "answer": "Lütfen bir soru girin.",
            "sources": [],
            "chunks": [],
            "latency": time.perf_counter() - start,
        }

    chunks = get_top_chunks(question)

    if not chunks:
        return {
            "answer": config.NO_ANSWER_TEXT,
            "sources": [],
            "chunks": [],
            "latency": time.perf_counter() - start,
        }

    recent_history = (history or [])[-2 * config.MAX_HISTORY_TURNS :]

    context = _format_context(chunks)
    messages = (
        [{"role": "system", "content": config.SYSTEM_PROMPT}]
        + recent_history
        + [{"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"}]
    )

    chat_client = _get_chat_client()
    response = chat_client.complete_chat(messages)
    answer = response.choices[0].message.content

    sources = list(dict.fromkeys(chunk.source_file for chunk in chunks))

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
        "latency": time.perf_counter() - start,
    }

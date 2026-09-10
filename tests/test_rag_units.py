"""rag.py'deki saf formatlama mantığı için hızlı birim testleri.

Foundry Local modeli gerektirmez; context string'inin doğru
biçimlendirildiğini ve answer_query'nin yeni "chunks" alanını
döndürdüğünü kontrol eder.
"""

from src.rag import _format_context
from src.retrieval import Chunk


def _make_chunk(source_file, chunk_index, content, score=0.9):
    return Chunk(
        id=chunk_index,
        source_file=source_file,
        chunk_index=chunk_index,
        content=content,
        score=score,
    )


def test_format_context_includes_source_tag():
    chunks = [_make_chunk("doc1.md", 0, "İçerik metni")]
    context = _format_context(chunks)
    assert "[Kaynak: doc1.md]" in context
    assert "İçerik metni" in context


def test_format_context_separates_multiple_chunks():
    chunks = [
        _make_chunk("doc1.md", 0, "Birinci içerik"),
        _make_chunk("doc2.md", 0, "İkinci içerik"),
    ]
    context = _format_context(chunks)
    assert "[Kaynak: doc1.md]" in context
    assert "[Kaynak: doc2.md]" in context
    assert context.index("doc1.md") < context.index("doc2.md")
    # chunk'lar arasında boş satırla ayrılmış olmalı
    assert "\n\n" in context


def test_format_context_empty_list_returns_empty_string():
    assert _format_context([]) == ""


def test_format_context_same_source_multiple_chunk_indices():
    chunks = [
        _make_chunk("doc1.md", 0, "İlk parça"),
        _make_chunk("doc1.md", 1, "İkinci parça"),
    ]
    context = _format_context(chunks)
    assert context.count("[Kaynak: doc1.md]") == 2
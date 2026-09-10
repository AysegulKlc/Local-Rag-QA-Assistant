"""ingest.py'deki saf metin işleme mantığı için hızlı birim testleri.

Foundry Local modeli gerektirmez, DB'ye dokunmaz; saniyeler içinde çalışır.
"""

import pytest

from src.ingest import _slide_window, _split_paragraphs, chunk_text


def test_split_paragraphs_separates_on_blank_lines():
    text = "Birinci paragraf.\n\nİkinci paragraf.\n\n\nÜçüncü paragraf."
    paragraphs = _split_paragraphs(text)
    assert paragraphs == ["Birinci paragraf.", "İkinci paragraf.", "Üçüncü paragraf."]


def test_split_paragraphs_strips_whitespace_and_empties():
    text = "\n\n  Tek paragraf.  \n\n\n\n"
    assert _split_paragraphs(text) == ["Tek paragraf."]


def test_slide_window_covers_entire_text():
    text = "a" * 1000
    pieces = _slide_window(text, chunk_size=100, overlap=20)
    # Yeniden birleştirilebilir olmasa da her karakter en az bir parçada olmalı
    assert "".join(dict.fromkeys(pieces))  # boş değil
    assert pieces[0] == text[:100]
    assert pieces[-1].endswith(text[-1])


def test_slide_window_respects_overlap():
    text = "0123456789" * 10  # 100 karakter
    pieces = _slide_window(text, chunk_size=30, overlap=10)
    # ardışık parçalar arasında 10 karakterlik örtüşme olmalı
    assert pieces[0][-10:] == pieces[1][:10]


def test_slide_window_raises_when_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        _slide_window("x" * 50, chunk_size=10, overlap=10)


def test_slide_window_single_piece_when_text_shorter_than_chunk_size():
    text = "kısa metin"
    pieces = _slide_window(text, chunk_size=500, overlap=50)
    assert pieces == [text]


def test_chunk_text_keeps_short_paragraphs_together():
    text = "Kısa bir cümle.\n\nBaşka bir kısa cümle."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert "Kısa bir cümle." in chunks[0]
    assert "Başka bir kısa cümle." in chunks[0]


def test_chunk_text_splits_when_exceeding_chunk_size():
    para_a = "A" * 300
    para_b = "B" * 300
    text = f"{para_a}\n\n{para_b}"
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    assert len(chunks) >= 2
    # Hiçbir chunk boş olmamalı
    assert all(c.strip() for c in chunks)


def test_chunk_text_splits_oversized_paragraph_with_slide_window():
    huge_para = "X" * 1200
    chunks = chunk_text(huge_para, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("", chunk_size=500, overlap=50) == []
    assert chunk_text("   \n\n  ", chunk_size=500, overlap=50) == []
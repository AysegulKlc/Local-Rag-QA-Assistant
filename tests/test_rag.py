"""answer_query() için pytest testleri.

Gerçek Foundry Local modelleriyle çalışır (mock yok); data/docs altındaki
örnek dokümanların önceden `python -m src.ingest` ile işlenmiş olması
gerekir. Küçük bir yerel model serbest metin ürettiğinden tam eşleşme
yerine anahtar kelime kontrolü yapılır.
"""

import pytest

from src.rag import answer_query

ANSWERABLE_CASES = [
    ("Foundry Local'i hangi şirket geliştirmiştir?", ["microsoft"]),
    (
        "Foundry Local donanım tespitinde CPU dışında hangi donanım türlerini destekler?",
        ["gpu", "npu"],
    ),
    (
        "RAG'de en alakalı chunk'lar hangi benzerlik ölçütüyle seçilir?",
        ["cosine"],
    ),
    (
        "Foundry Local; metin gömme ve chat completion dışında hangi ses özelliğini destekler?",
        ["transcription", "yazıya"],
    ),
    (
        "RAG sisteminde bilgi tabanını güncellemek için hangi adım tekrar çalıştırılır?",
        ["ingestion"],
    ),
]

UNANSWERABLE_QUESTIONS = [
    "Bu projenin geliştiricisinin en sevdiği müzik türü nedir?",
    "En sevdiğin renk hangisi?",
    "2030 yılında borsa endeksleri nasıl olacak?",
]


@pytest.mark.parametrize("question,keywords", ANSWERABLE_CASES)
def test_answerable_question_contains_keyword(question, keywords):
    result = answer_query(question)
    answer_lower = result["answer"].lower()

    assert any(kw in answer_lower for kw in keywords), (
        f"'{question}' cevabında beklenen anahtar kelimelerden hiçbiri yok: {keywords}\n"
        f"Cevap: {result['answer']}"
    )
    assert result["sources"], "Cevaplanabilir soru için kaynak listesi boş olmamalı"


@pytest.mark.parametrize("question", UNANSWERABLE_QUESTIONS)
def test_unanswerable_question_returns_fallback(question):
    # Küçük yerel model, config.NO_ANSWER_TEXT'i harfiyen tekrarlamayabilir
    # (ör. "bilgi" yerine "bilgileri" diyebilir); bu yüzden anlamı taşıyan
    # çekirdek ifadeyi kontrol ediyoruz, tam cümle eşleşmesini değil.
    result = answer_query(question)
    assert "dokümanlarımda yok" in result["answer"].lower()


def test_empty_query_returns_gracefully():
    result = answer_query("")
    assert result["answer"]
    assert result["sources"] == []
    assert result["latency"] >= 0


def test_whitespace_only_query_returns_gracefully():
    result = answer_query("   ")
    assert result["answer"]
    assert result["sources"] == []

"""MC_RAG kaynak paketi.

Bu paket, offline RAG (Retrieval-Augmented Generation) asistanının tüm
çekirdek mantığını barındırır:

- config.py         : merkezi konfigürasyon (model isimleri, chunk
                       parametreleri, prompt, eşik değerleri vb.)
- foundry_client.py : Foundry Local SDK sarmalayıcısı (embedding/chat
                       client'larını hazırlar)
- storage.py        : SQLite depolama katmanı (şema, chunk okuma/yazma)
- ingest.py         : doküman -> chunk -> embedding -> SQLite pipeline'ı
- retrieval.py      : sorgu -> embedding -> cosine similarity -> top-K
                       chunk bulma
- rag.py            : retrieval + generation'ı birleştiren uçtan uca
                       cevaplama fonksiyonu (answer_query)

Bu dosyanın kendisi paket seviyesinde ekstra bir başlatma kodu içermez;
diğer modüller birbirini `from src import config` gibi doğrudan modül
importlarıyla kullanır.
"""
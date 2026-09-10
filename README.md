# Offline Yerel RAG Q&A Asistanı

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)
![Offline](https://img.shields.io/badge/internet-gerekmez-lightgrey)

Microsoft Foundry Local üzerinde, internet bağlantısı olmadan tamamen
yerelde çalışan bir Retrieval-Augmented Generation (RAG) soru-cevap
asistanı.

> Durum: ingestion, retrieval, generation, CLI, Streamlit arayüzü ve
> pytest testleri (birim + entegrasyon) gerçek Foundry Local modelleriyle
> çalışır durumda.

**İçindekiler:** [Mimari](#mimari) · [Proje Yapısı](#proje-yapısı) ·
[Kurulum](#kurulum) · [Kullanım](#kullanım) · [Yapılandırma](#yapılandırma) ·
[Bilinen Sınırlamalar](#bilinen-sınırlamalar)

## Mimari

```
Doküman(lar) --chunk--> Embedding (qwen3-embedding-0.6b) --> SQLite (storage.py)
                                                                 |
Soru --embed--> cosine similarity (top-K) --> eşik filtresi <----
   |            (MIN_SCORE_THRESHOLD)
   v
Bağlam + Soru --> Chat modeli (phi-3.5-mini, Foundry Local) --> Cevap
```

- **Ingestion** (`src/ingest.py`): Dokümanları oku, paragraf sınırlarını
  koruyarak chunk'lara böl, her chunk için embedding üret, SQLite'a yaz.
  Idempotent — bir dosya tekrar işlendiğinde eski kayıtları silip
  yeniden yazar.
- **Depolama** (`src/storage.py`): SQLite şema yönetimi ve chunk
  okuma/yazma işlemleri tek bu modülden geçer; `ingest.py` ve
  `retrieval.py` doğrudan SQL çalıştırmaz.
- **Retrieval** (`src/retrieval.py`): Sorguyu embed et, kayıtlı chunk
  embedding'leriyle cosine similarity hesapla, en alakalı top-K chunk'ı
  getir. `config.MIN_SCORE_THRESHOLD` altında kalan chunk'lar elenir —
  bu, sorguyla yalnızca yüzeysel kelime örtüşmesi olan ama gerçekte
  alakasız chunk'ların modele "bir şeyler var" izlenimi vererek
  halüsinasyona yol açmasını önler. Hiçbir chunk eşiği geçemezse boş
  liste döner ve `rag.py` doğrudan "Bu bilgi dokümanlarımda yok." cevabı
  verir (LLM'e hiç gidilmez).
- **Generation** (`src/rag.py`): Bulunan chunk'ları `[Kaynak: dosya]`
  etiketleriyle bağlam olarak biçimlendirir, Foundry Local'deki chat
  modeline gönderir, cevabı + kaynak dosya adlarını + yanıt süresini
  döner. Çok turlu (multi-turn) sohbeti `history` parametresiyle
  destekler; fonksiyonun kendisi state tutmaz.
- **Arayüz**: CLI (`app_cli.py`) ve Streamlit (`app_streamlit.py`).
- <img width="955" height="500" alt="image" src="https://github.com/user-attachments/assets/2266ca5b-e5bb-473d-b8e6-d57ae7f7c6bf" />


## Proje Yapısı

Bkz. [src/](src/) altındaki modüllerin başındaki docstring'ler — her
modülün tek sorumluluğu orada açıklanır.

```
src/
├── config.py          # merkezi konfigürasyon (tüm ayarlanabilir değerler)
├── foundry_client.py  # Foundry Local SDK sarmalayıcısı
├── storage.py         # SQLite depolama katmanı
├── ingest.py          # doküman -> chunk -> embedding -> SQLite
├── retrieval.py       # sorgu -> embedding -> cosine similarity -> top-K
└── rag.py             # retrieval + generation birleşimi (answer_query)

tests/
├── test_ingest_units.py     # chunk_text/_slide_window birim testleri
├── test_retrieval_units.py  # cosine similarity birim testleri
├── test_rag_units.py        # _format_context birim testleri
└── test_rag.py               # answer_query entegrasyon testleri (gerçek model)
```

## Kurulum

[Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/)
SDK'sı kendi kendine yeterlidir (self-contained); ayrıca bir Foundry Local
CLI/runtime kurulumu gerekmez. İlk çalıştırmada `qwen3-embedding-0.6b` ve
`phi-3.5-mini` modelleri otomatik olarak Foundry Model Catalog'dan indirilip
yerel önbelleğe (`foundry_local_data/`) alınır; sonraki çalıştırmalar tamamen
offline'dır.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

1. `data/docs/` altına `.txt`/`.md` dokümanlarınızı koyun.
2. Dokümanları işleyip SQLite'a yazın (idempotent — dosya değişince tekrar çalıştırılabilir):
   ```bash
   python -m src.ingest
   ```
3. Sorgulayın:
   ```bash
   python app_cli.py            # etkileşimli CLI, /quit ile çık
   python app_cli.py --debug    # cevapla birlikte kullanılan chunk'ları da göster
   python -m streamlit run app_streamlit.py
   ```
   `python -m streamlit run` kullanımı önerilir (özellikle Windows'ta):
   `streamlit run` komutu bazı ortamlarda venv dışındaki bir Python
   yorumlayıcısını çağırıp `ModuleNotFoundError` üretebiliyor;
   `python -m streamlit run` her zaman aktif venv'i kullanır.

   Tek seferlik retrieval testi için:
   ```bash
   python -m src.retrieval "sorunuz"
   ```

### Testler

İki katman halinde test edilir:

**Hızlı birim testleri** (model gerektirmez, mock yok — saf mantık
testleri, saniyeler içinde biter):
```bash
python -m pytest tests/test_ingest_units.py tests/test_retrieval_units.py tests/test_rag_units.py -v
```

**Entegrasyon testleri** (gerçek Foundry Local modelleriyle çalışır,
`data/docs/` içeriğinin önceden `python -m src.ingest` ile işlenmiş
olmasını gerektirir, birkaç dakika sürebilir):
```bash
python -m pytest tests/test_rag.py -v
```

Hepsini birlikte çalıştırmak için:
```bash
python -m pytest tests/ -v
```

## Yapılandırma

Tüm merkezi parametreler [src/config.py](src/config.py) içinde:

- `EMBEDDING_MODEL_ALIAS`, `CHAT_MODEL_ALIAS` — Foundry Local model
  alias'ları.
- `CHUNK_SIZE`, `CHUNK_OVERLAP` — ingestion sırasında chunk'lama
  parametreleri.
- `TOP_K` — retrieval'da getirilecek chunk sayısı.
- `MIN_SCORE_THRESHOLD` — bu cosine similarity skorunun altında kalan
  chunk'lar context'e hiç dahil edilmez (halüsinasyon önleme).
  qwen3-embedding-0.6b ile bu projedeki dokümanlar üzerinde ölçüldü:
  alakalı sorgularda chunk skorları ~0.58–0.75 aralığında, alakasız
  sorgularda en yüksek skor ~0.44'te kalıyor; `0.50` bu iki grup
  arasındaki güvenli sınır. Farklı/yeni dokümanlar eklendiğinde
  `python -m src.retrieval "sorgu"` ile gerçek skorlar tekrar
  gözlemlenip gerekirse bu değer yeniden kalibre edilmelidir.
- `SYSTEM_PROMPT` — modelin uyacağı kurallar (yalnızca bağlamdaki
  bilgiyi kullanma, kaynak uydurmama, düz paragraf formatı vb.).
- `DB_PATH`, `DOCUMENTS_DIR` — SQLite ve doküman kaynağı yolları.

## Bilinen Sınırlamalar

- `phi-3.5-mini` (3.8B parametre) küçük bir model olduğu için, özellikle
  iki kavramı karşılaştırma gerektiren sorularda bazen gramatik olarak
  eksik cümleler üretebiliyor. Bu durum genel bir talimat-takip
  kapasitesi sınırıdır, sadece prompt mühendisliğiyle tam çözülemez.
- Streamlit'in dosya izleme (file watcher) mekanizması, uzun süren
  senkron Foundry Local çağrılarını bazen "Operation was cancelled"
  hatasıyla kesebiliyor. `.streamlit/config.toml`'daki
  `fileWatcherType = "none"` ayarı bu sorunu büyük ölçüde önler;
  `app_streamlit.py` ayrıca hata durumunda otomatik bir kez yeniden
  dener.

## Katkıda Bulunma

Bu proje şu an bireysel bir öğrenim/ders projesi olarak geliştiriliyor.
Yine de hata bildirimi, öneri veya pull request'ler memnuniyetle
karşılanır:

1. Depoyu fork'layın.
2. Değişikliğiniz için yeni bir branch açın (`git checkout -b ozellik/aciklama`).
3. `python -m pytest tests/ -v` ile testlerin geçtiğinden emin olun.
4. Pull request açın.


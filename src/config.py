"""Merkezi konfigürasyon.

Tüm modüller model isimlerini, chunk parametrelerini, dizin yollarını
ve Foundry Local ayarlarını buradan okur. Değer değiştirmek gerektiğinde
tek dokunulacak yer burasıdır.
"""

from pathlib import Path

# --- Proje kök dizini ---
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Foundry Local SDK ayarları ---
# Configuration(app_name=..., model_cache_dir=...) için kullanılır.
APP_NAME = "mc_rag"
MODEL_CACHE_DIR = str(BASE_DIR / "foundry_local_data" / "model_cache")

# Model catalog alias'ları -> manager.catalog.get_model(alias)
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"
CHAT_MODEL_ALIAS = "phi-3.5-mini"

# --- Chunking parametreleri ---
CHUNK_SIZE = 500       # karakter cinsinden hedef chunk boyutu
CHUNK_OVERLAP = 50     # ardışık chunk'lar arasındaki örtüşme

# --- Retrieval parametreleri ---
TOP_K = 5              # cosine similarity ile getirilecek chunk sayısı
# Bu skorun altındaki chunk'lar, sorguyla yeterince alakalı sayılmaz ve
# context'e hiç dahil edilmez. Amaç: sadece yüzeysel kelime örtüşmesi olan
# ama gerçekte alakasız chunk'ların modele "bir şeyler var" izlenimi
# vererek halüsinasyona yol açmasını önlemek. 0.0-1.0 arası cosine skoru;
# qwen3-embedding-0.6b ile projedeki dokümanlar üzerinde ölçüldü:
# gerçekten alakalı sorgularda chunk skorları ~0.58-0.75 aralığında,
# alakasız sorgularda (yüzeysel kelime örtüşmesi olsa bile) en yüksek
# skor ~0.44'te kalıyor. 0.50 bu iki grup arasındaki güvenli sınır.
MIN_SCORE_THRESHOLD = 0.50

# --- Depolama ---
DB_PATH = BASE_DIR / "db" / "rag.db"

# --- Doküman kaynağı ---
DOCUMENTS_DIR = BASE_DIR / "data" / "docs"

# --- Generation parametreleri ---
NO_ANSWER_TEXT = "Bu bilgi dokümanlarımda yok."
SYSTEM_PROMPT = (
    "Sen bir soru-cevap asistanısın. Sana '[Kaynak: dosya_adı]' etiketleriyle "
    "işaretlenmiş bağlam parçaları verilecek.\n\n"
    "Kurallar:\n"
    "1. Sadece verilen bağlamdaki bilgiyi kullan. Bağlam yetersizse veya "
    f'alakasızsa şunu yaz: "{NO_ANSWER_TEXT}"\n'
    "2. Cevabı düz paragraf halinde, tam ve akıcı cümlelerle yaz. Liste, "
    "madde işareti veya numara kullanma.\n"
    "3. Kaynak adı, dosya adı veya not ekleme; sadece cevabın kendisini yaz.\n"
)
MAX_TOKENS = 512
TEMPERATURE = 0.2

# --- Sohbet geçmişi ---
# Prompt'un şişip yavaşlamaması için hafızada tutulan soru-cevap tur sayısı.
MAX_HISTORY_TURNS = 3

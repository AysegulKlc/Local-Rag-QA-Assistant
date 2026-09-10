"""Foundry Local SDK sarmalayıcısı.

FoundryLocalManager singleton'ının başlatılması, modelin catalogdan
bulunup indirilmesi/yüklenmesi gibi tekrar eden SDK detaylarını burada
topluyoruz; ingestion ve retrieval/generation modülleri sadece hazır
bir client istiyor.
"""

from foundry_local_sdk import Configuration, FoundryLocalManager

from src import config

_manager: FoundryLocalManager | None = None


def get_manager() -> FoundryLocalManager:
    """FoundryLocalManager singleton'ını (gerekirse) başlatıp döner."""
    global _manager
    if _manager is None:
        cfg = Configuration(app_name=config.APP_NAME, model_cache_dir=config.MODEL_CACHE_DIR)
        FoundryLocalManager.initialize(cfg)
        _manager = FoundryLocalManager.instance
    return _manager


def _get_ready_model(alias: str):
    """alias'a karşılık gelen modeli catalogdan bulur, gerekirse indirir
    ve yükler, kullanıma hazır model nesnesini döner."""
    manager = get_manager()
    model = manager.catalog.get_model(alias)

    if not model.is_cached:
        print(f"{alias} indiriliyor...")
        model.download(
            lambda progress: print(f"\r  {progress:.1f}%", end="", flush=True)
        )
        print()

    if not model.is_loaded:
        print(f"{alias} yükleniyor...")
        model.load()

    return model


def get_embedding_client():
    """Embedding modelini hazırlayıp embedding client döner."""
    model = _get_ready_model(config.EMBEDDING_MODEL_ALIAS)
    return model.get_embedding_client()


def get_chat_client():
    """Chat modelini hazırlayıp chat client döner."""
    model = _get_ready_model(config.CHAT_MODEL_ALIAS)
    return model.get_chat_client()

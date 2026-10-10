from typing import Any, Dict, Optional
from src.models.base_adapter import BaseTTSAdapter
from src.models.f5_adapter import F5TTSAdapter


def get_tts_model(model_name: str = "f5-tts", config: Optional[Dict[str, Any]] = None) -> BaseTTSAdapter:
    """Factory creating appropriate TTS adapter instance.
    Enables future plug-and-play addition of new models without modifying pipeline logic.
    """
    key = model_name.lower().strip()
    if key == "f5-tts":
        return F5TTSAdapter(config=config)
    else:
        from src.models.downloader import MODEL_REGISTRY
        raise ValueError(
            f"Unsupported TTS model '{model_name}'. Available in registry: {list(MODEL_REGISTRY.keys())}"
        )


def ensure_base_model_cached(*args, **kwargs):
    from src.models.downloader import ensure_base_model_cached as _ensure
    return _ensure(*args, **kwargs)


def __getattr__(name: str):
    if name == "MODEL_REGISTRY":
        from src.models.downloader import MODEL_REGISTRY
        return MODEL_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseTTSAdapter",
    "F5TTSAdapter",
    "get_tts_model",
    "ensure_base_model_cached",
    "MODEL_REGISTRY",
]

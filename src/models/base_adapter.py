from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import os


class BaseTTSAdapter(ABC):
    """Abstract interface defining standard contract for all TTS base backbones.
    Enables plug-and-play swapping between F5-TTS, CosyVoice, Piper, etc.
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}

    @abstractmethod
    def load_base_weights(self, weights_path: str) -> None:
        """Loads pretrained base model weights."""
        pass

    @abstractmethod
    def build_lora_model(self, lora_config: Dict[str, Any], **kwargs) -> Any:
        """Wraps base model with trainable LoRA parameters."""
        pass

    @abstractmethod
    def export_onnx(
        self,
        checkpoint_path: str,
        output_onnx_path: str,
        opset_version: int = 17,
    ) -> str:
        """Exports merged backbone into standard ONNX graph format."""
        pass

    @abstractmethod
    def synthesize_speech(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        output_wav_path: str = "output.wav",
    ) -> Dict[str, Any]:
        """Runs end-to-end inference producing 24kHz studio waveform."""
        pass

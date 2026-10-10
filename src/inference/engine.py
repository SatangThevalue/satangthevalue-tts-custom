import argparse
import os
from pathlib import Path
import time
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf

from src.audio.mastering import apply_studio_mastering
from src.g2p.text_norm import text_to_phonemes
from src.inference.f5_infer import synthesize_f5
from src.utils.logger import setup_logger

logger = setup_logger("inference_engine")


class VoiceInferenceEngine:
    """Production Real-Time Voice Synthesis Engine.
    Leverages F5-TTS Flow Matching Diffusion with Vocos Vocoder and Studio DSP mastering.
    """

    def __init__(
        self,
        model_path: str = "/content/drive/MyDrive/tts-project/04_onnx_exports/model.onnx",
        hidden_dim: int = 512,
        use_gpu: bool = False,
    ):
        self.model_path = model_path
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.session = None

        logger.info(f"Initializing Voice Inference Engine -> Model: {model_path} (GPU: {use_gpu})")

        if os.path.exists(model_path) and model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if use_gpu
                    else ["CPUExecutionProvider"]
                )
                self.session = ort.InferenceSession(model_path, providers=providers)
                logger.info(f"Loaded ONNX model session successfully with providers: {providers}")
            except Exception as e:
                logger.warning(f"Could not load ONNX model ({e}). Engine will use PyTorch F5-TTS inference.")

    def synthesize(
        self,
        text: str,
        ref_audio_path: Optional[str] = None,
        output_wav_path: str = "output.wav",
        enable_mastering: bool = True,
        speed_factor: float = 1.0,
        pitch_semitones: float = 0.0,
        warmth_drive: float = 1.15,
        room_reverb_wet: float = 0.05,
        deess_gain_db: float = -2.5,
    ) -> Dict[str, Any]:
        """Synthesizes natural, human speech with high prosodic fidelity."""
        t0 = time.perf_counter()
        logger.info(f"Synthesizing text: '{text}' (Length: {len(text)} chars)")

        final_wav = synthesize_f5(
            text=text,
            speaker_id="satang",
            ref_audio_path=ref_audio_path,
            output_wav_path=output_wav_path,
            speed_factor=speed_factor,
            pitch_semitones=pitch_semitones,
            warmth_drive=warmth_drive,
            room_reverb_wet=room_reverb_wet,
            deess_gain_db=deess_gain_db,
            enable_mastering=enable_mastering,
        )

        total_elapsed = time.perf_counter() - t0
        dur = 0.0
        if os.path.exists(final_wav):
            info = sf.info(final_wav)
            dur = info.duration

        rtf = total_elapsed / dur if dur > 0 else 0.0
        logger.info(
            f"Synthesis complete! Generated {dur:.2f}s audio in {total_elapsed:.2f}s (RTF: {rtf:.3f}) -> {final_wav}"
        )

        return {
            "output_path": final_wav,
            "duration": dur,
            "duration_sec": dur,
            "latency_sec": total_elapsed,
            "elapsed_sec": total_elapsed,
            "rtf": rtf,
            "text": text,
            "phonemes": text_to_phonemes(text),
        }


# Backwards compatibility alias
TTSEngine = VoiceInferenceEngine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Engine Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="/content/drive/MyDrive/tts-project/04_onnx_exports/model.onnx",
    )
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--output", type=str, default="synthesized_output.wav")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--warmth", type=float, default=1.15)
    parser.add_argument("--reverb", type=float, default=0.05)
    parser.add_argument("--deess", type=float, default=-2.5)
    args = parser.parse_args()

    engine = VoiceInferenceEngine(model_path=args.model)
    engine.synthesize(
        text=args.text,
        ref_audio_path=args.ref_audio,
        output_wav_path=args.output,
        speed_factor=args.speed,
        pitch_semitones=args.pitch,
        warmth_drive=args.warmth,
        room_reverb_wet=args.reverb,
        deess_gain_db=args.deess,
    )

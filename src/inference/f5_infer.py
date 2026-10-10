"""
Real F5-TTS Inference Engine using official f5-tts library.
Supports in-context voice cloning + studio DSP mastering.
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from src.audio.mastering import apply_studio_mastering
from src.utils.logger import setup_logger

logger = setup_logger("f5_infer")

DEFAULT_WAVS_BASE = "/content/drive/MyDrive/tts-project/02_processed/wavs"
DEFAULT_META_PATH = "/content/drive/MyDrive/tts-project/02_processed/metadata.jsonl"


def _pick_best_ref_chunk(
    speaker_id: str,
    wavs_base: str = DEFAULT_WAVS_BASE,
    meta_path: str = DEFAULT_META_PATH,
    min_dur: float = 5.0,
    max_dur: float = 9.0,
) -> tuple[str, str]:
    """Picks the best reference audio chunk and its transcript for in-context cloning.
    Prefers chunks 5–9s with highest ASR confidence (avg_logprob closest to 0).
    """
    import json

    candidates = []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("speaker", "default") != speaker_id:
                        continue
                    dur = float(rec.get("duration", 0.0))
                    logprob = float(rec.get("avg_logprob", -1.0))
                    wav_p = rec.get("audio_path", "")
                    text = rec.get("normalized_text", rec.get("text", ""))
                    if min_dur <= dur <= max_dur and os.path.exists(wav_p) and text:
                        candidates.append((logprob, dur, wav_p, text))
                except Exception:
                    continue

    if not candidates:
        # Fallback: scan wavs directory for any .wav file
        wavs_dir = Path(wavs_base) / speaker_id
        if wavs_dir.exists():
            wav_files = sorted(list(wavs_dir.glob("*.wav")))
            if wav_files:
                logger.warning(f"No metadata match for '{speaker_id}', using first wav file as reference.")
                return str(wav_files[0]), ""
        raise FileNotFoundError(
            f"No reference audio chunks found for speaker '{speaker_id}'. "
            f"Run Step 3-4 (Enhance + ASR) to prepare training data first."
        )

    # Sort by logprob descending (closest to 0 = most confident)
    candidates.sort(key=lambda x: -x[0])
    best = candidates[0]
    logger.info(
        f"Selected reference chunk for '{speaker_id}': {Path(best[2]).name} "
        f"(dur={best[1]:.1f}s, logprob={best[0]:.3f})"
    )
    return best[2], best[3]


def synthesize_f5(
    text: str,
    speaker_id: str = "test01",
    output_wav_path: str = "output_human.wav",
    ckpt_path: Optional[str] = None,
    wavs_base: str = DEFAULT_WAVS_BASE,
    meta_path: str = DEFAULT_META_PATH,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    speed_factor: float = 1.0,
    pitch_semitones: float = 0.0,
    warmth_drive: float = 1.15,
    room_reverb_wet: float = 0.05,
    deess_gain_db: float = -2.5,
    enable_mastering: bool = True,
    nfe_steps: int = 32,
) -> str:
    """Synthesizes human-like speech using real F5-TTS Flow Matching + Vocos Vocoder.
    In-context voice cloning: automatically picks best reference chunk from training data.
    """
    try:
        from f5_tts.model import CFM, DiT
        from f5_tts.infer.utils_infer import (
            load_model,
            load_vocoder,
            preprocess_ref_audio_text,
            infer_process,
        )
        import torch
    except ImportError:
        raise ImportError(
            "f5-tts and vocos are required. Run: pip install f5-tts vocos"
        )

    t0 = time.perf_counter()
    logger.info(f"F5-TTS synthesizing for speaker '{speaker_id}': '{text[:50]}...'")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Select reference audio & text
    if ref_audio_path is None or ref_text is None:
        ref_audio_path, ref_text = _pick_best_ref_chunk(speaker_id, wavs_base, meta_path)

    logger.info(f"Reference: {Path(ref_audio_path).name} | Ref text: '{ref_text[:40]}...'")

    # 2. Load F5-TTS base model
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
    )
    if ckpt_path is None or not os.path.exists(ckpt_path):
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id="SWivid/F5-TTS",
                filename="F5TTS_Base/model_1200000.safetensors",
                cache_dir="/content/drive/MyDrive/tts-project/00_base_models/f5-tts-hf",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load F5-TTS weights: {e}. "
                "Ensure internet access and HuggingFace Hub is reachable."
            )

    logger.info(f"Loading F5-TTS DiT on {device}...")
    tts_model = load_model(DiT, model_cfg, ckpt_path, device=device)

    # 3. Load Vocos Vocoder
    logger.info("Loading Vocos 24kHz vocoder...")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    # 4. Preprocess reference audio
    audio_in, ref_text_proc = preprocess_ref_audio_text(
        ref_audio_path, ref_text, device=device
    )

    # 5. F5-TTS inference
    raw_wav_path = output_wav_path.replace(".wav", "_raw_f5.wav")
    generated, target_sr, _ = infer_process(
        audio_in,
        ref_text_proc,
        text,
        tts_model,
        vocoder,
        speed=speed_factor,
        nfe_step=nfe_steps,
        show_info=logger.info,
    )

    # Write raw output
    Path(raw_wav_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(raw_wav_path, generated, target_sr, subtype="PCM_16")
    logger.info(f"F5-TTS raw generation complete ({len(generated)/target_sr:.2f}s)")

    # 6. Studio DSP Mastering
    if enable_mastering:
        final_wav = apply_studio_mastering(
            raw_wav_path,
            output_wav_path,
            pitch_semitones=pitch_semitones,
            speed_factor=1.0,  # Speed already applied by F5-TTS natively
            warmth_drive=warmth_drive,
            room_reverb_wet=room_reverb_wet,
            deess_gain_db=deess_gain_db,
        )
    else:
        import shutil
        shutil.copyfile(raw_wav_path, output_wav_path)
        final_wav = output_wav_path

    # Cleanup raw
    if os.path.exists(raw_wav_path) and raw_wav_path != output_wav_path:
        os.remove(raw_wav_path)

    elapsed = time.perf_counter() - t0
    audio_dur = len(generated) / target_sr
    rtf = elapsed / audio_dur if audio_dur > 0 else 0
    logger.info(
        f"✅ F5-TTS synthesis complete in {elapsed:.2f}s | Audio: {audio_dur:.2f}s | RTF: {rtf:.3f} -> {final_wav}"
    )

    return final_wav

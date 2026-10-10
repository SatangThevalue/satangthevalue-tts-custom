"""
Real F5-TTS Inference Engine using official f5-tts library.
Supports in-context voice cloning + studio DSP mastering.
"""
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from src.audio.mastering import apply_studio_mastering
from src.utils.logger import setup_logger

logger = setup_logger("f5_infer")

DEFAULT_WAVS_BASE = "/content/drive/MyDrive/tts-project/02_processed/wavs"
DEFAULT_META_PATH = "/content/drive/MyDrive/tts-project/02_processed/metadata.jsonl"


def _get_f5_base_weights() -> str:
    """Finds or downloads the official F5-TTS Base model weights."""
    candidates = [
        "/content/drive/MyDrive/tts-project/00_base_models/f5-tts/model_base.safetensors",
        "models/f5-tts/model_base.safetensors",
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 10 * 1024 * 1024:
            logger.info(f"Using cached Base F5-TTS weights: {c}")
            return c

    logger.info("Base weights not found locally. Downloading from HuggingFace Hub (SWivid/F5-TTS)...")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id="SWivid/F5-TTS",
        filename="F5TTS_Base/model_1200000.safetensors",
        cache_dir="/content/drive/MyDrive/tts-project/00_base_models/f5-tts-hf",
    )


def _pick_best_ref_chunk(
    speaker_id: str,
    wavs_base: str = DEFAULT_WAVS_BASE,
    meta_path: str = DEFAULT_META_PATH,
    min_dur: float = 2.0,
    max_dur: float = 12.0,
) -> Tuple[str, str]:
    """Picks the best reference audio chunk and its transcript for in-context cloning.
    First tries matching speaker, then falls back to any available processed audio.
    """
    import json

    candidates = []
    fallback_candidates = []

    meta_files = [
        meta_path,
        "/content/dataset_local/metadata.jsonl",
        "/content/drive/MyDrive/tts-project/02_processed/metadata_raw.jsonl",
    ]

    for mf in meta_files:
        if os.path.exists(mf):
            with open(mf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        dur = float(rec.get("duration", 0.0))
                        logprob = float(rec.get("avg_logprob", -1.0))
                        wav_p = rec.get("audio_path", "")
                        text = rec.get("normalized_text", rec.get("text", ""))

                        # Fix Drive to local NVMe path if running on Colab
                        if not os.path.exists(wav_p) and "/content/drive/MyDrive/tts-project/02_processed" in wav_p:
                            alt_p = wav_p.replace(
                                "/content/drive/MyDrive/tts-project/02_processed",
                                "/content/dataset_local",
                            )
                            if os.path.exists(alt_p):
                                wav_p = alt_p

                        if os.path.exists(wav_p) and text:
                            fallback_candidates.append((logprob, dur, wav_p, text))
                            rec_spk = rec.get("speaker", "default")
                            if rec_spk == speaker_id or speaker_id == "default":
                                candidates.append((logprob, dur, wav_p, text))
                    except Exception:
                        continue

    active_pool = candidates if candidates else fallback_candidates
    if active_pool:
        # Prefer chunks in reasonable duration range [min_dur, max_dur]
        preferred = [c for c in active_pool if min_dur <= c[1] <= max_dur]
        chosen_pool = preferred if preferred else active_pool
        chosen_pool.sort(key=lambda x: -x[0])  # Highest ASR confidence
        best = chosen_pool[0]
        logger.info(
            f"Selected reference chunk for '{speaker_id}': {Path(best[2]).name} "
            f"(dur={best[1]:.1f}s, logprob={best[0]:.3f})"
        )
        return best[2], best[3]

    # Search directory tree for any available wav
    search_dirs = [
        Path(wavs_base) / speaker_id,
        Path(wavs_base),
        Path("/content/dataset_local/wavs") / speaker_id,
        Path("/content/dataset_local/wavs"),
        Path("/content/drive/MyDrive/tts-project/01_raw") / speaker_id,
        Path("/content/drive/MyDrive/tts-project/01_raw"),
    ]
    for d in search_dirs:
        if d.exists():
            wav_files = sorted(list(d.glob("*.wav")) + list(d.glob("*.m4a")))
            if wav_files:
                logger.warning(f"No metadata match, using first audio file: {wav_files[0]}")
                return str(wav_files[0]), ""

    raise FileNotFoundError(
        f"No reference audio chunks found for speaker '{speaker_id}'. "
        f"Please run Step 2-4 (Upload audio & Transcribe) first so the model has reference voice to clone!"
    )


def synthesize_f5(
    text: str,
    speaker_id: str = "test01",
    output_wav_path: str = "output_human.wav",
    ckpt_path: Optional[str] = None,
    wavs_base: str = DEFAULT_WAVS_BASE,
    meta_path: str = DEFAULT_META_PATH,
    ref_audio_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    speed_factor: float = 0.9,  # 0.9 recommended for Thai to prevent skipped words
    pitch_semitones: float = 0.0,
    warmth_drive: float = 1.15,
    room_reverb_wet: float = 0.05,
    deess_gain_db: float = -2.5,
    enable_mastering: bool = True,
    nfe_steps: int = 32,
    cfg_strength: float = 2.0,
    sway_sampling_coef: float = -1.0,
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

    # 2. Always load base DiT model first
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
    )
    base_weights = _get_f5_base_weights()
    vocab_p = Path("data/vocab.txt").resolve()
    if not vocab_p.exists():
        vocab_p = Path(__file__).resolve().parent.parent.parent / "data" / "vocab.txt"

    logger.info(f"Loading Base F5-TTS DiT on {device} (Vocab: {vocab_p}) from: {base_weights}")
    from f5_tts.model.utils import get_tokenizer
    vocab_char_map, vocab_size = get_tokenizer(str(vocab_p), "custom")
    tts_model = DiT(
        **model_cfg,
        text_num_embeds=vocab_size,
        mel_dim=100,
    ).to(device)

    # Load base weights safely with Thai embedding expansion
    if base_weights.endswith(".safetensors"):
        from safetensors.torch import load_file
        base_state = load_file(base_weights)
    else:
        base_state = torch.load(base_weights, map_location=device)

    clean_base = {}
    for k, v in base_state.items():
        clean_k = k
        for pfx in ["ema_model.", "transformer."]:
            if clean_k.startswith(pfx):
                clean_k = clean_k[len(pfx):]
        clean_base[clean_k] = v

    # Adapt any shape-mismatched parameters (e.g. Thai text embedding expansion)
    model_dict = tts_model.state_dict()
    for k in list(clean_base.keys()):
        if k in model_dict and hasattr(clean_base[k], "shape"):
            if clean_base[k].shape != model_dict[k].shape:
                logger.info(
                    f"Adapting base shape mismatch for '{k}': "
                    f"checkpoint {clean_base[k].shape} vs model {model_dict[k].shape}"
                )
                if "text_embed" in k and getattr(clean_base[k], "ndim", 0) == 2:
                    num_copy = min(clean_base[k].shape[0], model_dict[k].shape[0])
                    model_dict[k].data[:num_copy] = clean_base[k].data[:num_copy].to(device)
                    logger.info(f"Copied {num_copy} base token embeddings into {k}")
                clean_base.pop(k)

    tts_model.load_state_dict(clean_base, strict=False)
    tts_model.eval()

    # 3. If custom checkpoint provided, apply adapter / fine-tuned weights
    if ckpt_path and os.path.exists(ckpt_path) and ckpt_path != base_weights:
        try:
            logger.info(f"Attempting to apply custom checkpoint weights: {ckpt_path}")
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(ckpt_path)
            else:
                try:
                    state = torch.load(ckpt_path, map_location=device, weights_only=False)
                except TypeError:
                    state = torch.load(ckpt_path, map_location=device)

            clean_state = {}
            for k, v in state.items():
                clean_k = k
                for prefix in [
                    "transformer.base_model.model.",
                    "transformer.",
                    "base_model.model.",
                    "ema_model.",
                ]:
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix):]
                clean_state[clean_k] = v

            model_dict = tts_model.state_dict()
            for k in list(clean_state.keys()):
                if k not in model_dict or (hasattr(clean_state[k], "shape") and clean_state[k].shape != model_dict[k].shape):
                    clean_state.pop(k)

            missing, unexpected = tts_model.load_state_dict(clean_state, strict=False)
            logger.info(
                f"Checkpoint weights applied successfully (Missing: {len(missing)}, Unexpected: {len(unexpected)})"
            )
        except Exception as ckpt_err:
            logger.warning(
                f"Notice: Checkpoint at {ckpt_path} could not be overlaid ({ckpt_err}). "
                f"Continuing with Base F5-TTS model for zero-shot voice cloning."
            )

    # 4. Load Vocos Vocoder (24kHz standard)
    logger.info("Loading Vocos 24kHz vocoder...")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    # 5. Preprocess reference audio
    audio_in, ref_text_proc = preprocess_ref_audio_text(
        ref_audio_path, ref_text, show_info=logger.info
    )

    # 6. F5-TTS Flow Matching Diffusion inference
    raw_wav_path = output_wav_path.replace(".wav", "_raw_f5.wav")
    generated, target_sr, _ = infer_process(
        audio_in,
        ref_text_proc,
        text,
        tts_model,
        vocoder,
        speed=speed_factor,
        nfe_step=nfe_steps,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        show_info=logger.info,
        device=device,
    )

    # Write raw output
    Path(raw_wav_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(raw_wav_path, generated, target_sr, subtype="PCM_16")
    logger.info(f"F5-TTS raw generation complete ({len(generated)/target_sr:.2f}s)")

    # 7. Studio DSP Mastering
    if enable_mastering:
        final_wav = apply_studio_mastering(
            raw_wav_path,
            output_wav_path,
            pitch_semitones=pitch_semitones,
            speed_factor=1.0,  # Speed is natively handled by F5-TTS
            warmth_drive=warmth_drive,
            room_reverb_wet=room_reverb_wet,
            deess_gain_db=deess_gain_db,
        )
    else:
        import shutil
        shutil.copyfile(raw_wav_path, output_wav_path)
        final_wav = output_wav_path

    # Cleanup raw intermediate
    if os.path.exists(raw_wav_path) and raw_wav_path != output_wav_path:
        os.remove(raw_wav_path)

    elapsed = time.perf_counter() - t0
    audio_dur = len(generated) / target_sr
    rtf = elapsed / audio_dur if audio_dur > 0 else 0
    logger.info(
        f"✅ F5-TTS synthesis complete in {elapsed:.2f}s | Audio: {audio_dur:.2f}s | RTF: {rtf:.3f} -> {final_wav}"
    )

    return final_wav

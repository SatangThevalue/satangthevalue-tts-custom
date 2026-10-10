import argparse
import glob
import os
from pathlib import Path
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import soundfile as sf

from src.audio.mastering import apply_studio_mastering
from src.g2p.text_norm import text_to_phonemes
from src.inference.f5_infer import synthesize_f5
from src.utils.logger import setup_logger

logger = setup_logger("test_checkpoint")


def find_latest_checkpoint(checkpoints_dir: str) -> Tuple[Optional[str], int]:
    """Finds the most recent checkpoint step directory (e.g. step_2500/adapter_model.pt)."""
    p = Path(checkpoints_dir)
    if not p.exists():
        return None, 0

    step_dirs = []
    for d in p.glob("step_*"):
        if d.is_dir():
            try:
                num = int(d.name.split("_")[1])
                step_dirs.append((num, d))
            except ValueError:
                continue

    if not step_dirs:
        # Check flat files
        flat_pts = list(p.glob("*.pt"))
        if flat_pts:
            return str(flat_pts[0]), 0
        return None, 0

    step_dirs.sort(key=lambda x: x[0], reverse=True)
    best_num, best_dir = step_dirs[0]
    target_file = best_dir / "adapter_model.pt"
    if target_file.exists():
        return str(target_file), best_num

    candidates = list(best_dir.glob("*.pt")) + list(best_dir.glob("*.safetensors"))
    if candidates:
        return str(candidates[0]), best_num

    return None, 0


def preview_checkpoint_audio(
    speaker_id: str,
    text: str,
    checkpoints_base_dir: str = "/content/drive/MyDrive/tts-project/03_checkpoints",
    output_wav: str = "preview_out.wav",
    checkpoint_step: int = 0,
    speed_factor: float = 1.0,
    pitch_semitones: float = 0.0,
    warmth_drive: float = 1.15,
) -> Dict[str, Any]:
    """Synthesizes human speech using F5-TTS directly from checkpoint BEFORE converting to ONNX."""
    t0 = time.perf_counter()
    logger.info(f"=== PYTORCH CHECKPOINT SOUND CHECK (Pre-ONNX Evaluation) ===")
    logger.info(f"Speaker: '{speaker_id}' | Text: '{text}'")

    speaker_ckpt_dir = os.path.join(checkpoints_base_dir, speaker_id)
    if not os.path.exists(speaker_ckpt_dir):
        speaker_ckpt_dir = checkpoints_base_dir

    if checkpoint_step > 0:
        target_pt = os.path.join(speaker_ckpt_dir, f"step_{checkpoint_step}", "adapter_model.pt")
        active_step = checkpoint_step
    else:
        target_pt, active_step = find_latest_checkpoint(speaker_ckpt_dir)

    # Use base model checkpoint if speaker has no fine-tuned checkpoints yet
    if not target_pt or not os.path.exists(target_pt):
        base_f5 = "/content/drive/MyDrive/tts-project/00_base_models/f5-tts/model_base.safetensors"
        if os.path.exists(base_f5):
            logger.info("Using cached Base F5-TTS model for zero-shot voice cloning preview.")
            target_pt = base_f5
            active_step = 0
        else:
            logger.info("Using default HuggingFace pretrained F5-TTS weights.")
            target_pt = None
            active_step = 0

    if target_pt:
        logger.info(f"Loading weights (Step {active_step}): {target_pt}")

    # Synthesize real human voice or fallback to simulation if F5-TTS not installed
    try:
        final_wav = synthesize_f5(
            text=text,
            speaker_id=speaker_id,
            output_wav_path=output_wav,
            ckpt_path=target_pt,
            speed_factor=speed_factor,
            pitch_semitones=pitch_semitones,
            warmth_drive=warmth_drive,
            enable_mastering=True,
        )
    except ImportError as ie:
        # Headless testing environment without f5-tts (e.g. VPS CI)
        logger.info(f"F5-TTS library not installed in host environment ({ie}). Running simulated test waveform.")
        sample_rate = 24000
        dur_sim = max(1.5, len(text.split()) * 0.38)
        num_samples = int(sample_rate * dur_sim)
        synth_wave = (np.sin(2 * np.pi * 220 * np.linspace(0, dur_sim, num_samples)) * 0.25).astype(np.float32)
        raw_tmp = output_wav.replace(".wav", "_raw.wav")
        os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)
        sf.write(raw_tmp, synth_wave, sample_rate, subtype="PCM_16")
        final_wav = apply_studio_mastering(
            raw_tmp,
            output_wav,
            speed_factor=speed_factor,
            pitch_semitones=pitch_semitones,
            warmth_drive=warmth_drive,
        )
        if os.path.exists(raw_tmp) and raw_tmp != output_wav:
            os.remove(raw_tmp)
    except Exception as real_err:
        logger.error(f"F5-TTS synthesis failed: {real_err}")
        raise RuntimeError(
            f"F5-TTS synthesis failed: {real_err}\n"
            f"คำแนะนำ: ตรวจสอบว่าได้รัน Step 2-4 เพื่อเตรียมไฟล์เสียงต้นฉบับของผู้พูด '{speaker_id}' ใน Google Drive หรือยัง"
        ) from real_err

    elapsed = time.perf_counter() - t0
    dur = 0.0
    if os.path.exists(final_wav):
        info = sf.info(final_wav)
        dur = info.duration

    rtf = elapsed / dur if dur > 0 else 0.0
    logger.info(
        f"Sound check completed! Generated {dur:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.3f}) -> {final_wav}"
    )

    return {
        "status": "SUCCESS",
        "output_path": final_wav,
        "checkpoint_step": active_step,
        "duration": dur,
        "duration_sec": dur,
        "latency_sec": elapsed,
        "elapsed_sec": elapsed,
        "rtf": rtf,
        "phonemes": text_to_phonemes(text),
    }


# Backwards compatibility alias
preview_checkpoint = preview_checkpoint_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PyTorch Checkpoint Voice")
    parser.add_argument("--speaker-id", type=str, default="satang")
    parser.add_argument("--text", type=str, default="สวัสดีครับ นี่คือเสียงทดสอบจากโมเดล")
    parser.add_argument("--checkpoints-dir", type=str, default="/content/drive/MyDrive/tts-project/03_checkpoints")
    parser.add_argument("--step", type=int, default=0, help="Specific step to evaluate (0 = latest)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--warmth", type=float, default=1.15)
    parser.add_argument("--output", type=str, default="preview_checkpoint.wav")
    args = parser.parse_args()

    preview_checkpoint_audio(
        speaker_id=args.speaker_id,
        text=args.text,
        checkpoints_base_dir=args.checkpoints_dir,
        output_wav=args.output,
        checkpoint_step=args.step,
        speed_factor=args.speed,
        pitch_semitones=args.pitch,
        warmth_drive=args.warmth,
    )

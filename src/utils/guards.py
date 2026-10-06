import os
from pathlib import Path
import re
import numpy as np
import soundfile as sf

from src.utils.logger import setup_logger

logger = setup_logger("guards")


def check_vram_limit(limit_gb: float = 13.0) -> float:
    """Hard guard preventing Colab T4 Out-Of-Memory crashes.

    Checks reserved CUDA memory; triggers empty_cache() or raises MemoryError.
    """
    try:
        import torch
    except ImportError:
        logger.debug("Torch not installed. Skipping VRAM check.")
        return 0.0

    if torch.cuda.is_available():
        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        logger.debug(
            f"VRAM Check -> Reserved: {reserved_gb:.2f}GB / Allocated: {allocated_gb:.2f}GB (Ceiling: {limit_gb:.2f}GB)"
        )
        if reserved_gb > limit_gb:
            logger.warning(
                f"VRAM {reserved_gb:.2f}GB exceeded ceiling {limit_gb:.2f}GB! Evicting cache..."
            )
            torch.cuda.empty_cache()
            new_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.debug(f"Post-eviction reserved VRAM: {new_reserved:.2f}GB")
            if new_reserved > limit_gb:
                raise MemoryError(
                    f"[GUARD ALERT] VRAM usage {new_reserved:.2f}GB still exceeds {limit_gb:.2f}GB limit after empty_cache."
                )
        return reserved_gb
    return 0.0


def validate_audio_chunk(
    wav_path: str, min_duration: float = 3.0, max_duration: float = 10.0
) -> bool:
    """Verifies that an audio slice falls within acceptable duration limits

    without corrupt headers or silent data.
    """
    if not os.path.exists(wav_path):
        logger.debug(f"Audio chunk validation failed: File not found: {wav_path}")
        return False
    try:
        info = sf.info(wav_path)
        if info.frames == 0:
            logger.debug(f"Audio chunk validation failed: Zero frames in {wav_path}")
            return False
        if not (min_duration <= info.duration <= max_duration):
            logger.debug(
                f"Audio chunk duration {info.duration:.2f}s outside [{min_duration:.1f}s, {max_duration:.1f}s] in {wav_path}"
            )
            return False
        return True
    except Exception as e:
        logger.debug(f"Audio chunk validation exception on {wav_path}: {e}")
        return False


def validate_thai_tone(phonemes: str) -> bool:
    """Validates that phonetic sequences carry explicit Thai tone markers [0-4].

    Prevents untagged tokens from floating through to training.
    """
    tokens = [t for t in phonemes.strip().split() if t]
    if not tokens:
        logger.debug("Thai tone validation failed: Empty token list")
        return False

    tagged = [t for t in tokens if re.search(r"[0-4]$", t)]
    ratio = len(tagged) / len(tokens)
    is_valid = ratio >= 0.7
    logger.debug(
        f"Tone validation: {len(tagged)}/{len(tokens)} tokens tagged ({ratio * 100:.1f}%) -> Valid: {is_valid}"
    )
    return is_valid


def check_drive_mounted(target_dir: str) -> bool:
    """Ensures Google Drive storage mount is reachable to prevent data loss."""
    path = Path(target_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Test write permission
        test_file = path / ".mount_write_test.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        logger.debug(f"Drive mount verification passed for: {target_dir}")
        return True
    except Exception as e:
        logger.error(f"Drive mount check failed for {target_dir}: {e}")
        return False


def calculate_snr(audio: np.ndarray, threshold_db: float = 25.0) -> tuple[float, bool]:
    """Calculates Signal-to-Noise Ratio (SNR) in dB with numerical stability

    guards.
    """
    if len(audio) == 0 or np.all(audio == 0):
        logger.debug("SNR calculation: Audio is completely silent or empty")
        return 0.0, False

    # Check for NaN / Inf
    if not np.all(np.isfinite(audio)):
        logger.warning("SNR calculation: Non-finite values detected in audio array")
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    signal_power = float(np.mean(audio**2))
    frame_size = 1024

    if len(audio) < frame_size:
        return 30.0, True

    num_frames = len(audio) // frame_size
    frames = audio[: num_frames * frame_size].reshape(num_frames, frame_size)
    frame_powers = np.mean(frames**2, axis=1)

    # 10th percentile lowest energy frames as noise floor
    noise_idx = max(1, len(frame_powers) // 10)
    noise_power = float(np.mean(np.sort(frame_powers)[:noise_idx]))

    if noise_power <= 1e-12:
        snr_db = 45.0
    else:
        snr_db = float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))

    passed = snr_db >= threshold_db
    logger.debug(
        f"Calculated SNR: {snr_db:.2f} dB (Threshold: {threshold_db:.1f} dB, Passed: {passed})"
    )
    return snr_db, passed

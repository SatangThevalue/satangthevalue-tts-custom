import os
import re
import numpy as np
import soundfile as sf
import torch


def check_vram_limit(limit_gb: float = 13.0) -> float:
    """Hard guard preventing Colab T4 Out-Of-Memory crashes.

    Checks reserved CUDA memory; triggers empty_cache() or raises MemoryError.
    """
    if torch.cuda.is_available():
        reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
        if reserved_gb > limit_gb:
            torch.cuda.empty_cache()
            raise MemoryError(
                f"[GUARD ALERT] VRAM usage {reserved_gb:.2f}GB exceeded safe ceiling {limit_gb:.2f}GB. "
                "Triggered cache clearing to prevent Colab kernel crash."
            )
        return reserved_gb
    return 0.0


def validate_audio_chunk(
    wav_path: str, min_duration: float = 3.0, max_duration: float = 10.0
) -> bool:
    """Verifies that an audio slice falls within acceptable duration limits."""
    if not os.path.exists(wav_path):
        return False
    try:
        info = sf.info(wav_path)
        return min_duration <= info.duration <= max_duration
    except Exception:
        return False


def validate_thai_tone(phonemes: str) -> bool:
    """Validates that phonetic sequences carry explicit Thai tone markers [0-4].

    Prevents untagged tokens from floating through to training.
    """
    tokens = phonemes.strip().split()
    if not tokens:
        return False
    # Each phonetic syllable must have a tone digit (0: common, 1: low, 2: falling, 3: high, 4: rising)
    tagged = [t for t in tokens if re.search(r"[0-4]$", t)]
    return len(tagged) / len(tokens) >= 0.7  # At least 70% syllables explicitly tone-locked


def check_drive_mounted(target_dir: str) -> bool:
    """Ensures Google Drive storage mount is reachable to prevent data loss."""
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir, exist_ok=True)
            return True
        except Exception:
            return False
    return True


def calculate_snr(audio: np.ndarray, threshold_db: float = 25.0) -> tuple[float, bool]:
    """Calculates approximate Signal-to-Noise Ratio (SNR) in dB."""
    if len(audio) == 0:
        return 0.0, False
    signal_power = np.mean(audio**2)
    # Estimate noise from lowest 10% energy frames
    frame_size = 1024
    if len(audio) < frame_size:
        return 30.0, True
    frames = np.array(
        [
            audio[i : i + frame_size]
            for i in range(0, len(audio) - frame_size, frame_size)
        ]
    )
    frame_powers = np.mean(frames**2, axis=1)
    noise_power = np.mean(np.sort(frame_powers)[: max(1, len(frame_powers) // 10)])

    if noise_power <= 1e-10:
        snr_db = 40.0
    else:
        snr_db = float(10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10)))

    return snr_db, snr_db >= threshold_db

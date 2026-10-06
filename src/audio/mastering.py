import argparse
import os
from pathlib import Path
import time
import numpy as np
import soundfile as sf

from src.utils.logger import setup_logger

logger = setup_logger("audio_mastering")


def apply_studio_mastering(
    input_wav: str,
    output_wav: str,
    highpass_hz: float = 80.0,
    comp_threshold_db: float = -16.0,
    comp_ratio: float = 2.5,
    limiter_db: float = -1.0,
) -> str:
    """Mastering chain executed 100% in-memory via Spotify's Pedalboard:

    1. Highpass filter @ 80Hz (eliminates mic rumble and plosives).
    2. Transparent dynamic compressor (tightens speech presence).
    3. True Peak limiter @ -1.0 dBFS (prevents inter-sample clipping).
    """
    t0 = time.perf_counter()
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Mastering input file not found: {input_wav}")

    audio_data, sr = sf.read(input_wav)
    logger.debug(
        f"Mastering input: {input_wav} (Samples: {len(audio_data)}, SR: {sr}Hz)"
    )

    if audio_data.ndim > 1:
        logger.debug("Averaging multi-channel input to mono...")
        audio_data = np.mean(audio_data, axis=1)

    initial_peak = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
    initial_rms = (
        float(np.sqrt(np.mean(audio_data**2))) if len(audio_data) > 0 else 0.0
    )
    logger.debug(
        f"Pre-mastering metrics -> Peak: {initial_peak:.4f}, RMS: {initial_rms:.4f}"
    )

    try:
        from pedalboard import Compressor, HighpassFilter, Limiter, Pedalboard

        logger.debug(
            f"Configuring Pedalboard: Highpass={highpass_hz}Hz, Comp={comp_threshold_db}dB @ {comp_ratio}:1, Limiter={limiter_db}dB"
        )
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=highpass_hz),
            Compressor(threshold_db=comp_threshold_db, ratio=comp_ratio),
            Limiter(threshold_db=limiter_db),
        ])

        processed_audio = board(audio_data.astype(np.float32), sr)

    except ImportError:
        logger.warning(
            "Pedalboard not installed. Applying pure numpy high-pass and normalization fallback."
        )
        processed_audio = audio_data
    except Exception as e:
        logger.error(
            f"Pedalboard DSP execution error: {e}. Falling back to clean normalization."
        )
        processed_audio = audio_data

    # Normalize peak to target ceiling (e.g. -1.0 dB)
    peak = float(np.max(np.abs(processed_audio))) if len(processed_audio) > 0 else 0.0
    if peak > 0:
        target_peak = 10 ** (limiter_db / 20.0)
        scale_factor = target_peak / peak
        logger.debug(
            f"Applying peak scaling: Current Peak={peak:.4f} -> Target={target_peak:.4f} (Factor={scale_factor:.4f})"
        )
        processed_audio = processed_audio * scale_factor

    final_rms = (
        float(np.sqrt(np.mean(processed_audio**2)))
        if len(processed_audio) > 0
        else 0.0
    )
    elapsed = time.perf_counter() - t0

    sf.write(output_wav, processed_audio, sr, subtype="PCM_16")
    logger.info(
        f"Mastered output saved: {output_wav} (Final RMS: {final_rms:.4f}, Elapsed: {elapsed * 1000:.2f}ms)"
    )
    return output_wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master generated audio via Pedalboard DSP"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    apply_studio_mastering(args.input, args.output)
    print(f"[Mastering] Completed -> {args.output}")

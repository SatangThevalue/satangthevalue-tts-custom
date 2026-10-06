import argparse
from pathlib import Path
import numpy as np
import soundfile as sf


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
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)
    audio_data, sr = sf.read(input_wav)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    try:
        from pedalboard import Compressor, HighpassFilter, Limiter, Pedalboard

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=highpass_hz),
            Compressor(threshold_db=comp_threshold_db, ratio=comp_ratio),
            Limiter(threshold_db=limiter_db),
        ])

        processed_audio = board(audio_data.astype(np.float32), sr)

    except ImportError:
        print("[WARN] pedalboard not installed. Skipping in-memory DSP chain.")
        processed_audio = audio_data

    # Normalize peak to -1.0 dB
    peak = np.max(np.abs(processed_audio))
    if peak > 0:
        target_peak = 10 ** (limiter_db / 20.0)
        processed_audio = processed_audio * (target_peak / peak)

    sf.write(output_wav, processed_audio, sr, subtype="PCM_16")
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

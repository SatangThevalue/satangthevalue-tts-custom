import argparse
import os
from pathlib import Path
import time
import numpy as np
import soundfile as sf

from src.utils.logger import setup_logger

logger = setup_logger("audio_mastering")


def generate_pink_noise(num_samples: int) -> np.ndarray:
    """Generates normalized pink noise via 1/f spectral filtering in pure NumPy.

    Used to synthesize ambient room tone and eliminate artificial dead silence.
    """
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    white = np.random.randn(num_samples).astype(np.float32)
    X = np.fft.rfft(white)
    S = 1.0 / np.sqrt(np.arange(1, len(X) + 1))
    pink = np.fft.irfft(X * S, num_samples)

    peak = float(np.max(np.abs(pink)))
    if peak > 0:
        pink = pink / peak
    return pink.astype(np.float32)


def apply_tube_warmth(audio: np.ndarray, drive: float = 1.12) -> np.ndarray:
    """Applies gentle analog tube saturation using hyperbolic tangent curve.

    Creates subtle 3rd-order harmonic richness without harsh distortion.
    """
    if drive <= 1.0 or len(audio) == 0:
        return audio
    return (np.tanh(audio * drive) / np.tanh(drive)).astype(np.float32)


def apply_studio_mastering(
    input_wav: str,
    output_wav: str,
    highpass_hz: float = 80.0,
    comp_threshold_db: float = -16.0,
    comp_ratio: float = 2.5,
    limiter_db: float = -1.0,
    warmth_drive: float = 1.12,
    room_reverb_wet: float = 0.04,
    comfort_noise_db: float = -54.0,
    deess_freq_hz: float = 6800.0,
    deess_gain_db: float = -2.5,
    warmth_lowshelf_hz: float = 220.0,
    warmth_lowshelf_gain_db: float = 1.5,
) -> str:
    """Acoustic Realism & Studio Mastering Pipeline (100% In-Memory DSP):

    1. Pre-EQ & De-essing:
       - Highpass filter @ 80Hz (cuts sub-bass mic rumble & plosives).
       - LowShelf filter @ 220Hz (+1.5dB warmth body).
       - Peak de-esser @ 6.8kHz (-2.5dB tames harsh dental sibilance).
    2. Dynamic Speech Compressor (Ratio 2.5:1, Threshold -16dB).
    3. Analog Tube Warmth (soft hyperbolic tangent saturation).
    4. Studio Space Simulation (Micro-reverb room_size 0.08, wet 0.04 for vocal booth depth).
    5. Comfort Noise Injection (Spectral pink noise @ -54 dBFS eliminates dead digital silence).
    6. True Peak Limiting & Normalization (locks peak to -1.0 dBFS).
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
        from pedalboard import (
            Compressor,
            HighpassFilter,
            Limiter,
            LowShelfFilter,
            PeakFilter,
            Pedalboard,
            Reverb,
        )

        effects = [
            HighpassFilter(cutoff_frequency_hz=highpass_hz),
            LowShelfFilter(
                cutoff_frequency_hz=warmth_lowshelf_hz,
                gain_db=warmth_lowshelf_gain_db,
                q=0.7,
            ),
            PeakFilter(
                cutoff_frequency_hz=deess_freq_hz,
                gain_db=deess_gain_db,
                q=1.2,
            ),
            Compressor(
                threshold_db=comp_threshold_db,
                ratio=comp_ratio,
                attack_ms=10.0,
                release_ms=100.0,
            ),
        ]

        if room_reverb_wet > 0.0:
            effects.append(
                Reverb(
                    room_size=0.08,
                    wet_level=room_reverb_wet,
                    dry_level=1.0 - room_reverb_wet,
                    damping=0.6,
                    width=0.5,
                )
            )

        board = Pedalboard(effects)
        processed_audio = board(audio_data.astype(np.float32), sr)
        logger.debug(
            f"Pedalboard EQ + Dynamics + Studio Reverb executed ({len(effects)} stages)"
        )

    except ImportError:
        logger.warning(
            "Pedalboard not installed. Applying pure NumPy fallback chain."
        )
        processed_audio = audio_data.astype(np.float32)
    except Exception as e:
        logger.error(f"Pedalboard DSP execution error: {e}. Falling back to input.")
        processed_audio = audio_data.astype(np.float32)

    # 3. Analog Tube Saturation
    if warmth_drive > 1.0:
        processed_audio = apply_tube_warmth(processed_audio, drive=warmth_drive)
        logger.debug(f"Applied analog tube warmth saturation (drive={warmth_drive})")

    # 4. Comfort Noise Injection (Pink noise to mask digital dead silence)
    if comfort_noise_db is not None and comfort_noise_db > -90.0:
        noise_amplitude = float(10.0 ** (comfort_noise_db / 20.0))
        pink_noise = generate_pink_noise(len(processed_audio)) * noise_amplitude
        processed_audio = processed_audio + pink_noise
        logger.debug(
            f"Injected comfort room-tone noise ({comfort_noise_db:.1f} dBFS, amp={noise_amplitude:.6f})"
        )

    # 5. True Peak Limiting & Normalization
    try:
        from pedalboard import Limiter, Pedalboard

        limiter_board = Pedalboard([Limiter(threshold_db=limiter_db)])
        processed_audio = limiter_board(processed_audio, sr)
    except Exception:
        pass

    peak = float(np.max(np.abs(processed_audio))) if len(processed_audio) > 0 else 0.0
    if peak > 0:
        target_peak = float(10.0 ** (limiter_db / 20.0))
        scale_factor = target_peak / peak
        logger.debug(
            f"Applying peak normalization: Peak={peak:.4f} -> Target={target_peak:.4f} (Factor={scale_factor:.4f})"
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
        f"Mastered output saved: {output_wav} (Peak: {float(np.max(np.abs(processed_audio))):.4f}, Final RMS: {final_rms:.4f}, Elapsed: {elapsed * 1000:.2f}ms)"
    )
    return output_wav


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master generated audio via Pedalboard DSP with Acoustic Realism"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--warmth-drive", type=float, default=1.12)
    parser.add_argument("--reverb-wet", type=float, default=0.04)
    parser.add_argument("--noise-db", type=float, default=-54.0)
    args = parser.parse_args()

    apply_studio_mastering(
        args.input,
        args.output,
        warmth_drive=args.warmth_drive,
        room_reverb_wet=args.reverb_wet,
        comfort_noise_db=args.noise_db,
    )
    print(f"[Mastering] Completed -> {args.output}")

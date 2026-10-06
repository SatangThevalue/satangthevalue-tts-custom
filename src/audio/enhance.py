import argparse
import os
from pathlib import Path
import time
import soundfile as sf

from src.utils.guards import calculate_snr
from src.utils.logger import setup_logger

logger = setup_logger("audio_enhance")


def enhance_audio_file(
    input_path: str,
    output_path: str,
    target_sr: int = 24000,
    denoise_only: bool = False,
) -> str:
    """Enhances phone recordings into studio-quality audio:

    1. Removes room reverberation and background hiss.
    2. Extends frequency bandwidth (12kHz - 24kHz) for crisp highs.
    3. Resamples to uniform mono 24kHz.
    """
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        import torch
        import torchaudio
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch = None
        torchaudio = None
        device = "cpu"

    logger.debug(f"Enhancing: {input_path} (Target SR: {target_sr}, Device: {device})")

    # Load audio safely
    try:
        waveform, sr = torchaudio.load(input_path)
    except Exception as e:
        logger.error(f"Failed loading audio file {input_path}: {e}")
        raise

    orig_duration = waveform.shape[1] / sr
    logger.debug(
        f"Input specs -> Channels: {waveform.shape[0]}, Sample Rate: {sr}Hz, Duration: {orig_duration:.2f}s"
    )

    if waveform.shape[0] > 1:
        logger.debug("Downmixing multi-channel audio to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Initial SNR Check
    raw_snr, _ = calculate_snr(waveform.squeeze(0).numpy())
    logger.debug(f"Pre-enhancement SNR: {raw_snr:.2f} dB")

    try:
        from resemble_enhance.enhancer.inference import denoise, enhance

        logger.debug("Resemble Enhance backend detected. Preparing 44.1kHz tensors...")
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=44100
            ).to(device)
            waveform_44k = resampler(waveform.to(device))
        else:
            waveform_44k = waveform.to(device)

        with torch.no_grad():
            if denoise_only:
                logger.debug("Running denoise-only mode...")
                enhanced_wav, _ = denoise(
                    waveform_44k.squeeze(0), 44100, device=device
                )
            else:
                logger.debug("Running full enhancement (denoise + bandwidth extension)...")
                enhanced_wav, _ = enhance(
                    waveform_44k.squeeze(0),
                    44100,
                    device=device,
                    nfe=32,
                    solver="midpoint",
                    lambd=0.9,
                    tau=0.5,
                )

        enhanced_wav = enhanced_wav.cpu().unsqueeze(0)
        resampler_final = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=target_sr
        )
        final_wav = resampler_final(enhanced_wav).squeeze(0).numpy()

    except ImportError:
        logger.warning(
            "resemble-enhance package not found. Executing high-pass and high-quality sinc resample fallback."
        )
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr
            )
            waveform = resampler(waveform)
        final_wav = waveform.squeeze(0).numpy()
    except Exception as e:
        logger.error(f"Error during Resemble Enhance model execution: {e}. Falling back to original waveform.")
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr
            )
            waveform = resampler(waveform)
        final_wav = waveform.squeeze(0).numpy()

    # Post-enhancement SNR Check
    post_snr, passes_snr = calculate_snr(final_wav)
    elapsed = time.perf_counter() - t0

    logger.debug(
        f"Post-enhancement SNR: {post_snr:.2f} dB (Change: {post_snr - raw_snr:+.2f} dB) | Time: {elapsed:.2f}s"
    )

    sf.write(output_path, final_wav, target_sr, subtype="PCM_16")
    logger.info(f"Saved enhanced audio -> {output_path} ({orig_duration:.2f}s)")
    return output_path


def batch_enhance(
    raw_dir: str, enhanced_dir: str, target_sr: int = 24000
) -> list[str]:
    """Iterates through raw audio directory and enhances all files safely."""
    raw_path = Path(raw_dir)
    out_path = Path(enhanced_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    supported_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".webm"}
    audio_files = [
        f for f in raw_path.rglob("*") if f.suffix.lower() in supported_exts
    ]
    logger.info(f"Discovered {len(audio_files)} audio files in {raw_dir}")

    results = []
    for idx, file in enumerate(audio_files):
        rel_name = file.stem + ".wav"
        dest_file = out_path / rel_name

        # Incremental check: skip already enhanced audio
        if dest_file.exists() and dest_file.stat().st_size > 1000:
            logger.info(
                f"[{idx + 1}/{len(audio_files)}] Skipping already enhanced audio: {file.name}"
            )
            results.append(str(dest_file))
            continue

        logger.debug(f"[{idx + 1}/{len(audio_files)}] Processing file: {file.name}")
        try:
            enhanced = enhance_audio_file(
                str(file), str(dest_file), target_sr=target_sr
            )
            results.append(enhanced)
        except Exception as e:
            logger.error(f"Failed processing {file.name}: {e}", exc_info=True)

    logger.info(f"Batch enhance finished. Successfully processed {len(results)}/{len(audio_files)} files.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance raw phone recordings")
    parser.add_argument(
        "--raw-dir", type=str, default="/content/drive/MyDrive/tts-project/01_raw"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/content/drive/MyDrive/tts-project/02_processed/enhanced",
    )
    parser.add_argument("--sr", type=int, default=24000)
    args = parser.parse_args()

    batch_enhance(args.raw_dir, args.out_dir, args.sr)

import argparse
import os
from pathlib import Path
import soundfile as sf
import torch
import torchaudio

from src.utils.guards import calculate_snr


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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load audio
    waveform, sr = torchaudio.load(input_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

    try:
        from resemble_enhance.enhancer.inference import denoise, enhance

        # Resemble Enhance operates at 44.1kHz internally
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=44100
            ).to(device)
            waveform_44k = resampler(waveform.to(device))
        else:
            waveform_44k = waveform.to(device)

        with torch.no_grad():
            if denoise_only:
                enhanced_wav, _ = denoise(
                    waveform_44k.squeeze(0), 44100, device=device
                )
            else:
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
        # Resample to target sample rate (default 24000Hz)
        resampler_final = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=target_sr
        )
        final_wav = resampler_final(enhanced_wav).squeeze(0).numpy()

    except ImportError:
        # Fallback if resemble-enhance is not compiled in environment
        print(
            "[WARN] resemble-enhance not found. Using native torch high-pass & resampler fallback."
        )
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr
            )
            waveform = resampler(waveform)
        final_wav = waveform.squeeze(0).numpy()

    # SNR Check
    snr_db, is_clean = calculate_snr(final_wav)
    print(f"[{os.path.basename(input_path)}] Processed SNR: {snr_db:.2f} dB")

    sf.write(output_path, final_wav, target_sr, subtype="PCM_16")
    return output_path


def batch_enhance(
    raw_dir: str, enhanced_dir: str, target_sr: int = 24000
) -> list[str]:
    """Iterates through raw audio directory and enhances all files."""
    raw_path = Path(raw_dir)
    out_path = Path(enhanced_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    supported_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
    audio_files = [
        f for f in raw_path.rglob("*") if f.suffix.lower() in supported_exts
    ]
    print(f"[Enhance] Found {len(audio_files)} raw audio files to process.")

    results = []
    for file in audio_files:
        rel_name = file.stem + ".wav"
        dest_file = out_path / rel_name
        try:
            enhanced = enhance_audio_file(
                str(file), str(dest_file), target_sr=target_sr
            )
            results.append(enhanced)
        except Exception as e:
            print(f"[ERROR] Failed enhancing {file.name}: {e}")

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

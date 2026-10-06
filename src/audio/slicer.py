import argparse
from pathlib import Path
import soundfile as sf
import torch
import torchaudio

from src.utils.guards import validate_audio_chunk


def slice_audio_with_vad(
    input_wav: str,
    output_dir: str,
    min_duration: float = 3.0,
    max_duration: float = 10.0,
    pad_start_ms: int = 150,
    pad_end_ms: int = 200,
) -> list[str]:
    """Slices long recordings into optimal training chunks (3-10s) using Silero

    VAD. Preserves natural inhalation/exhalation via customizable start/end
    padding.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    waveform, sr = torchaudio.load(input_wav)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Load Silero VAD model
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=True,
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = (
        utils
    )

    # Silero requires 16000Hz for timestamp analysis
    resampler_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform_16k = resampler_16k(waveform).squeeze(0)

    speech_timestamps = get_speech_timestamps(
        waveform_16k,
        model,
        sampling_rate=16000,
        min_speech_duration_ms=int(min_duration * 1000),
        max_speech_duration_s=max_duration,
        min_silence_duration_ms=400,
        speech_pad_ms=max(pad_start_ms, pad_end_ms),
    )

    pad_start_samples = int(sr * (pad_start_ms / 1000.0))
    pad_end_samples = int(sr * (pad_end_ms / 1000.0))
    total_samples = waveform.shape[1]

    created_chunks = []
    base_name = Path(input_wav).stem

    for idx, ts in enumerate(speech_timestamps):
        # Convert 16k timestamps back to native sample rate
        start_samp = max(0, int(ts["start"] * (sr / 16000)) - pad_start_samples)
        end_samp = min(
            total_samples, int(ts["end"] * (sr / 16000)) + pad_end_samples
        )

        chunk_wave = waveform[:, start_samp:end_samp]
        duration = chunk_wave.shape[1] / sr

        if min_duration <= duration <= max_duration:
            chunk_filename = out_dir / f"{base_name}_seg_{idx:04d}.wav"
            sf.write(
                str(chunk_filename),
                chunk_wave.squeeze(0).numpy(),
                sr,
                subtype="PCM_16",
            )

            if validate_audio_chunk(
                str(chunk_filename), min_duration, max_duration
            ):
                created_chunks.append(str(chunk_filename))
            else:
                chunk_filename.unlink(missing_ok=True)

    print(
        f"[{base_name}] Successfully sliced into {len(created_chunks)} valid chunks (3-10s with breath padding)."
    )
    return created_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slice audio using Silero VAD with breath preservation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to enhanced audio file or directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/content/drive/MyDrive/tts-project/02_processed/wavs",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.is_file():
        slice_audio_with_vad(str(in_path), args.out_dir)
    elif in_path.is_dir():
        for audio_file in in_path.glob("*.wav"):
            slice_audio_with_vad(str(audio_file), args.out_dir)

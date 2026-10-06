import argparse
import os
from pathlib import Path
import soundfile as sf

from src.utils.guards import validate_audio_chunk
from src.utils.logger import setup_logger

logger = setup_logger("audio_slicer")


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
    base_name = Path(input_wav).stem

    # Incremental check: skip if file was already sliced
    existing_chunks = sorted(list(out_dir.glob(f"{base_name}_seg_*.wav")))
    if existing_chunks:
        logger.info(
            f"Audio file '{base_name}' already sliced into {len(existing_chunks)} chunks. Skipping slicing."
        )
        return [str(c) for c in existing_chunks]

    logger.debug(f"Slicing input: {input_wav} -> output_dir: {output_dir}")

    try:
        import torch
        import torchaudio
    except ImportError:
        logger.error("torch/torchaudio must be installed to run slice_audio_with_vad.")
        return []

    try:
        waveform, sr = torchaudio.load(input_wav)
    except Exception as e:
        logger.error(f"Cannot read audio file {input_wav}: {e}")
        return []

    if waveform.shape[0] > 1:
        logger.debug("Converting multi-channel waveform to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    total_samples = waveform.shape[1]
    total_duration_sec = total_samples / sr
    logger.debug(
        f"Input duration: {total_duration_sec:.2f}s ({total_samples} samples @ {sr}Hz)"
    )

    if total_duration_sec < min_duration:
        logger.warning(
            f"Input file {input_wav} is shorter than min_duration ({total_duration_sec:.2f}s < {min_duration}s). Skipping."
        )
        return []

    # Load Silero VAD model with fallback
    try:
        logger.debug("Loading Silero VAD model via torch.hub...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=True,
        )
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    except Exception as e:
        logger.error(f"Failed to load Silero VAD from torch.hub: {e}. Attempting fallback...")
        return _fallback_energy_slicer(waveform, sr, input_wav, str(out_dir), min_duration, max_duration)

    # Resample to 16kHz for Silero VAD detection
    resampler_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform_16k = resampler_16k(waveform).squeeze(0)

    try:
        speech_timestamps = get_speech_timestamps(
            waveform_16k,
            model,
            sampling_rate=16000,
            min_speech_duration_ms=int(min_duration * 1000),
            max_speech_duration_s=max_duration,
            min_silence_duration_ms=300,
            speech_pad_ms=max(pad_start_ms, pad_end_ms),
        )
    except Exception as e:
        logger.error(f"Silero VAD execution error: {e}")
        return []

    logger.debug(f"Silero VAD identified {len(speech_timestamps)} speech segments.")

    if not speech_timestamps:
        logger.warning(f"No voice activity detected by Silero in {input_wav}. Attempting energy fallback.")
        return _fallback_energy_slicer(waveform, sr, input_wav, str(out_dir), min_duration, max_duration)

    pad_start_samples = int(sr * (pad_start_ms / 1000.0))
    pad_end_samples = int(sr * (pad_end_ms / 1000.0))

    created_chunks = []
    base_name = Path(input_wav).stem

    for idx, ts in enumerate(speech_timestamps):
        start_samp = max(0, int(ts["start"] * (sr / 16000)) - pad_start_samples)
        end_samp = min(total_samples, int(ts["end"] * (sr / 16000)) + pad_end_samples)

        chunk_wave = waveform[:, start_samp:end_samp]
        duration = chunk_wave.shape[1] / sr

        logger.debug(
            f"Segment {idx}: start={start_samp}, end={end_samp}, duration={duration:.2f}s"
        )

        if min_duration <= duration <= max_duration:
            chunk_filename = out_dir / f"{base_name}_seg_{idx:04d}.wav"
            sf.write(
                str(chunk_filename),
                chunk_wave.squeeze(0).numpy(),
                sr,
                subtype="PCM_16",
            )

            if validate_audio_chunk(str(chunk_filename), min_duration, max_duration):
                created_chunks.append(str(chunk_filename))
                logger.debug(f"Accepted chunk: {chunk_filename.name} ({duration:.2f}s)")
            else:
                logger.debug(f"Rejected chunk: {chunk_filename.name} by duration/audio guard.")
                chunk_filename.unlink(missing_ok=True)
        else:
            logger.debug(
                f"Skipped segment {idx}: duration {duration:.2f}s outside [{min_duration:.1f}s, {max_duration:.1f}s]"
            )

    logger.info(
        f"Sliced {input_wav}: Created {len(created_chunks)} valid chunks (with {pad_start_ms}ms/{pad_end_ms}ms breath pads)."
    )
    return created_chunks


def _fallback_energy_slicer(
    waveform,
    sr: int,
    input_wav: str,
    output_dir: str,
    min_duration: float,
    max_duration: float,
) -> list[str]:
    """Fallback fixed-window slicer if Silero VAD is unavailable or yields no

    speech.
    """
    logger.info(f"Running fallback window slicer for {input_wav}...")
    out_dir = Path(output_dir)
    base_name = Path(input_wav).stem
    chunk_samples = int(sr * 6.0)  # Default 6-second window
    total_samples = waveform.shape[1]

    created = []
    idx = 0
    for start in range(0, total_samples, chunk_samples):
        end = min(total_samples, start + chunk_samples)
        chunk = waveform[:, start:end]
        dur = chunk.shape[1] / sr
        if dur >= min_duration:
            out_file = out_dir / f"{base_name}_fallback_{idx:04d}.wav"
            sf.write(str(out_file), chunk.squeeze(0).numpy(), sr, subtype="PCM_16")
            created.append(str(out_file))
            idx += 1
    logger.info(f"Fallback slicer produced {len(created)} chunks.")
    return created


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

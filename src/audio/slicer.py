import argparse
import os
from pathlib import Path
import shutil
import time
import soundfile as sf

from src.utils.guards import validate_audio_chunk
from src.utils.logger import setup_logger

logger = setup_logger("audio_slicer")


def get_clean_stem_prefix(filename: str) -> str:
    """Extracts compact reference ID or clean prefix from input filename."""
    stem = Path(filename).stem
    if "_" in stem:
        possible_id = stem.split("_")[-1].strip()
        if len(possible_id) >= 6 and all(c.isalnum() or c in "-_" for c in possible_id):
            return possible_id
    # Sanitize and truncate
    clean = "".join(c for c in stem if c.isalnum() or c in "-_")
    return clean[:20] if clean else "audio_seg"


def slice_audio_with_vad(
    input_wav: str,
    output_dir: str,
    min_duration: float = 3.0,
    max_duration: float = 10.0,
    pad_start_ms: int = 150,
    pad_end_ms: int = 200,
) -> list[str]:
    """Slices long recordings into optimal training chunks (3-10s) using Silero VAD
    with high-speed local NVMe staging and compact filenames.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = get_clean_stem_prefix(input_wav)

    # Incremental check: skip if file was already sliced
    existing_chunks = sorted(list(out_dir.glob(f"{prefix}_seg_*.wav")))
    if not existing_chunks:
        # Check original stem in case of prior run
        orig_stem = Path(input_wav).stem
        existing_chunks = sorted(list(out_dir.glob(f"{orig_stem}_seg_*.wav")))

    if existing_chunks:
        logger.info(
            f"Audio file '{prefix}' already sliced into {len(existing_chunks)} chunks. Skipping slicing."
        )
        return [str(c) for c in existing_chunks]

    t0 = time.perf_counter()
    logger.info(f"Slicing input: {Path(input_wav).name} -> output_dir: {output_dir}")

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
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    total_samples = waveform.shape[1]
    total_duration_sec = total_samples / sr
    logger.info(
        f"Input duration: {total_duration_sec:.2f}s ({total_samples} samples @ {sr}Hz)"
    )

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

    logger.info(f"Silero VAD identified {len(speech_timestamps)} raw speech segments.")

    if not speech_timestamps:
        logger.warning(f"No voice activity detected by Silero in {input_wav}. Attempting energy fallback.")
        return _fallback_energy_slicer(waveform, sr, input_wav, str(out_dir), min_duration, max_duration)

    pad_start_samples = int(sr * (pad_start_ms / 1000.0))
    pad_end_samples = int(sr * (pad_end_ms / 1000.0))

    # Fast local staging to avoid slow Google Drive FUSE random write latency
    staging_slice_dir = Path("/tmp") / f"slice_staging_{prefix}_{int(time.time())}"
    staging_slice_dir.mkdir(parents=True, exist_ok=True)

    created_chunks = []
    chunk_numpy_cache = []

    try:
        for idx, ts in enumerate(speech_timestamps):
            start_samp = max(0, int(ts["start"] * (sr / 16000)) - pad_start_samples)
            end_samp = min(total_samples, int(ts["end"] * (sr / 16000)) + pad_end_samples)

            chunk_wave = waveform[:, start_samp:end_samp]
            duration = chunk_wave.shape[1] / sr

            if min_duration <= duration <= max_duration:
                chunk_name = f"{prefix}_seg_{idx:04d}.wav"
                staging_path = staging_slice_dir / chunk_name
                sf.write(
                    str(staging_path),
                    chunk_wave.squeeze(0).numpy(),
                    sr,
                    subtype="PCM_16",
                )
                chunk_numpy_cache.append((staging_path, chunk_name))

            if (idx + 1) % 100 == 0 or (idx + 1) == len(speech_timestamps):
                logger.info(
                    f"VAD Slice Progress: [{idx + 1}/{len(speech_timestamps)}] segments evaluated ({len(chunk_numpy_cache)} accepted)."
                )

        # Batch copy from NVMe staging to final Google Drive directory
        logger.info(f"Syncing {len(chunk_numpy_cache)} sliced chunks to Google Drive...")
        for staging_p, chunk_name in chunk_numpy_cache:
            dest_file = out_dir / chunk_name
            shutil.copyfile(str(staging_p), str(dest_file))
            created_chunks.append(str(dest_file))

    finally:
        shutil.rmtree(staging_slice_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Sliced {input_wav}: Created {len(created_chunks)} valid chunks in {elapsed:.1f}s."
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
    """Fallback fixed-window slicer if Silero VAD is unavailable or yields no speech."""
    logger.info(f"Running fallback window slicer for {input_wav}...")
    out_dir = Path(output_dir)
    prefix = get_clean_stem_prefix(input_wav)
    chunk_samples = int(sr * 6.0)
    total_samples = waveform.shape[1]

    created = []
    idx = 0
    for start in range(0, total_samples, chunk_samples):
        end = min(total_samples, start + chunk_samples)
        chunk = waveform[:, start:end]
        dur = chunk.shape[1] / sr
        if dur >= min_duration:
            out_file = out_dir / f"{prefix}_fallback_{idx:04d}.wav"
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
        for audio_file in sorted(in_path.rglob("*.wav")):
            try:
                rel_parent = audio_file.parent.relative_to(in_path)
                target_out = Path(args.out_dir) / rel_parent
            except ValueError:
                target_out = Path(args.out_dir)
            target_out.mkdir(parents=True, exist_ok=True)
            slice_audio_with_vad(str(audio_file), str(target_out))

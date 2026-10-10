import argparse
import json
import os
from pathlib import Path
import time
from typing import Optional
import numpy as np
try:
    import torch
except ImportError:
    torch = None  # type: ignore

from src.utils.logger import setup_logger

logger = setup_logger("asr_transcribe")

# Compatibility guard for PyAV in older Colab environments
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        try:
            return _orig_av_open(*args, **kwargs)
        except TypeError as te:
            if "metadata_errors" in str(te):
                kwargs.pop("metadata_errors", None)
                return _orig_av_open(*args, **kwargs)
            raise

    av.open = _safe_av_open
except Exception:
    pass


def load_audio_for_whisper(audio_path: str) -> np.ndarray:
    """Loads WAV audio via soundfile directly into float32 mono 16kHz array.
    Completely bypasses PyAV container bugs and metadata_errors keyword issues.
    """
    import soundfile as sf
    from scipy.signal import resample

    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != 16000:
        target_len = int(len(data) * 16000 / sr)
        data = resample(data, target_len)

    return data.astype(np.float32)


def transcribe_chunk(
    model, audio_path: str, min_logprob: float = -0.5
) -> dict | None:
    """Transcribes a single audio chunk using Faster-Whisper.
    Filters out hallucinations if average log probability is lower than threshold.
    """
    logger.debug(f"Transcribing chunk: {os.path.basename(audio_path)}")
    t0 = time.perf_counter()

    try:
        # Pass decoded 16kHz float32 numpy array directly
        try:
            audio_input = load_audio_for_whisper(audio_path)
            duration_sec = len(audio_input) / 16000.0
        except Exception as load_e:
            logger.debug(f"Soundfile fast-load fallback ({load_e}). Passing path directly.")
            audio_input = audio_path
            duration_sec = 0.0

        segments, info = model.transcribe(
            audio_input,
            language="th",
            task="transcribe",
            beam_size=5,
            word_timestamps=False,
            vad_filter=False,  # Already sliced with Silero VAD
        )

        full_text = []
        logprobs = []

        for seg in segments:
            clean_seg_text = seg.text.strip()
            if clean_seg_text:
                full_text.append(clean_seg_text)
                logprobs.append(seg.avg_logprob)

        if not full_text:
            logger.debug(f"Empty transcription returned for {os.path.basename(audio_path)}")
            return None

        combined_text = " ".join(full_text).strip()
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -1.0
        elapsed = time.perf_counter() - t0
        calc_duration = info.duration if hasattr(info, "duration") and info.duration > 0 else duration_sec

        logger.debug(
            f"Transcript: '{combined_text}' | Duration: {calc_duration:.2f}s | avg_logprob: {avg_logprob:.3f} | Elapsed: {elapsed:.2f}s"
        )

        if avg_logprob < min_logprob:
            logger.warning(
                f"Filtered out {os.path.basename(audio_path)} due to low confidence: {avg_logprob:.3f} < threshold {min_logprob:.3f}"
            )
            return None

        return {
            "audio_path": os.path.abspath(audio_path),
            "text": combined_text,
            "avg_logprob": avg_logprob,
            "duration": calc_duration,
        }

    except Exception as e:
        logger.error(f"Whisper inference failed on {audio_path}: {e}")
        return None


def transcribe_dataset(
    wavs_dir: str,
    output_jsonl: str,
    model_size: str = "large-v3",
    min_logprob: float = -0.5,
    download_root: str = "/content/drive/MyDrive/tts-project/00_base_models/whisper",
) -> int:
    """Iterates through sliced audio chunks and produces metadata.jsonl with
    permanent Google Drive model caching and resume support.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("❌ faster-whisper is not installed. Run: pip install faster-whisper")
        raise ImportError("faster-whisper is required for ASR transcription. Run: pip install faster-whisper")

    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    compute_type = "float16" if (torch is not None and torch.cuda.is_available()) else "int8"

    os.makedirs(download_root, exist_ok=True)
    logger.info(
        f"Loading Faster-Whisper ({model_size}) on device={device} (compute_type={compute_type}, cache={download_root})..."
    )
    try:
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
    except Exception as e:
        logger.warning(
            f"Failed loading Whisper on {device} ({e}). Falling back to CPU int8..."
        )
        try:
            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root=download_root,
            )
        except Exception as cpu_e:
            logger.error(f"❌ Failed to load Faster-Whisper model on both GPU and CPU: {cpu_e}")
            raise

    wav_files = sorted(list(Path(wavs_dir).rglob("*.wav")))
    if not wav_files:
        logger.warning(f"No WAV files found in directory: {wavs_dir}")
        return 0

    logger.info(f"Discovered {len(wav_files)} chunks in {wavs_dir}")

    # Resume support: Read existing processed files
    existing_paths = set()
    output_path = Path(output_jsonl)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing_paths.add(record.get("audio_path"))
                    except Exception:
                        pass
        logger.info(
            f"Resuming ASR: Found {len(existing_paths)} already transcribed entries."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_count = len(existing_paths)

    with open(output_path, "a", encoding="utf-8") as f_out:
        for idx, wav in enumerate(wav_files):
            abs_wav = str(wav.resolve())
            if abs_wav in existing_paths:
                logger.debug(f"Skipping already transcribed: {wav.name}")
                continue

            logger.debug(f"[{idx + 1}/{len(wav_files)}] Transcribing {wav.name}...")
            result = transcribe_chunk(model, abs_wav, min_logprob=min_logprob)

            if result:
                # Extract speaker from directory hierarchy
                speaker_name = "default"
                try:
                    rel_p = wav.parent.relative_to(Path(wavs_dir))
                    if str(rel_p) != ".":
                        speaker_name = str(rel_p).split(os.sep)[0]
                except ValueError:
                    pass
                result["speaker"] = speaker_name

                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()  # Atomic flush to disk
                valid_count += 1
                existing_paths.add(abs_wav)

            if (idx + 1) % 25 == 0:
                logger.info(
                    f"ASR Progress: {idx + 1}/{len(wav_files)} files evaluated ({valid_count} accepted)."
                )

    logger.info(
        f"ASR Transcription complete! Total valid entries: {valid_count}/{len(wav_files)} saved to {output_jsonl}"
    )
    return valid_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio slices to JSONL"
    )
    parser.add_argument(
        "--wavs-dir",
        type=str,
        default="/content/drive/MyDrive/tts-project/02_processed/wavs",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="/content/drive/MyDrive/tts-project/02_processed/metadata_raw.jsonl",
    )
    parser.add_argument("--model-size", type=str, default="large-v3")
    parser.add_argument("--min-logprob", type=float, default=-0.5)
    parser.add_argument(
        "--download-root",
        type=str,
        default="/content/drive/MyDrive/tts-project/00_base_models/whisper",
        help="Google Drive path to cache Whisper model weights",
    )
    args = parser.parse_args()

    transcribe_dataset(
        args.wavs_dir,
        args.output_jsonl,
        model_size=args.model_size,
        min_logprob=args.min_logprob,
        download_root=args.download_root,
    )

import argparse
import json
import os
from pathlib import Path
import time
import torch

from src.utils.logger import setup_logger

logger = setup_logger("asr_transcribe")


def transcribe_chunk(
    model, audio_path: str, min_logprob: float = -0.5
) -> dict | None:
    """Transcribes a single audio chunk using Faster-Whisper.

    Filters out hallucinations if average log probability is lower than
    threshold.
    """
    logger.debug(f"Transcribing chunk: {os.path.basename(audio_path)}")
    t0 = time.perf_counter()

    try:
        segments, info = model.transcribe(
            audio_path,
            language="th",
            task="transcribe",
            beam_size=5,
            word_timestamps=False,
            vad_filter=False,  # Already sliced with Silero VAD
        )
    except Exception as e:
        logger.error(f"Whisper inference failed on {audio_path}: {e}")
        return None

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

    logger.debug(
        f"Transcript: '{combined_text}' | Duration: {info.duration:.2f}s | avg_logprob: {avg_logprob:.3f} | Elapsed: {elapsed:.2f}s"
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
        "duration": info.duration,
    }


def transcribe_dataset(
    wavs_dir: str,
    output_jsonl: str,
    model_size: str = "large-v3",
    min_logprob: float = -0.5,
) -> int:
    """Iterates through sliced audio chunks and produces metadata.jsonl with

    resume support.
    """
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    logger.info(
        f"Loading Faster-Whisper ({model_size}) on device={device} (compute_type={compute_type})..."
    )
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        logger.warning(
            f"Failed loading Whisper on {device} ({e}). Falling back to CPU int8..."
        )
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    wav_files = sorted(list(Path(wavs_dir).glob("*.wav")))
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
    args = parser.parse_args()

    transcribe_dataset(
        args.wavs_dir,
        args.output_jsonl,
        model_size=args.model_size,
        min_logprob=args.min_logprob,
    )

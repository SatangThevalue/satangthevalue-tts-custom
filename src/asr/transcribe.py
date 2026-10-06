import argparse
import json
import os
from pathlib import Path
import torch


def transcribe_chunk(
    model, audio_path: str, min_logprob: float = -0.5
) -> dict | None:
    """Transcribes a single audio chunk using Faster-Whisper.

    Filters out hallucinations if average log probability is lower than
    threshold.
    """
    segments, info = model.transcribe(
        audio_path,
        language="th",
        task="transcribe",
        beam_size=5,
        word_timestamps=False,
    )
    full_text = []
    logprobs = []

    for seg in segments:
        full_text.append(seg.text.strip())
        logprobs.append(seg.avg_logprob)

    if not full_text:
        return None

    combined_text = " ".join(full_text).strip()
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -1.0

    if avg_logprob < min_logprob:
        print(
            f"[ASR Filtered] {os.path.basename(audio_path)} discarded (Low logprob: {avg_logprob:.2f} < {min_logprob})"
        )
        return None

    return {
        "audio_path": audio_path,
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
    """Iterates through sliced audio chunks and produces metadata.jsonl."""
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    print(
        f"[ASR] Loading Faster-Whisper ({model_size}) on {device} ({compute_type})..."
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    wav_files = sorted(list(Path(wavs_dir).glob("*.wav")))
    print(f"[ASR] Transcribing {len(wav_files)} chunks...")

    valid_count = 0
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for idx, wav in enumerate(wav_files):
            result = transcribe_chunk(
                model, str(wav), min_logprob=min_logprob
            )
            if result:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                valid_count += 1
            if (idx + 1) % 50 == 0:
                print(f"[ASR Progress] Processed {idx + 1}/{len(wav_files)}")

    print(
        f"[ASR] Finished! {valid_count}/{len(wav_files)} chunks passed confidence threshold."
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
    args = parser.parse_args()

    transcribe_dataset(args.wavs_dir, args.output_jsonl, args.model_size)

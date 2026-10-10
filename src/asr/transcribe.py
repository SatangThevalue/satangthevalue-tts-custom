import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import time
from typing import Optional
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from src.utils.logger import setup_logger

logger = setup_logger("asr_transcribe")


def setup_cuda_shared_libraries():
    """Finds and pre-loads NVIDIA CUDA 12 and cuDNN shared libraries into process memory.
    Resolves 'Library libcublas.so.12 is not found or cannot be loaded' in Linux/Colab environments.
    """
    search_dirs = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]

    try:
        import site
        for s_dir in site.getsitepackages() + [site.getusersitepackages()]:
            nvidia_base = os.path.join(s_dir, "nvidia")
            if os.path.exists(nvidia_base):
                for sub in os.listdir(nvidia_base):
                    lib_dir = os.path.join(nvidia_base, sub, "lib")
                    if os.path.exists(lib_dir):
                        search_dirs.append(lib_dir)
    except Exception:
        pass

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    all_paths = [d for d in search_dirs if os.path.exists(d)]
    if all_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(all_paths) + ":" + current_ld

    # Pre-load required libraries with ctypes into global process namespace
    libs_to_load = [
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn_ops_infer.so.8",
        "libcudnn.so.8",
    ]
    for lib_name in libs_to_load:
        for d in all_paths:
            candidate = os.path.join(d, lib_name)
            if os.path.exists(candidate):
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                    logger.debug(f"Pre-loaded CUDA library: {candidate}")
                    break
                except Exception:
                    pass


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


def clean_corrupt_drive_cache(target_dir: str):
    """Detects and purges text symlinks created on Google Drive FUSE which cause
    'Unsupported model binary version (v774843950)' CTranslate2 errors.
    """
    p = Path(target_dir)
    if not p.exists():
        return

    for bin_file in list(p.rglob("model.bin")):
        try:
            if bin_file.stat().st_size < 1024 * 1024:  # True model.bin is > 1.5 GB
                with open(bin_file, "rb") as f:
                    head = f.read(4)
                    if head.startswith(b".") or head.startswith(b"vers"):
                        logger.warning(
                            f"Purging corrupt FUSE symlink pointer at: {bin_file}"
                        )
                        shutil.rmtree(bin_file.parent, ignore_errors=True)
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
            return None

        combined_text = " ".join(full_text).strip()
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else -1.0
        elapsed = time.perf_counter() - t0
        calc_duration = info.duration if hasattr(info, "duration") and info.duration > 0 else duration_sec

        logger.debug(
            f"Transcript: '{combined_text[:30]}...' | Duration: {calc_duration:.2f}s | avg_logprob: {avg_logprob:.3f} | Elapsed: {elapsed:.2f}s"
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


def init_whisper_engine(
    model_size: str = "large-v3",
    download_root: Optional[str] = None,
):
    """Initializes Faster-Whisper with automatic recovery from corrupted FUSE caches and CUDA fallbacks."""
    from faster_whisper import WhisperModel

    setup_cuda_shared_libraries()

    # If download_root is on Google Drive FUSE, purge any corrupt text symlinks
    if download_root and "/content/drive" in download_root:
        clean_corrupt_drive_cache(download_root)

    has_cuda = torch is not None and torch.cuda.is_available()

    # Try CUDA first if available
    if has_cuda:
        try:
            logger.info(f"Loading Faster-Whisper ({model_size}) on CUDA GPU...")
            model = WhisperModel(
                model_size,
                device="cuda",
                compute_type="float16",
                download_root=download_root,
            )
            # Run quick probe to verify CTranslate2 CUDA binary compatibility
            test_audio = np.zeros(8000, dtype=np.float32)
            _ = list(model.transcribe(test_audio, language="th")[0])
            logger.info("✅ CUDA CTranslate2 acceleration verified and operational!")
            return model
        except Exception as cuda_err:
            err_msg = str(cuda_err)
            logger.warning(f"⚠️ CUDA init notice: {err_msg}")
            # If corrupted binary on Google Drive FUSE was hit, purge and reset download_root
            if "Unsupported model binary version" in err_msg or "774843950" in err_msg:
                logger.warning("FUSE corrupt cache detected. Purging drive cache and switching to local NVMe...")
                if download_root and os.path.exists(download_root):
                    shutil.rmtree(download_root, ignore_errors=True)
                try:
                    return WhisperModel(model_size, device="cuda", compute_type="float16", download_root=None)
                except Exception:
                    pass

    # Fallback to CPU int8 with local NVMe caching
    logger.info(f"Loading Faster-Whisper ({model_size}) on CPU int8...")
    try:
        return WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=download_root,
        )
    except Exception as cpu_err:
        err_msg = str(cpu_err)
        if "Unsupported model binary version" in err_msg or "774843950" in err_msg:
            logger.warning("Purging corrupt cache and reloading on CPU with local NVMe storage...")
            if download_root and os.path.exists(download_root):
                shutil.rmtree(download_root, ignore_errors=True)
            return WhisperModel(model_size, device="cpu", compute_type="int8", download_root=None)
        raise


def transcribe_dataset(
    wavs_dir: str,
    output_jsonl: str,
    model_size: str = "large-v3",
    min_logprob: float = -0.5,
    download_root: Optional[str] = None,
) -> int:
    """Iterates through sliced audio chunks and produces metadata.jsonl with
    CUDA library auto-healing, permanent Google Drive caching, and CPU fallback.
    """
    model = init_whisper_engine(model_size=model_size, download_root=download_root)

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
                continue

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
                valid_count += 1
                existing_paths.add(abs_wav)

            if (idx + 1) % 25 == 0 or (idx + 1) == len(wav_files):
                f_out.flush()  # Periodic flush
                pct = ((idx + 1) / len(wav_files)) * 100
                logger.info(
                    f"🎙️ ASR Progress: [{idx + 1}/{len(wav_files)}] ({pct:.1f}%) | {valid_count} chunks transcribed."
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
        default=None,
        help="Optional cache root (defaults to fast local NVMe)",
    )
    args = parser.parse_args()

    transcribe_dataset(
        args.wavs_dir,
        args.output_jsonl,
        model_size=args.model_size,
        min_logprob=args.min_logprob,
        download_root=args.download_root,
    )

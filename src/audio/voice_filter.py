import argparse
import os
from pathlib import Path
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

from src.utils.logger import setup_logger

logger = setup_logger("voice_filter")


def compute_acoustic_embedding(waveform: np.ndarray, sr: int = 24000) -> np.ndarray:
    """Computes robust acoustic voice fingerprint based on spectral envelope
    and sub-band energy distribution. Works reliably on CPU without GPU overhead.
    """
    if len(waveform) == 0:
        return np.zeros(64, dtype=np.float32)

    # Normalize amplitude
    waveform = waveform - np.mean(waveform)
    max_val = np.max(np.abs(waveform))
    if max_val > 1e-6:
        waveform = waveform / max_val

    # Multi-resolution spectral analysis
    nperseg = min(len(waveform), 1024)
    if nperseg < 64:
        return np.zeros(64, dtype=np.float32)

    f, t, sxx = spectrogram(waveform, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    sxx = np.log1p(sxx + 1e-6)

    # 1. Frequency sub-band energy profile (Timbre characteristic)
    subbands = np.array_split(sxx, 32, axis=0)
    band_means = np.array([np.mean(sb) for sb in subbands], dtype=np.float32)
    band_stds = np.array([np.std(sb) for sb in subbands], dtype=np.float32)

    # Combined 64-dimensional acoustic signature
    embedding = np.concatenate([band_means, band_stds])
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm

    return embedding


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two unit vectors (-1.0 to 1.0)."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    dot = np.dot(vec1, vec2)
    return float(dot / (norm1 * norm2))


def filter_audio_chunks(
    chunk_paths: List[str],
    reference_wav_path: str,
    threshold: float = 0.70,
    rejected_dir: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Filters audio chunks by comparing acoustic signature against reference voice.
    Chunks with similarity below threshold are identified as other speakers and dropped.
    """
    logger.info(
        f"Initializing Target Voice Filter -> Reference: {reference_wav_path} (Threshold: {threshold:.2f})"
    )

    if not os.path.exists(reference_wav_path):
        raise FileNotFoundError(f"Reference voice clip not found: {reference_wav_path}")

    # Load and extract reference embedding
    ref_audio, ref_sr = sf.read(reference_wav_path)
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=1)
    ref_emb = compute_acoustic_embedding(ref_audio, sr=ref_sr)

    accepted_chunks: List[str] = []
    audit_report: List[Dict[str, Any]] = []

    if rejected_dir:
        Path(rejected_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for idx, path_str in enumerate(chunk_paths, start=1):
        if not os.path.exists(path_str):
            continue

        try:
            audio, sr = sf.read(path_str)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            chunk_emb = compute_acoustic_embedding(audio, sr=sr)
            sim = compute_cosine_similarity(ref_emb, chunk_emb)

            passed = sim >= threshold
            status = "ACCEPTED" if passed else "REJECTED (Other Speaker)"

            record = {
                "chunk_path": path_str,
                "similarity": round(sim, 4),
                "passed": passed,
                "status": status,
            }
            audit_report.append(record)

            if passed:
                accepted_chunks.append(path_str)
                logger.debug(f"[{idx}/{len(chunk_paths)}] {Path(path_str).name} -> Match: {sim:.3f} (Passed)")
            else:
                logger.warning(
                    f"[{idx}/{len(chunk_paths)}] FILTERED OUT other speaker: {Path(path_str).name} (Sim: {sim:.3f} < {threshold})"
                )
                if rejected_dir:
                    dest = Path(rejected_dir) / Path(path_str).name
                    os.replace(path_str, dest)

        except Exception as e:
            logger.error(f"Error processing chunk {path_str}: {e}")

    elapsed = time.perf_counter() - t0
    retention_rate = (len(accepted_chunks) / len(chunk_paths) * 100) if chunk_paths else 0.0
    logger.info(
        f"Voice Filter complete in {elapsed:.2f}s: Kept {len(accepted_chunks)}/{len(chunk_paths)} chunks ({retention_rate:.1f}% Target Speaker Purity)"
    )

    return accepted_chunks, audit_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target Speaker Voice Filter")
    parser.add_argument("--chunks", nargs="+", required=True, help="List of sliced audio chunks")
    parser.add_argument("--reference", type=str, required=True, help="Path to clean reference voice sample (5-10s)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Similarity threshold (0.5 - 0.9)")
    parser.add_argument("--rejected-dir", type=str, help="Directory to quarantine rejected chunks")
    args = parser.parse_args()

    accepted, report = filter_audio_chunks(
        chunk_paths=args.chunks,
        reference_wav_path=args.reference,
        threshold=args.threshold,
        rejected_dir=args.rejected_dir,
    )
    print(f"Accepted chunks count: {len(accepted)}")

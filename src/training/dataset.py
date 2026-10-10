import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

from src.utils.logger import setup_logger

logger = setup_logger("dataset")


class TTSVoiceDataset(Dataset):
    """Loads processed audio segments along with their tone-tagged phoneme

    sequences with corrupted-audio resilience.
    """

    def __init__(
        self,
        metadata_path: str,
        target_sr: int = 24000,
        speaker_id: str | None = None,
    ):
        self.items = []
        self.target_sr = target_sr
        self.speaker_id = speaker_id

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file does not exist: {metadata_path}"
            )

        logger.info(f"Loading dataset metadata from: {metadata_path}")
        skipped = 0

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        item = json.loads(line)
                        if self.speaker_id and item.get("speaker") != self.speaker_id:
                            continue
                        audio_path = item.get("audio_path", "")
                        if Path(audio_path).exists():
                            self.items.append(item)
                        else:
                            skipped += 1
                            logger.debug(
                                f"Skipping missing audio path (line {line_no}): {audio_path}"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error parsing metadata line {line_no}: {e}"
                        )
                        skipped += 1

        logger.info(
            f"Dataset loaded: {len(self.items)} valid audio samples (Skipped/Missing: {skipped})"
        )
        if len(self.items) == 0:
            raise ValueError(
                f"Dataset at {metadata_path} contains 0 valid samples. Check your audio paths!"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        wav_path = item["audio_path"]

        # High-speed local NVMe redirection: bypass Google Drive FUSE latency if local cache exists
        if "/content/drive/MyDrive/tts-project/02_processed" in wav_path:
            local_alt = wav_path.replace(
                "/content/drive/MyDrive/tts-project/02_processed",
                "/content/dataset_local"
            )
            if os.path.exists(local_alt):
                wav_path = local_alt

        try:
            import soundfile as sf
            audio_np, sr = sf.read(wav_path)
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)

            if sr != self.target_sr:
                from scipy.signal import resample
                num_target = int(len(audio_np) * self.target_sr / sr)
                audio_np = resample(audio_np, num_target)

            waveform = torch.from_numpy(audio_np.astype(np.float32))

            return {
                "waveform": waveform,
                "text": item.get("normalized_text", ""),
                "phonemes": item.get("phonemes", ""),
                "audio_path": wav_path,
                "valid": True,
            }
        except Exception as e:
            logger.error(f"Error loading waveform from {wav_path}: {e}")
            dummy_wave = torch.zeros(self.target_sr, dtype=torch.float32)
            return {
                "waveform": dummy_wave,
                "text": item.get("normalized_text", ""),
                "phonemes": item.get("phonemes", ""),
                "audio_path": wav_path,
                "valid": False,
            }


def collate_fn(batch: list[dict]) -> dict:
    """Collates variable-length audio waveforms and token sequences with

    padding.
    """
    valid_batch = [b for b in batch if b.get("valid", True)]
    if not valid_batch:
        valid_batch = batch

    waveforms = [b["waveform"] for b in valid_batch]
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = torch.max(lengths).item()

    padded_waves = torch.zeros(
        len(waveforms), max_len, dtype=torch.float32
    )
    for i, w in enumerate(waveforms):
        padded_waves[i, : w.shape[0]] = w

    logger.debug(
        f"Collate: Batch Size={len(valid_batch)}, Max Audio Length={max_len} samples ({max_len / 24000:.2f}s)"
    )

    return {
        "waveforms": padded_waves,
        "lengths": lengths,
        "texts": [b["text"] for b in valid_batch],
        "phonemes": [b["phonemes"] for b in valid_batch],
        "audio_paths": [b["audio_path"] for b in valid_batch],
    }

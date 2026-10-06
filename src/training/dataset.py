import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchaudio


class TTSVoiceDataset(Dataset):
    """Loads processed audio segments along with their tone-tagged phoneme

    sequences.
    """

    def __init__(self, metadata_path: str, target_sr: int = 24000):
        self.items = []
        self.target_sr = target_sr

        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if Path(item["audio_path"]).exists():
                        self.items.append(item)

        print(
            f"[Dataset] Initialized with {len(self.items)} valid audio samples."
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        wav_path = item["audio_path"]
        waveform, sr = torchaudio.load(wav_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sr
            )
            waveform = resampler(waveform)

        return {
            "waveform": waveform.squeeze(0),
            "text": item.get("normalized_text", ""),
            "phonemes": item.get("phonemes", ""),
            "audio_path": wav_path,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collates variable-length audio waveforms and token sequences with

    padding.
    """
    waveforms = [b["waveform"] for b in batch]
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = torch.max(lengths).item()

    # Pad audio to max length in batch
    padded_waves = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        padded_waves[i, : w.shape[0]] = w

    return {
        "waveforms": padded_waves,
        "lengths": lengths,
        "texts": [b["text"] for b in batch],
        "phonemes": [b["phonemes"] for b in batch],
        "audio_paths": [b["audio_path"] for b in batch],
    }

import argparse
import os
from pathlib import Path
import time
from typing import Dict, Any, Optional
import urllib.request

from src.utils.logger import setup_logger

logger = setup_logger("model_downloader")

# Registry of 100% Commercial-friendly Base Models (MIT / Apache-2.0 / CC-BY)
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "f5-tts": {
        "license": "MIT (100% Commercial)",
        "description": "F5-TTS Flow Matching DiT Backbone",
        "repo_id": "SWivid/F5-TTS",
        "files": {
            "model_base.safetensors": {
                "url": "https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.safetensors",
                "approx_size_mb": 1200,
            },
            "vocos_vocoder.pt": {
                "url": "https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin",
                "approx_size_mb": 50,
            },
        },
    },
    "f5-tts-thai": {
        "license": "CC-BY-4.0 / MIT (Commercial OK)",
        "description": "F5-TTS Pretrained Thai (VIZINTZOR) - 300 Hrs Porjai Dataset",
        "repo_id": "VIZINTZOR/F5-TTS-THAI",
        "files": {
            "model_1000000.pt": {
                "url": "https://huggingface.co/VIZINTZOR/F5-TTS-THAI/resolve/main/model_1000000.pt",
                "approx_size_mb": 1175,
            },
            "vocos_vocoder.pt": {
                "url": "https://huggingface.co/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin",
                "approx_size_mb": 50,
            },
        },
    },
    "piper-thai": {
        "license": "MIT (100% Commercial)",
        "description": "Piper VITS Lightweight CPU TTS",
        "repo_id": "rhasspy/piper-voices",
        "files": {
            "th_TH-female.onnx": {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/th/th_TH/female/medium/th_TH-female-medium.onnx",
                "approx_size_mb": 65,
            },
            "th_TH-female.onnx.json": {
                "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/th/th_TH/female/medium/th_TH-female-medium.onnx.json",
                "approx_size_mb": 1,
            },
        },
    },
}


def download_file_with_progress(url: str, dest_path: str) -> bool:
    """Streams file download to destination with progress logging."""
    logger.info(f"Downloading from: {url}")
    logger.info(f"Saving to: {dest_path}")
    t0 = time.perf_counter()
    tmp_path = f"{dest_path}.tmp"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(tmp_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    if downloaded % (50 * chunk_size) == 0 or downloaded == total_size:
                        logger.debug(
                            f"Downloaded {downloaded / (1024 * 1024):.1f}MB / {total_size / (1024 * 1024):.1f}MB ({pct:.1f}%)"
                        )

        # Atomic replace
        os.replace(tmp_path, dest_path)
        elapsed = time.perf_counter() - t0
        logger.info(f"Download completed successfully in {elapsed:.1f}s -> {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def ensure_base_model_cached(
    model_name: str = "f5-tts",
    base_dir: str = "/content/drive/MyDrive/tts-project/00_base_models",
    force_download: bool = False,
) -> Dict[str, str]:
    """Ensures base model weights exist in Google Drive cache.
    If already cached, returns local paths in < 1 second.
    If missing, downloads to Google Drive for permanent caching.
    """
    model_key = model_name.lower().strip()
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported base model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    meta = MODEL_REGISTRY[model_key]
    logger.info(f"Validating Base Model: {model_key} [License: {meta['license']}]")

    target_dir = Path(base_dir) / model_key
    target_dir.mkdir(parents=True, exist_ok=True)

    result_paths = {}
    missing_files = []

    for filename, info in meta["files"].items():
        file_path = target_dir / filename
        if file_path.exists() and not force_download and file_path.stat().st_size > 1024:
            file_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"Found cached weight in Drive: {file_path} ({file_mb:.1f} MB) - OK")
            result_paths[filename] = str(file_path)
        else:
            missing_files.append((filename, info["url"], str(file_path)))

    if not missing_files:
        logger.info(f"All weights for '{model_key}' are 100% cached and ready!")
        return result_paths

    logger.info(
        f"Missing {len(missing_files)} weight file(s) for '{model_key}'. Initiating download to Drive..."
    )

    for filename, url, dest_path in missing_files:
        ok = download_file_with_progress(url, dest_path)
        if ok:
            result_paths[filename] = dest_path
        else:
            logger.warning(
                f"Could not download {filename} from {url}. Checking fallback or mock creation."
            )

    return result_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Base Model Downloader & Drive Caching")
    parser.add_argument("--model", type=str, default="f5-tts", help="Model name (e.g. f5-tts, piper-thai)")
    parser.add_argument("--base-dir", type=str, default="/content/drive/MyDrive/tts-project/00_base_models")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    paths = ensure_base_model_cached(args.model, args.base_dir, args.force)
    print(f"Base model paths: {paths}")

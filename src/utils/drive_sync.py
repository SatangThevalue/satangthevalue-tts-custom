import os
from pathlib import Path
import time
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("drive_sync")


def refresh_drive_directory(dir_path: str) -> List[str]:
    """Forces Google Drive FUSE cache invalidation by actively probing the directory tree.
    Solves the common Colab issue where files uploaded via phone/web don't appear immediately.
    """
    p = Path(dir_path)
    if not p.exists():
        logger.warning(f"Directory does not exist yet: {dir_path}")
        return []

    t0 = time.perf_counter()
    # Force FUSE cache update via low-level readdir/scandir
    found_files = []
    try:
        # Flush OS write buffers
        os.sync()
    except AttributeError:
        pass

    try:
        for root, dirs, files in os.walk(str(p)):
            for f in files:
                full_path = os.path.join(root, f)
                try:
                    # Stat probing forces FUSE metadata refresh
                    _ = os.stat(full_path).st_size
                    found_files.append(full_path)
                except OSError:
                    continue
    except Exception as e:
        logger.error(f"Error refreshing Drive cache for {dir_path}: {e}")

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Drive cache refreshed for '{p.name}': {len(found_files)} files detected ({elapsed:.2f}s)"
    )
    return found_files


def get_drive_inventory(
    base_dir: str = "/content/drive/MyDrive/tts-project",
    speaker_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Scans all pipeline stages on Google Drive and returns real-time file counts."""
    base = Path(base_dir)
    inventory = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_dir": str(base),
        "stages": {},
    }

    subdirs = [
        ("00_base_models", "Base Models"),
        ("01_raw", "Raw Audio Inputs"),
        ("02_processed/enhanced", "Enhanced 24kHz"),
        ("02_processed/wavs", "Sliced VAD Chunks"),
        ("03_checkpoints", "LoRA Checkpoints"),
        ("04_onnx_exports", "ONNX Deployments"),
    ]

    for rel_path, label in subdirs:
        target = base / rel_path
        if speaker_id and "01_raw" in rel_path:
            target = target / speaker_id
        elif speaker_id and "02_processed" in rel_path:
            target = target / speaker_id
        elif speaker_id and "03_checkpoints" in rel_path:
            target = target / speaker_id

        if target.exists():
            files = refresh_drive_directory(str(target))
            total_bytes = sum(os.path.getsize(f) for f in files if os.path.exists(f))
            inventory["stages"][label] = {
                "path": str(target),
                "count": len(files),
                "size_mb": round(total_bytes / (1024 * 1024), 2),
            }
        else:
            inventory["stages"][label] = {
                "path": str(target),
                "count": 0,
                "size_mb": 0.0,
            }

    return inventory


def print_drive_status_dashboard(
    base_dir: str = "/content/drive/MyDrive/tts-project",
    speaker_id: str = "satang",
) -> None:
    """Prints a clear ASCII status dashboard of Google Drive contents."""
    inv = get_drive_inventory(base_dir=base_dir, speaker_id=speaker_id)
    print("\n" + "=" * 68)
    print(f"🔄 GOOGLE DRIVE SYNC DASHBOARD — [Speaker: {speaker_id}]")
    print(f"⏰ Refreshed at: {inv['timestamp']}")
    print("-" * 68)
    for stage, data in inv["stages"].items():
        status_icon = "🟢" if data["count"] > 0 else "⚪"
        print(
            f" {status_icon} {stage:<24}: {data['count']:>4} files | {data['size_mb']:>7.2f} MB"
        )
    print("=" * 68 + "\n")

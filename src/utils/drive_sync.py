import os
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Tuple

from src.utils.logger import setup_logger

logger = setup_logger("drive_sync")


def scan_drive_directory(dir_path: str, label: Optional[str] = None) -> Tuple[List[str], int]:
    """Forces Google Drive FUSE cache invalidation by actively probing the directory tree.
    Computes file list and total size in a single pass to eliminate redundant FUSE roundtrips.
    """
    p = Path(dir_path)
    if not p.exists():
        logger.warning(f"Directory does not exist yet: {dir_path}")
        return [], 0

    t0 = time.perf_counter()
    found_files = []
    total_bytes = 0
    try:
        # Flush OS write buffers
        os.sync()
    except AttributeError:
        pass

    try:
        for root, _, files in os.walk(str(p)):
            for f in files:
                full_path = os.path.join(root, f)
                try:
                    # Stat probing forces FUSE metadata refresh while capturing file size
                    total_bytes += os.stat(full_path).st_size
                    found_files.append(full_path)
                except OSError:
                    continue
    except Exception as e:
        logger.error(f"Error refreshing Drive cache for {dir_path}: {e}")

    elapsed = time.perf_counter() - t0
    display_name = label or (f"{p.parent.name}/{p.name}" if p.parent.name else p.name)
    logger.info(
        f"Drive cache refreshed for '{display_name}': {len(found_files)} files detected ({elapsed:.2f}s)"
    )
    return found_files, total_bytes


def refresh_drive_directory(dir_path: str, label: Optional[str] = None) -> List[str]:
    """Forces Google Drive FUSE cache invalidation by actively probing the directory tree.
    Solves the common Colab issue where files uploaded via phone/web don't appear immediately.
    """
    files, _ = scan_drive_directory(dir_path, label=label)
    return files


def get_pipeline_recommendation(stages: Dict[str, Dict[str, Any]]) -> str:
    """Evaluates inventory counts to recommend the immediate next pipeline step."""
    base_models = stages.get("Base Models", {}).get("count", 0)
    raw = stages.get("Raw Audio Inputs", {}).get("count", 0)
    enhanced = stages.get("Enhanced 24kHz", {}).get("count", 0)
    chunks = stages.get("Sliced VAD Chunks", {}).get("count", 0)
    checkpoints = stages.get("LoRA Checkpoints", {}).get("count", 0)
    onnx = stages.get("ONNX Deployments", {}).get("count", 0)

    if base_models == 0:
        return "⚠️ Base Model Missing -> Run [STEP 1.1] Setup & Base Weights Download"
    if onnx > 0:
        return "✅ Model Deployed -> Ready for ONNX Inference & Testing"
    if checkpoints > 0:
        return "📦 Checkpoints Available -> Ready for ONNX Export [STEP 4]"
    if chunks > 0:
        return f"🔥 {chunks} VAD Chunks Ready -> Ready for LoRA Training [STEP 3]"
    if enhanced > 0:
        return "✂️ Enhanced Audio Ready -> Run Slicing & G2P ASR [STEP 2.3]"
    if raw > 0:
        return "🎛️ Raw Audio Available -> Run Audio Enhancement [STEP 2.2]"
    return "📥 No Audio Data -> Download or Upload Audio Files [STEP 2.1]"


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
        stage_tag = label
        if speaker_id and ("01_raw" in rel_path or "02_processed" in rel_path or "03_checkpoints" in rel_path):
            target = target / speaker_id
            stage_tag = f"{label} ({rel_path}/{speaker_id})"
        else:
            stage_tag = f"{label} ({rel_path})"

        if target.exists():
            files, total_bytes = scan_drive_directory(str(target), label=stage_tag)
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
    """Prints a clear ASCII status dashboard of Google Drive contents with next step guidance."""
    inv = get_drive_inventory(base_dir=base_dir, speaker_id=speaker_id)
    print("\n" + "=" * 68)
    print(f"🔄 GOOGLE DRIVE SYNC DASHBOARD — [Speaker: {speaker_id}]")
    print(f"⏰ Refreshed at: {inv['timestamp']}")
    print("-" * 68)
    for stage, data in inv["stages"].items():
        status_icon = "🟢" if data["count"] > 0 else "⚪"
        size_mb = data["size_mb"]
        size_str = f"{size_mb / 1024:>6.2f} GB" if size_mb >= 1024 else f"{size_mb:>7.2f} MB"
        print(
            f" {status_icon} {stage:<24}: {data['count']:>4} files | {size_str}"
        )
    print("-" * 68)
    rec = get_pipeline_recommendation(inv["stages"])
    print(f" 💡 Next Step: {rec}")
    print("=" * 68 + "\n")

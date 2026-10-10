"""
Checkpoint & Model Version Lifecycle Manager.
Supports:
- Inspecting existing checkpoint steps and sizes per speaker
- Archiving trained runs into isolated version folders (v1.0, v2.0, etc.)
- Resetting active checkpoints to start fresh training from step 0
- Restoring archived checkpoints
"""
import glob
import os
from pathlib import Path
import shutil
import time
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("checkpoint_manager")

DEFAULT_CKPT_BASE = "/content/drive/MyDrive/tts-project/03_checkpoints"


def list_speaker_checkpoints(
    speaker_id: str,
    base_dir: str = DEFAULT_CKPT_BASE,
) -> Dict[str, Any]:
    """Lists all active and archived checkpoints for a given speaker."""
    spk_dir = Path(base_dir) / speaker_id
    if not spk_dir.exists():
        return {
            "speaker_id": speaker_id,
            "status": "EMPTY",
            "active_steps": [],
            "archived_versions": [],
            "total_size_mb": 0.0,
        }

    # 1. Active Step Checkpoints (step_*)
    active_steps = []
    total_bytes = 0

    for step_dir in sorted(spk_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]) if "_" in p.name and p.name.split("_")[1].isdigit() else 0):
        if step_dir.is_dir():
            try:
                num = int(step_dir.name.split("_")[1])
            except ValueError:
                num = 0

            # Calculate size
            dir_bytes = sum(f.stat().st_size for f in step_dir.glob("*") if f.is_file())
            total_bytes += dir_bytes
            has_weights = (step_dir / "adapter_model.pt").exists()
            active_steps.append({
                "step": num,
                "path": str(step_dir),
                "size_mb": round(dir_bytes / (1024 * 1024), 2),
                "has_adapter": has_weights,
            })

    # 2. Archived Versions
    archived_versions = []
    archive_base = spk_dir / "archives"
    if archive_base.exists():
        for ver_dir in sorted(archive_base.iterdir()):
            if ver_dir.is_dir():
                ver_bytes = sum(f.stat().st_size for f in ver_dir.rglob("*") if f.is_file())
                total_bytes += ver_bytes
                steps_in_ver = len(list(ver_dir.glob("step_*")))
                archived_versions.append({
                    "version_tag": ver_dir.name,
                    "path": str(ver_dir),
                    "steps_count": steps_in_ver,
                    "size_mb": round(ver_bytes / (1024 * 1024), 2),
                })

    return {
        "speaker_id": speaker_id,
        "status": "FOUND",
        "active_steps": active_steps,
        "archived_versions": archived_versions,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
    }


def archive_checkpoints(
    speaker_id: str,
    version_tag: str,
    base_dir: str = DEFAULT_CKPT_BASE,
) -> Dict[str, Any]:
    """Moves all active step_* checkpoints into an isolated archive folder.
    Frees up active space so next training run starts fresh from Step 0 without losing prior model history.
    """
    spk_dir = Path(base_dir) / speaker_id
    if not spk_dir.exists():
        raise FileNotFoundError(f"Speaker directory not found: {spk_dir}")

    active_steps = list(spk_dir.glob("step_*"))
    if not active_steps:
        logger.warning(f"No active checkpoints found to archive for speaker '{speaker_id}'.")
        return {"status": "NO_ACTIVE_STEPS", "archived_steps": 0}

    archive_target = spk_dir / "archives" / version_tag
    archive_target.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    for step_dir in active_steps:
        dest = archive_target / step_dir.name
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(step_dir), str(dest))
        moved_count += 1

    logger.info(
        f"Archived {moved_count} checkpoints for '{speaker_id}' -> {archive_target}"
    )
    return {
        "status": "SUCCESS",
        "speaker_id": speaker_id,
        "version_tag": version_tag,
        "archived_steps": moved_count,
        "archive_path": str(archive_target),
    }


def reset_checkpoints(
    speaker_id: str,
    base_dir: str = DEFAULT_CKPT_BASE,
    backup_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Deletes active step_* checkpoints to start training from Step 0.
    Optionally auto-archives them to a timestamped backup first.
    """
    spk_dir = Path(base_dir) / speaker_id
    if not spk_dir.exists():
        return {"status": "EMPTY", "deleted_steps": 0}

    active_steps = list(spk_dir.glob("step_*"))
    if not active_steps:
        return {"status": "ALREADY_EMPTY", "deleted_steps": 0}

    if backup_tag:
        archive_checkpoints(speaker_id, backup_tag, base_dir=base_dir)
        return {
            "status": "ARCHIVED_AND_RESET",
            "speaker_id": speaker_id,
            "backup_version": backup_tag,
            "deleted_steps": len(active_steps),
        }

    # Delete active step directories
    deleted_count = 0
    for step_dir in active_steps:
        shutil.rmtree(step_dir, ignore_errors=True)
        deleted_count += 1

    logger.info(f"Purged {deleted_count} active checkpoints for speaker '{speaker_id}'.")
    return {
        "status": "PURGED",
        "speaker_id": speaker_id,
        "deleted_steps": deleted_count,
    }


def restore_archived_checkpoints(
    speaker_id: str,
    version_tag: str,
    base_dir: str = DEFAULT_CKPT_BASE,
) -> Dict[str, Any]:
    """Restores checkpoints from an archive back to active step_* directories."""
    spk_dir = Path(base_dir) / speaker_id
    archive_dir = spk_dir / "archives" / version_tag
    if not archive_dir.exists():
        raise FileNotFoundError(f"Archive version '{version_tag}' not found at {archive_dir}")

    archived_steps = list(archive_dir.glob("step_*"))
    if not archived_steps:
        return {"status": "NO_STEPS_IN_ARCHIVE", "restored_count": 0}

    restored_count = 0
    for step_dir in archived_steps:
        dest = spk_dir / step_dir.name
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(str(step_dir), str(dest))
        restored_count += 1

    logger.info(f"Restored {restored_count} checkpoints from '{version_tag}' to active training folder.")
    return {
        "status": "RESTORED",
        "speaker_id": speaker_id,
        "version_tag": version_tag,
        "restored_count": restored_count,
    }

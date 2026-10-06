import os
from pathlib import Path
import tarfile

from src.utils.logger import setup_logger

logger = setup_logger("pack")


def pack_dataset(source_dir: str, output_tar_path: str) -> str:
    """Packs individual wav chunks and metadata into a single tar archive.

    Drastically speeds up Google Drive sync by eliminating 1000s of small I/O
    calls.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    output_tar = Path(output_tar_path)
    output_tar.parent.mkdir(parents=True, exist_ok=True)

    files_to_pack = [f for f in source_path.rglob("*") if f.is_file()]
    logger.debug(
        f"Packing {len(files_to_pack)} files from {source_dir} -> {output_tar_path}"
    )

    with tarfile.open(output_tar, "w") as tar:
        for idx, file in enumerate(files_to_pack):
            arcname = file.relative_to(source_path)
            tar.add(file, arcname=arcname)
            if (idx + 1) % 500 == 0:
                logger.debug(f"Packed {idx + 1}/{len(files_to_pack)} files...")

    tar_size_mb = output_tar.stat().st_size / (1024 * 1024)
    logger.info(
        f"Successfully packed {len(files_to_pack)} files into {output_tar_path} ({tar_size_mb:.2f} MB)"
    )
    return str(output_tar)


def unpack_dataset(tar_path: str, extract_to: str) -> str:
    """Unpacks the dataset archive directly to local high-speed disk (e.g.

    /content/ in Colab).
    """
    tar_file = Path(tar_path)
    if not tar_file.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    dest_path = Path(extract_to)
    dest_path.mkdir(parents=True, exist_ok=True)

    logger.debug(
        f"Unpacking {tar_path} ({tar_file.stat().st_size / (1024 * 1024):.2f} MB) -> {extract_to}..."
    )

    with tarfile.open(tar_file, "r") as tar:
        tar.extractall(dest_path)

    extracted_count = len(list(dest_path.rglob("*")))
    logger.info(
        f"Unpacked archive successfully. Extracted {extracted_count} items to {extract_to}"
    )
    return str(dest_path)

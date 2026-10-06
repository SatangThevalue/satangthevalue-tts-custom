import os
import tarfile
from pathlib import Path


def pack_dataset(source_dir: str, output_tar_path: str) -> str:
    """Packs individual wav chunks and metadata into a single tar archive.

    Drastically speeds up Google Drive sync by eliminating 1000s of small I/O
    calls.
    """
    source_path = Path(source_dir)
    output_tar = Path(output_tar_path)
    output_tar.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output_tar, "w") as tar:
        for file in source_path.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(source_path)
                tar.add(file, arcname=arcname)
    return str(output_tar)


def unpack_dataset(tar_path: str, extract_to: str) -> str:
    """Unpacks the dataset archive directly to local high-speed disk (e.g.

    /content/ in Colab).
    """
    dest_path = Path(extract_to)
    dest_path.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest_path)
    return str(dest_path)

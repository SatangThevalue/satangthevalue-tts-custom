import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import time
from typing import List, Dict, Any, Optional

from src.utils.logger import setup_logger

logger = setup_logger("audio_download")


def sanitize_filename(name: str) -> str:
    """Sanitizes title string into a filesystem-safe filename."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = re.sub(r"\s+", "_", clean).strip(" ._-")
    return clean[:60] if clean else "audio_track"


def _download_single_task(
    url_str: str,
    target_dir: Path,
    ydl_opts: Dict[str, Any],
    speaker_id: str,
) -> Dict[str, Any]:
    """Worker task executing individual URL extraction and audio conversion."""
    import yt_dlp

    t0 = time.perf_counter()
    logger.info(f"[PARALLEL TASK] Initiating download: {url_str}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
            info = ydl.extract_info(url_str, download=True)
            if not info:
                logger.warning(
                    f"Could not extract info from URL: {url_str} (May require login or is private)"
                )
                return {
                    "url": url_str,
                    "speaker_id": speaker_id,
                    "status": "error",
                    "error": "No info extracted",
                }

            title = str(info.get("title") or f"track_{int(time.time())}")
            duration = info.get("duration", 0)
            video_id = str(info.get("id") or "unknown")
            sanitized_title = sanitize_filename(title)
            expected_wav = target_dir / f"{sanitized_title}_{video_id}.wav"

            # Check if file was extracted
            if not expected_wav.exists():
                candidates = list(target_dir.glob(f"*{video_id}*.wav"))
                if candidates:
                    expected_wav = candidates[0]

            elapsed = time.perf_counter() - t0
            logger.info(
                f"[DOWNLOAD FINISHED] '{title}' ({duration}s) -> {expected_wav.name} (Elapsed: {elapsed:.1f}s)"
            )

            return {
                "url": url_str,
                "title": title,
                "video_id": video_id,
                "duration_sec": duration,
                "speaker_id": speaker_id,
                "output_path": str(expected_wav),
                "status": "success",
            }

    except Exception as e:
        logger.error(f"[DOWNLOAD ERROR] Failed {url_str}: {e}")
        return {
            "url": url_str,
            "speaker_id": speaker_id,
            "status": "error",
            "error": str(e),
        }


def download_media_audio(
    urls: List[str],
    output_dir: str = "/content/drive/MyDrive/tts-project/01_raw",
    speaker_id: str = "default",
    cookies_file: Optional[str] = None,
    target_sr: int = 24000,
    max_workers: int = 3,
) -> List[Dict[str, Any]]:
    """Downloads audio streams from YouTube, Facebook, and Instagram concurrently.
    Converts directly to uniform 24kHz Mono 16-bit PCM WAV in 01_raw/{speaker_id}/.
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is required. Install via: pip install yt-dlp")
        raise

    target_dir = Path(output_dir) / speaker_id
    target_dir.mkdir(parents=True, exist_ok=True)

    clean_urls = [u.strip() for u in urls if u.strip() and not u.strip().startswith("#")]
    if not clean_urls:
        logger.warning("No valid URLs provided to download.")
        return []

    workers = min(max_workers, len(clean_urls))
    logger.info(
        f"Starting parallel download of {len(clean_urls)} URLs (Workers: {workers}) for Speaker: '{speaker_id}' -> {target_dir}"
    )

    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(target_dir / "%(title).60s_%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "postprocessor_args": [
            "-ar", str(target_sr),
            "-ac", "1",
        ],
        "quiet": False,
        "no_warnings": False,
        "ignoreerrors": True,
        "nocheckcertificate": True,
    }

    if cookies_file and os.path.exists(cookies_file):
        logger.info(f"Using cookies file: {cookies_file}")
        ydl_opts["cookiefile"] = cookies_file

    results: List[Dict[str, Any]] = []

    if workers <= 1:
        for url_str in clean_urls:
            res = _download_single_task(url_str, target_dir, ydl_opts, speaker_id)
            results.append(res)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_url = {
                executor.submit(_download_single_task, url, target_dir, ydl_opts, speaker_id): url
                for url in clean_urls
            }
            for future in as_completed(future_to_url):
                res = future.result()
                results.append(res)

    success_count = sum(1 for r in results if r.get("status") == "success")
    logger.info(
        f"Parallel download batch finished: {success_count}/{len(results)} items downloaded successfully."
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Social Media Audio Ingestion (YT, FB, IG)")
    parser.add_argument("--urls", nargs="+", help="One or more media URLs")
    parser.add_argument("--url-file", type=str, help="Text file containing URLs (one per line)")
    parser.add_argument("--out-dir", type=str, default="/content/drive/MyDrive/tts-project/01_raw")
    parser.add_argument("--speaker-id", type=str, default="default", help="Speaker identifier")
    parser.add_argument("--cookies", type=str, help="Path to cookies.txt (optional)")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent download threads")
    args = parser.parse_args()

    url_list = []
    if args.urls:
        url_list.extend(args.urls)
    if args.url_file and os.path.exists(args.url_file):
        with open(args.url_file, "r", encoding="utf-8") as f:
            url_list.extend([line.strip() for line in f if line.strip()])

    if not url_list:
        print("Please provide --urls or --url-file")
    else:
        download_media_audio(
            urls=url_list,
            output_dir=args.out_dir,
            speaker_id=args.speaker_id,
            cookies_file=args.cookies,
            max_workers=args.workers,
        )

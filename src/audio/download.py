import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import re
import shutil
import time
from typing import List, Dict, Any, Optional

from src.utils.logger import setup_logger
from src.utils.registry import record_media_item

logger = setup_logger("audio_download")


def extract_media_id_from_url(url: str) -> Optional[str]:
    """Extracts unique video/reel ID from common YouTube, Facebook, and Instagram URLs."""
    url_str = url.strip()
    # YouTube (youtu.be, watch?v=, shorts/, live/)
    m_yt = re.search(r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|shorts/|live/))([A-Za-z0-9_-]{11})", url_str)
    if m_yt:
        return m_yt.group(1)

    # Instagram (reel/, p/)
    m_ig = re.search(r"instagram\.com/(?:reel|p)/([A-Za-z0-9_-]+)", url_str)
    if m_ig:
        return m_ig.group(1)

    # Facebook (reel/, videos/)
    m_fb = re.search(r"facebook\.com/(?:reel|videos)/([0-9]+)", url_str)
    if m_fb:
        return m_fb.group(1)

    return None


def sanitize_filename(name: str) -> str:
    """Sanitizes title string into a filesystem-safe filename."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = re.sub(r"\s+", "_", clean).strip(" ._-")
    return clean[:60] if clean else "audio_track"


def _download_single_task(
    url_str: str,
    target_dir: Path,
    staging_dir: Path,
    base_ydl_opts: Dict[str, Any],
    speaker_id: str,
) -> Dict[str, Any]:
    """Worker task executing download in local NVMe staging, converting to 24kHz WAV,
    and atomically syncing to Google Drive.
    """
    import yt_dlp

    t0 = time.perf_counter()

    # 1. Instant Duplicate Check
    pre_id = extract_media_id_from_url(url_str)
    if pre_id:
        target_wav = target_dir / f"{pre_id}.wav"
        if target_wav.exists() and target_wav.stat().st_size > 1024:
            logger.info(
                f"[INSTANT SKIP] '{pre_id}' already exists in Drive -> {target_wav.name} (0.00s)"
            )
            # Ensure recorded in registry
            try:
                record_media_item(
                    speaker_id=speaker_id,
                    source_type="social_media",
                    source_identifier=url_str,
                    raw_path=str(target_wav),
                    duration_sec=0.0,
                    status="raw",
                )
            except Exception:
                pass

            return {
                "url": url_str,
                "title": f"cached_{pre_id}",
                "video_id": pre_id,
                "duration_sec": 0,
                "speaker_id": speaker_id,
                "output_path": str(target_wav),
                "status": "cached",
            }

    logger.info(f"[PARALLEL TASK] Initiating download: {url_str}")

    # Configure worker-specific staging template
    ydl_opts = dict(base_ydl_opts)
    ydl_opts["outtmpl"] = str(staging_dir / "%(id)s.%(ext)s")

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

            # Handle playlists or single video
            if "entries" in info and info["entries"]:
                entries = [e for e in info["entries"] if e]
                first_entry = entries[0]
                title = str(first_entry.get("title") or "playlist_media")
                duration = first_entry.get("duration", 0)
                video_id = str(first_entry.get("id") or "").strip()
            else:
                title = str(info.get("title") or "media_track")
                duration = info.get("duration", 0)
                video_id = str(info.get("id") or "").strip()

            if not video_id:
                video_id = hashlib.md5(url_str.encode()).hexdigest()[:11]

            # Find generated WAV in local NVMe staging
            staging_wav = staging_dir / f"{video_id}.wav"
            if not staging_wav.exists():
                candidates = list(staging_dir.glob(f"*{video_id}*.wav"))
                if candidates:
                    staging_wav = candidates[0]

            if not staging_wav.exists():
                raise FileNotFoundError(f"Staging audio extraction failed for video ID {video_id}")

            # Atomically move from staging to final Google Drive directory
            dest_drive_wav = target_dir / f"{video_id}.wav"
            shutil.copyfile(str(staging_wav), str(dest_drive_wav))
            staging_wav.unlink(missing_ok=True)

            elapsed = time.perf_counter() - t0
            logger.info(
                f"[DOWNLOAD FINISHED] ID: {video_id} ('{title[:40]}...', {duration}s) -> {dest_drive_wav.name} (Elapsed: {elapsed:.1f}s)"
            )

            # Auto-register to SQLite registry database
            try:
                record_media_item(
                    speaker_id=speaker_id,
                    source_type="social_media",
                    source_identifier=url_str,
                    raw_path=str(dest_drive_wav),
                    duration_sec=float(duration or 0),
                    status="raw",
                )
            except Exception as reg_e:
                logger.debug(f"Registry log notice: {reg_e}")

            return {
                "url": url_str,
                "title": title,
                "video_id": video_id,
                "duration_sec": duration,
                "speaker_id": speaker_id,
                "output_path": str(dest_drive_wav),
                "status": "success",
            }

    except Exception as e:
        err_msg = str(e)
        hint = ""
        if "Private video" in err_msg or "Sign in" in err_msg or "login" in err_msg.lower():
            hint = " [คำแนะนำ: วิดีโอนี้อาจเป็นแบบส่วนตัว หรือติดจำกัดอายุ กรุณาแนบไฟล์ cookies.txt เพื่อยืนยันตัวตน]"
        elif "HTTP Error 429" in err_msg:
            hint = " [คำแนะนำ: เซิร์ฟเวอร์ต้นทางจำกัดอัตราดาวน์โหลดชั่วคราว ลองลดจำนวน parallel workers ลงเหลือ 1-2]"
        elif "Incomplete" in err_msg or "timed out" in err_msg.lower():
            hint = " [คำแนะนำ: เครือข่ายขัดข้องระหว่างสตรีมเสียง ลองกดดาวน์โหลดใหม่อีกครั้ง]"

        logger.error(f"[DOWNLOAD ERROR] Failed {url_str}: {err_msg}{hint}")
        return {
            "url": url_str,
            "speaker_id": speaker_id,
            "status": "error",
            "error": f"{err_msg}{hint}",
        }


def download_media_audio(
    urls: List[str],
    output_dir: str = "/content/drive/MyDrive/tts-project/01_raw",
    speaker_id: str = "default",
    cookies_file: Optional[str] = None,
    target_sr: int = 24000,
    max_workers: int = 3,
    max_playlist_items: int = 10,
) -> List[Dict[str, Any]]:
    """Downloads audio streams from YouTube, Facebook, and Instagram concurrently
    using high-speed local NVMe staging and atomic Google Drive syncing.
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is required. Install via: pip install yt-dlp")
        raise

    target_dir = Path(output_dir) / speaker_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Local NVMe staging area (avoids FUSE Cloud I/O bottleneck)
    staging_dir = Path("/tmp") / f"tts_download_staging_{speaker_id}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    clean_urls = [u.strip() for u in urls if u.strip() and not u.strip().startswith("#")]
    if not clean_urls:
        logger.warning("No valid URLs provided to download.")
        return []

    workers = min(max_workers, len(clean_urls))
    logger.info(
        f"Starting parallel download of {len(clean_urls)} URLs (Workers: {workers}) for Speaker: '{speaker_id}' -> {target_dir}"
    )

    base_ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
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
        "max_downloads": max_playlist_items,
    }

    # Bind Node.js JS Runtime if available on the system to silence warning
    node_bin = shutil.which("node") or "/usr/bin/node"
    if os.path.exists(node_bin):
        base_ydl_opts["js_runtimes"] = {"node": {"path": node_bin}}

    if cookies_file and os.path.exists(cookies_file):
        logger.info(f"Using cookies file: {cookies_file}")
        base_ydl_opts["cookiefile"] = cookies_file

    results: List[Dict[str, Any]] = []

    try:
        if workers <= 1:
            for url_str in clean_urls:
                res = _download_single_task(url_str, target_dir, staging_dir, base_ydl_opts, speaker_id)
                results.append(res)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_url = {
                    executor.submit(_download_single_task, url, target_dir, staging_dir, base_ydl_opts, speaker_id): url
                    for url in clean_urls
                }
                for future in as_completed(future_to_url):
                    res = future.result()
                    results.append(res)
    finally:
        # Clean up temporary NVMe staging
        shutil.rmtree(staging_dir, ignore_errors=True)

    success_count = sum(1 for r in results if r.get("status") in ("success", "cached"))
    logger.info(
        f"Download batch finished: {success_count}/{len(results)} items ready in Google Drive."
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
    parser.add_argument("--max-playlist", type=int, default=10, help="Max items per playlist")
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
            max_playlist_items=args.max_playlist,
        )

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from typing import Dict, Any, List, Optional
import soundfile as sf

from src.utils.logger import setup_logger

logger = setup_logger("speaker_inspector")


def audit_speaker_dataset(
    metadata_path: str = "/content/drive/MyDrive/tts-project/02_processed/metadata.jsonl",
    speaker_id: Optional[str] = None,
    min_duration_minutes: float = 30.0,
) -> Dict[str, Any]:
    """Inspects dataset readiness, linguistic statistics, and tonal distribution for a given speaker."""
    logger.info(f"Auditing dataset from: {metadata_path} (Target Speaker: {speaker_id or 'ALL'})")

    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {
            "status": "NOT_FOUND",
            "verdict": "NO_DATA",
            "message": f"Metadata file not found at {metadata_path}",
        }

    records: List[Dict[str, Any]] = []
    available_speakers = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    spk = item.get("speaker", "default")
                    available_speakers.add(spk)
                    if speaker_id is None or spk == speaker_id:
                        records.append(item)
                except Exception:
                    continue

    if not records:
        logger.warning(
            f"No records found for speaker: '{speaker_id}'. Available speakers: {sorted(list(available_speakers))}"
        )
        print(f"\n⚠️ ไม่พบข้อมูลของ Speaker: '{speaker_id}' ใน metadata.jsonl")
        if available_speakers:
            print("💡 รายชื่อ Speaker ที่มีอยู่ในระบบให้เลือกใช้:")
            for s in sorted(list(available_speakers)):
                print(f"   • {s}")
        print()
        return {
            "status": "EMPTY",
            "speaker_id": speaker_id or "all",
            "available_speakers": sorted(list(available_speakers)),
            "total_chunks": 0,
            "total_minutes": 0.0,
            "verdict": "NO_SAMPLES",
        }

    # 1. Audio Duration & Chunks
    total_seconds = 0.0
    valid_audio_count = 0
    missing_audio_count = 0

    for rec in records:
        audio_p = rec.get("audio_path", "")
        dur = rec.get("duration", 0.0)
        if dur > 0:
            total_seconds += dur
            valid_audio_count += 1
        elif os.path.exists(audio_p):
            try:
                info = sf.info(audio_p)
                total_seconds += info.duration
                valid_audio_count += 1
            except Exception:
                missing_audio_count += 1
        else:
            missing_audio_count += 1

    total_minutes = total_seconds / 60.0

    # 2. Text & Linguistic Statistics
    words: List[str] = []
    tones_counter: Counter = Counter()

    for rec in records:
        text = rec.get("normalized_text", rec.get("text", ""))
        tokens = [w.strip() for w in text.split() if w.strip()]
        words.extend(tokens)

        phonemes = rec.get("phonemes", "")
        # Extract explicit tone markers [0-4]
        found_tones = re.findall(r"\[([0-4])\]|([0-4])", phonemes)
        for t_group in found_tones:
            t = t_group[0] or t_group[1]
            if t:
                tones_counter[t] += 1

    total_words = len(words)
    word_freq = Counter(words)
    unique_words = len(word_freq)
    ttr = (unique_words / total_words) if total_words > 0 else 0.0
    top_repeated = word_freq.most_common(5)

    # 3. Tone Coverage [0-4]
    covered_tones = sorted(list(tones_counter.keys()))
    has_full_tones = all(str(t) in tones_counter for t in range(5))

    # 4. Readiness Verdict
    is_ready = total_minutes >= min_duration_minutes and has_full_tones
    verdict = "READY TO TRAIN" if is_ready else "NEEDS_MORE_DATA"

    result = {
        "speaker_id": speaker_id or "all",
        "total_chunks": len(records),
        "valid_audio_files": valid_audio_count,
        "missing_audio_files": missing_audio_count,
        "total_seconds": round(total_seconds, 2),
        "total_minutes": round(total_minutes, 2),
        "min_required_minutes": min_duration_minutes,
        "total_words": total_words,
        "unique_words": unique_words,
        "ttr_lexical_diversity": round(ttr, 4),
        "top_repeated_words": top_repeated,
        "tone_distribution": dict(tones_counter),
        "has_full_tone_coverage": has_full_tones,
        "verdict": verdict,
    }

    # Print formatted Colab UI Dashboard
    print("\n" + "=" * 70)
    print(f"🎙️ SPEAKER AUDIT DASHBOARD — [Speaker: {speaker_id or 'ALL'}]")
    print("=" * 70)
    print("📊 Audio Metrics:")
    print(f"   • Sliced Chunks (3-10s):    {len(records)} segments")
    print(f"   • Valid Audio Files:        {valid_audio_count} (Missing: {missing_audio_count})")
    print(
        f"   • Total Speech Duration:    {total_minutes:.2f} minutes (Target: {min_duration_minutes:.1f}m)"
    )
    status_dur = "✅ Pass" if total_minutes >= min_duration_minutes else f"⚠️ Need +{max(0.0, min_duration_minutes - total_minutes):.1f}m"
    print(f"   • Duration Status:          {status_dur}")

    print("\n📝 Text & Linguistic Statistics:")
    print(f"   • Total Words:              {total_words:,}")
    print(f"   • Unique Vocabulary:        {unique_words:,} (TTR Diversity: {ttr:.3f})")
    print("   • Top 5 Frequent Words:")
    for rank, (w, count) in enumerate(top_repeated, start=1):
        print(f"     {rank}. '{w}' ({count:,} times)")

    print("\n🎵 Thai Tone Coverage [0-4]:")
    tone_names = {
        "0": "สามัญ (Mid)",
        "1": "เอก (Low)",
        "2": "โท (Falling)",
        "3": "ตรี (High)",
        "4": "จัตวา (Rising)",
    }
    for t_idx in range(5):
        key = str(t_idx)
        c = tones_counter.get(key, 0)
        mark = "✅" if c > 0 else "❌"
        print(f"   • Tone {key} - {tone_names[key]}: {c:,} occurrences {mark}")

    print("-" * 70)
    if is_ready:
        print(f"🎯 FINAL VERDICT: [{verdict}] — พร้อมเริ่ม Fine-Tuning ได้ทันที!")
    else:
        print(f"🎯 FINAL VERDICT: [{verdict}] — แนะนำให้อัด/ดึงคลิปเพิ่มเพื่อให้โมเดลเรียนรู้ครบถ้วน")
    print("=" * 70 + "\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Speaker Dataset for TTS Training Readiness")
    parser.add_argument(
        "--metadata",
        type=str,
        default="/content/drive/MyDrive/tts-project/02_processed/metadata.jsonl",
    )
    parser.add_argument("--speaker", type=str, help="Specific Speaker ID to audit")
    parser.add_argument("--min-minutes", type=float, default=30.0, help="Minimum minutes required")
    args = parser.parse_args()

    audit_speaker_dataset(
        metadata_path=args.metadata,
        speaker_id=args.speaker,
        min_duration_minutes=args.min_minutes,
    )

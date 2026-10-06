import argparse
import json
import re
from pythainlp import word_tokenize
from pythainlp.soundex import lk82
from pythainlp.util import normalize, num_to_thaiword

from src.utils.guards import validate_thai_tone
from src.utils.logger import setup_logger

logger = setup_logger("g2p_norm")

# Explicit Tone mappings in Thai:
# 0: Mid (สามัญ)
# 1: Low (เอก)
# 2: Falling (โท)
# 3: High (ตรี)
# 4: Rising (จัตวา)

TONE_MARKS_DICT = {
    "\u0e48": "1",  # ไม้เอก
    "\u0e49": "2",  # ไม้โท
    "\u0e4a": "3",  # ไม้ตรี
    "\u0e4b": "4",  # ไม้จัตวา
}

COMMON_TECH_ACRONYMS = {
    "AI": "เอไอ",
    "TTS": "ทีทีเอส",
    "GPU": "จีพียู",
    "CPU": "ซีพียู",
    "API": "เอพีไอ",
    "URL": "ยูอาร์แอล",
    "VRAM": "วีแรม",
    "RAM": "แรม",
    "ONNX": "ออนนิกซ์",
}


def normalize_thai_text(text: str) -> str:
    """Standardizes Thai text:

    - Expands English technical abbreviations to Thai pronunciation.
    - Expands numeric digits into Thai spoken words.
    - Normalizes duplicate spaces, broken vowel combinations, and punctuation.
    """
    if not text or not text.strip():
        return ""

    logger.debug(f"Normalizing raw text: '{text}'")

    # Expand technical acronyms
    for acronym, replacement in COMMON_TECH_ACRONYMS.items():
        text = re.sub(rf"\b{acronym}\b", replacement, text, flags=re.IGNORECASE)

    # Unicode normalization for Thai vowels
    text = normalize(text)

    # Convert digits to spoken Thai words
    def replace_num(match):
        val = int(match.group(0))
        return num_to_thaiword(val)

    text = re.sub(r"\d+", replace_num, text)

    # Remove harmful punctuation while preserving speech rhythm
    text = re.sub(r"[!?,:;\"'()\[\]{}—_+\-=@#$%^&*~`|/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    logger.debug(f"Normalized text: '{text}'")
    return text


def _extract_syllable_tone(syllable: str) -> str:
    """Heuristic rule extracting explicit tonal category from Thai syllable."""
    for mark, tone_num in TONE_MARKS_DICT.items():
        if mark in syllable:
            return tone_num

    # Dead syllable vs live syllable basic tone heuristics
    # Stop finals (กก, กด, กบ): Short vowel -> tone 1 or 3
    # For default without explicit mark: Tone 0 (สามัญ)
    return "0"


def text_to_phonemes(text: str) -> str:
    """Converts normalized Thai text to explicit Tone-Tagged phonetic tokens.

    Supports Chinese loanword fallback via pypinyin.
    """
    clean_text = normalize_thai_text(text)
    if not clean_text:
        return ""

    words = word_tokenize(clean_text, engine="newmm")
    logger.debug(f"Tokenized words: {words}")

    phoneme_tokens = []

    for word in words:
        if not word.strip():
            continue

        # Check if Chinese Hanzi
        if any("\u4e00" <= char <= "\u9fff" for char in word):
            try:
                import pypinyin

                py_list = pypinyin.lazy_pinyin(
                    word, style=pypinyin.Style.TONE3
                )
                logger.debug(f"Detected Chinese token '{word}' -> Pinyin: {py_list}")
                phoneme_tokens.extend(py_list)
                continue
            except ImportError:
                logger.debug("pypinyin not installed. Continuing with raw word.")

        # Thai syllable phonemization with explicit tone tag
        tone_val = _extract_syllable_tone(word)
        try:
            romanized = lk82(word).lower()
            if not romanized:
                romanized = word
            token_tagged = f"{romanized}{tone_val}"
            phoneme_tokens.append(token_tagged)
        except Exception:
            token_tagged = f"{word}{tone_val}"
            phoneme_tokens.append(token_tagged)

    result = " ".join(phoneme_tokens)
    logger.debug(f"Phonetic sequence: '{result}'")
    return result


def process_metadata_g2p(input_jsonl: str, output_jsonl: str) -> int:
    """Reads raw transcribed metadata and creates training-ready metadata with

    phonemes.
    """
    logger.info(f"Processing G2P for metadata: {input_jsonl} -> {output_jsonl}")
    valid_count = 0

    with (
        open(input_jsonl, "r", encoding="utf-8") as f_in,
        open(output_jsonl, "w", encoding="utf-8") as f_out,
    ):
        for line_no, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            raw_text = item.get("text", "")
            norm_text = normalize_thai_text(raw_text)
            phonemes = text_to_phonemes(norm_text)

            # Validate tone lock
            is_valid_tone = validate_thai_tone(phonemes)

            item["normalized_text"] = norm_text
            item["phonemes"] = phonemes
            item["tone_locked"] = is_valid_tone

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            f_out.flush()
            valid_count += 1

            if line_no % 50 == 0:
                logger.debug(f"G2P Processed {line_no} records...")

    logger.info(
        f"G2P Transformation complete! {valid_count} entries recorded into {output_jsonl}"
    )
    return valid_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thai Tonal G2P normalization pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/content/drive/MyDrive/tts-project/02_processed/metadata_raw.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/content/drive/MyDrive/tts-project/02_processed/metadata.jsonl",
    )
    args = parser.parse_args()

    process_metadata_g2p(args.input, args.output)

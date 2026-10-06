import argparse
import json
import re
from pythainlp import word_tokenize
from pythainlp.soundex import lk82
from pythainlp.util import normalize, num_to_thaiword

from src.utils.guards import validate_thai_tone

# Tone mappings in Thai:
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


def normalize_thai_text(text: str) -> str:
    """Standardizes Thai text:

    - Expands numeric digits into Thai spoken words.
    - Normalizes duplicate spaces, broken vowel combinations, and punctuation.
    """
    text = normalize(text)

    # Convert digits to spoken Thai words
    def replace_num(match):
        val = int(match.group(0))
        return num_to_thaiword(val)

    text = re.sub(r"\d+", replace_num, text)

    # Clean punctuation
    text = re.sub(r"[!?,:;\"'()\[\]{}]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
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
    words = word_tokenize(clean_text, engine="newmm")

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
                phoneme_tokens.extend(py_list)
                continue
            except ImportError:
                pass

        # Thai syllable phonemization with explicit tone tag
        tone_val = _extract_syllable_tone(word)
        try:
            # Generate Romanized / Soundex base
            romanized = lk82(word).lower()
            if not romanized:
                romanized = word
            phoneme_tokens.append(f"{romanized}{tone_val}")
        except Exception:
            phoneme_tokens.append(f"{word}{tone_val}")

    result = " ".join(phoneme_tokens)
    return result


def process_metadata_g2p(input_jsonl: str, output_jsonl: str) -> int:
    """Reads raw transcribed metadata and creates training-ready metadata with

    phonemes.
    """
    valid_count = 0
    with (
        open(input_jsonl, "r", encoding="utf-8") as f_in,
        open(output_jsonl, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
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
            valid_count += 1

    print(
        f"[G2P] Processed {valid_count} entries -> saved to {output_jsonl}"
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

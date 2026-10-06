# 03. ASR & Thai Tonal G2P Pipeline

## 1. Automatic Speech Recognition (ASR) via Faster-Whisper (`src/asr/transcribe.py`)
เนื่องจากผู้ใช้งานบันทึกเสียงสดโดยไม่มีสคริปต์ (Unscripted Mobile Recordings) ระบบจึงต้องถอดเสียงพูดภาษาไทยโดยอัตโนมัติ

* **โมเดลที่เลือก**: `faster-whisper` ขนาด **Large-v3** (รันด้วย CTranslate2 FP16 บน GPU T4)
* **Confidence Filtering (Hard Guard)**:
  โมเดล Whisper อาจเกิดอาการหลอน (Hallucination) ในช่วงที่มีเสียงหายใจยาวหรือเสียงพึมพำ ระบบจึงมีตัวกรอง Log Probability:
  ```python
  if avg_logprob < -0.5:
      # ทิ้งคลิปทันที ไม่นำไปใช้เทรน
  ```
  การคัดทิ้งนี้ช่วยป้องกันไม่ให้โมเดล TTS เรียนรู้คู่ข้อความ-เสียงที่ผิดพลาด

---

## 2. Text Normalization ภาษาไทย (`src/g2p/text_norm.py`)
ข้อความที่ถอดออกมามักมีตัวเลขและสัญลักษณ์ เช่น "2026", "100 บาท", "50%" ซึ่งโมเดล TTS อ่านตรงๆ ไม่ได้

ระบบใช้ `pythainlp.util.num_to_thaiword` และ Regex แปลงเป็นคำอ่านเต็ม:
* `2026` $\rightarrow$ `สองพันยี่สิบหก`
* `50%` $\rightarrow$ `ห้าสิบเปอร์เซ็นต์`
* ตัดเครื่องหมายวรรคตอนที่ไม่จำเป็นออก พร้อม normalize สระและวรรณยุกต์ซ้อน

---

## 3. Explicit Thai Tonal G2P (หัวใจสำคัญของความเป็นมนุษย์)
สาเหตุที่โมเดล TTS จากต่างประเทศพูดภาษาไทยแล้ว "สำเนียงเพี้ยน" หรือ "เสียงแบน" เกิดจากการที่โมเดลไม่รู้ระดับวรรณยุกต์ (Tone)

ภาษาไทยมี 5 ระดับเสียงวรรณยุกต์:
| รหัส (Tone Tag) | วรรณยุกต์ไทย | ตัวอย่าง | Pitch Contour |
|---|---|---|---|
| **0** | เสียงสามัญ | กา (ka0) | กลางราบ (Mid Level) |
| **1** | เสียงเอก | ก่า (ka1) | ต่ำ (Low Falling) |
| **2** | เสียงโท | ก้า (ka2) | สูงตก (High Falling) |
| **3** | เสียงตรี | ก๊า (ka3) | สูงชัน (High Rising) |
| **4** | เสียงจัตวา | ก๋า (ka4) | ต่ำแล้วขึ้น (Dipping Rising) |

### การทำงานของ Tokenizer
ระบบทำการตัดคำด้วย `pythainlp.word_tokenize(engine='newmm')` และแปลงเป็นหน่วยเสียงที่กำกับด้วยรหัสวรรณยุกต์เสมอ เช่น:
```
ข้อความ: สวัสดีครับ
↓
Tokenize: [สวัส, ดี, ครับ]
↓
Phonemes: [sawat1, dii0, krap3]
```

### Fallback รองรับคำยืมภาษาจีน (`pypinyin`)
หากในข้อความมีอักษรจีนปนอยู่ เช่น ชื่อเฉพาะหรือคำทับศัพท์ ระบบจะใช้ `pypinyin` แปลงเป็นพินอินพร้อมวรรณยุกต์ (เช่น `hao3`) อัตโนมัติ

---

## 4. Metadata JSONL Output
ผลลัพธ์ของขั้นตอนนี้จะถูกบันทึกลง `/content/drive/MyDrive/tts-project/02_processed/metadata.jsonl`:
```json
{
  "audio_path": "/content/drive/MyDrive/tts-project/02_processed/wavs/rec_seg_0001.wav",
  "text": "สวัสดีครับยินดีต้อนรับ",
  "normalized_text": "สวัสดีครับ ยินดีต้อนรับ",
  "phonemes": "sawat1 dii0 krap3 yin0 dii0 ton3 rap3",
  "tone_locked": true
}
```

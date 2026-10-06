# 09. คู่มือการทดสอบและสังเคราะห์เสียงจาก ONNX บน Google Colab (`docs/09_onnx_inference_testing_guide.md`)

คู่มือนี้อธิบายวิธีการนำไฟล์โมเดล `.onnx` (ทั้งรุ่น FP16 และ INT8) ที่เทรนและ Export ไว้ใน Google Drive มาทดสอบสังเคราะห์เสียงพูดภาษาไทยจริงบน Google Colab พร้อมระบบจำลองอารมณ์ (In-Context Emotion Conditioning) และระบบปรับแต่งเสียงระดับสตูดิโอ (Spotify Pedalboard DSP)

---

## 1. จุดเด่นของการทดสอบด้วย ONNX
1. **ไม่ต้องใช้ GPU**: รันบน Google Colab CPU ธรรมดาได้ ไม่ต้องเสียโควตา T4 GPU
2. **โหลดไว**: ใช้เวลาโหลดโมเดลเข้าหน่วยความจำไม่ถึง 2 วินาที
3. **อ่าน-เขียนตรงกับ Google Drive**: ชี้ไฟล์โมเดลตรงจาก `/MyDrive/tts-project/04_onnx_exports/` และบันทึกไฟล์เสียงผลลัพธ์กลับเข้า Drive ทันที
4. **ทดสอบฟังเสียงได้ทันที**: ใช้ `IPython.display.Audio` ฟังผลลัพธ์ผ่าน Web Browser โดยไม่ต้องดาวน์โหลดไฟล์ลงเครื่อง

---

## 2. โครงสร้างไฟล์ใน Google Drive ที่เกี่ยวข้อง

```
/MyDrive/tts-project/
├── 02_processed/
│   └── wavs/                     # คลิปเสียง 3-10s ที่ตัดไว้ (นำมาใช้เป็น Emotion Reference ได้)
└── 04_onnx_exports/
    ├── model.onnx                # โมเดลเต็ม (FP16 / FP32)
    ├── model_quant_int8.onnx     # โมเดลขนาดกะทัดรัด (< 250MB) รันบน CPU ไวที่สุด
    └── output_*.wav              # ไฟล์เสียงผลลัพธ์ที่สร้างออกมา
```

---

## 3. วิธีการรันบน Google Colab (แบบทีละขั้นตอน)

### ขั้นตอนที่ 1: Mount Google Drive และติดตั้ง Dependencies
```python
from google.colab import drive
drive.mount('/content/drive')

# ติดตั้ง uv และแพ็กเกจที่จำเป็นสำหรับ Inference (ใช้เวลา ~30 วินาที)
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"

# Clone หรือ Pull โค้ดล่าสุด
!git clone https://github.com/SatangThevalue/satangthevalue-tts-custom.git /content/satangthevalue-tts-custom
%cd /content/satangthevalue-tts-custom
!uv pip install --system onnxruntime pythainlp soundfile pedalboard
```

---

### ขั้นตอนที่ 2: โหลดโมเดล ONNX จาก Google Drive

```python
from src.inference.engine import TTSEngine

# เลือกรุ่น INT8 เพื่อความเร็วสูงสุด หรือเลือกรุ่น model.onnx ปกติ
onnx_model_path = "/content/drive/MyDrive/tts-project/04_onnx_exports/model_quant_int8.onnx"

engine = TTSEngine(onnx_model_path)
print("โหลดโมเดล ONNX จาก Google Drive สำเร็จ!")
```

---

### ขั้นตอนที่ 3: สังเคราะห์เสียงและฟังผลลัพธ์ใน Colab

```python
from IPython.display import Audio, display

# 1. กำหนดข้อความที่ต้องการให้ AI พูด
text_to_speak = "สวัสดีครับ วันนี้เรากำลังทดสอบระบบเสียงสังเคราะห์ภาษาไทย บนระบบออนนิกซ์รันไทม์ คุณภาพเสียงนุ่มนวลและเป็นธรรมชาติครับ"

# 2. สังเคราะห์เสียง (บันทึกตรงเข้า Google Drive)
output_file = "/content/drive/MyDrive/tts-project/04_onnx_exports/test_output_1.wav"

result = engine.synthesize(
    text=text_to_speak,
    ref_audio_path=None,           # ใส่ None สำหรับเสียงโทนปกติ (Neutral)
    output_wav_path=output_file,
    enable_mastering=True          # เปิดใช้ Spotify Pedalboard DSP เพื่อความนุ่มและชัด
)

print(f"หน่วยเสียง (Phonemes): {result['phonemes']}")
print(f"ความยาวเสียง: {result['duration_sec']:.2f} วินาที")
print(f"เวลาที่ใช้ประมวลผล: {result['elapsed_sec']:.2f} วินาที")
print(f"Real-Time Factor (RTF): {result['rtf']:.3f} (ยิ่งต่ำยิ่งเร็ว)")

# 3. แสดง Audio Player ให้กดฟังบน Colab ทันที
display(Audio(result['output_path']))
```

---

## 4. การคุมอารมณ์และจังหวะพูดด้วย Reference Audio (In-Context Conditioning)

หากต้องการให้เสียงที่สังเคราะห์ออกมามีอารมณ์ (ตื่นเต้น, สุภาพ, เล่าเรื่อง) ให้ชี้ `ref_audio_path` ไปยังไฟล์คลิปเสียงตัวอย่าง 3–5 วินาทีของอารมณ์นั้น:

```python
# ชี้ไปยังคลิปเสียงตัวอย่างใน Drive ที่พูดด้วยอารมณ์ตื่นเต้นหรือจริงจัง
ref_clip = "/content/drive/MyDrive/tts-project/02_processed/wavs/your_sample_seg_0005.wav"

result = engine.synthesize(
    text="โปรโมชั่นพิเศษสุดคุ้ม มีเฉพาะวันนี้วันเดียวเท่านั้น ห้ามพลาดเด็ดขาดเลยนะครับ",
    ref_audio_path=ref_clip,       # ระบบจะเลียนแบบไดนามิกและจังหวะของคลิปนี้
    output_wav_path="/content/drive/MyDrive/tts-project/04_onnx_exports/test_excited.wav",
    enable_mastering=True
)

display(Audio(result['output_path']))
```

---

## 5. การรันผ่าน Command Line (CLI)

สามารถเรียกสคริปต์ `src/inference/test_drive_onnx.py` ผ่าน Terminal หรือ Cell ใน Colab ได้โดยตรง:

```bash
python -m src.inference.test_drive_onnx \
    --model "/content/drive/MyDrive/tts-project/04_onnx_exports/model_quant_int8.onnx" \
    --text "ทดสอบการสังเคราะห์เสียงผ่านคำสั่งเทอร์มินัล" \
    --output "/content/drive/MyDrive/tts-project/04_onnx_exports/cli_output.wav"
```

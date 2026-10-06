# 07. DSP Studio Mastering & Runtime Inference (`src/inference/` & `src/audio/mastering.py`)

## 1. Runtime Inference Engine (`src/inference/engine.py`)
โมดูลสำหรับเรียกใช้งานโมเดลจริง (Production Inference):
1. **Text Input**: รับข้อความภาษาไทยเข้ามา
2. **G2P**: แปลงตัวเลข สัญลักษณ์ และส่งเข้า Tone Locking Engine ได้ Phonemes
3. **In-Context Emotion Conditioning**:
   * อนุญาตให้ใส่ไฟล์เสียงอ้างอิงความยาว 3–5 วินาที (`ref_audio_path`) ของอารมณ์ที่ต้องการ (เช่น ตื่นเต้น, สุภาพ, ผ่อนคลาย)
   * ตัวโมเดลจะทำการคัดลอก Prosody และ Dynamics ของคลิปอ้างอิงนั้นมาใช้กับประโยคใหม่
4. **ONNX Runtime Engine**: รันผ่าน CPU หรือ CUDA Execution Provider คำนวณคลื่นเสียงดิบออกมา

---

## 2. In-Memory Studio Mastering Chain (Spotify Pedalboard)
ความลับที่ทำให้เสียงของ ElevenLabs มีมิติ "แพงและน่าฟัง" ไม่ใช่แค่ตัวโมเดล AI แต่เป็น **Post-Processing Chain**

แทนที่จะเรียกคำสั่ง FFmpeg ภายนอกซึ่งช้าและกิน Disk I/O ระบบใช้ **Spotify Pedalboard** ทำงานบน RAM (NumPy Arrays) ด้วย C++ เร็วกว่าเดิม 5 เท่า:

### ลำดับเอฟเฟกต์ (DSP Chain):
```
Raw Audio Array (24kHz)
          │
          ▼
1. Highpass Filter (80Hz) ──────────► ตัด Sub-bass, ลมกระแทกไมค์ (Plosives), เสียงเครื่องปรับอากาศ
          │
          ▼
2. Speech Compressor ───────────────► Ratio 2.5:1, Threshold -16dB
                                       ดึงคำที่พูดเบาให้ชัด และกดคำที่ดังเกินไปให้สม่ำเสมอ
          │
          ▼
3. True Peak Limiter (-1.0 dBFS) ───► ป้องกันการเกิด Digital Clipping (เสียงแตกพร่า)
          │
          ▼
4. Output Normalization ────────────► ปรับระดับความดังให้ได้มาตรฐาน Studio/Podcast (-16 LUFS)
```

---

## 3. Real-Time Factor (RTF) Benchmark
$$\text{RTF} = \frac{\text{Processing Time (วินาที)}}{\text{Audio Duration (วินาที)}}$$
* เกณฑ์ผ่าน: $\text{RTF} < 0.6$ บน CPU ปกติ (สามารถสร้างเสียง 10 วินาที ได้ภายใน 6 วินาที)
* เป้าหมายเมื่อใช้ร่วมกับ INT8 ONNX: $\text{RTF} \approx 0.25 - 0.35$

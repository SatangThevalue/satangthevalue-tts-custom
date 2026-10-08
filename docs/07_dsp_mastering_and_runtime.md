# 07. DSP Studio Mastering & Acoustic Realism Pipeline (`src/audio/mastering.py` & `src/inference/`)

## 1. Runtime Inference Engine (`src/inference/engine.py`)
โมดูลสำหรับเรียกใช้งานโมเดลจริง (Production Inference):
1. **Text Input**: รับข้อความภาษาไทยเข้ามา ทำ Normalization ขยายตัวเลขและคำย่อ
2. **Thai Tonal G2P**: แปลงหน่วยเสียงพร้อม Tone Locking `[0-4]` ควบคุมระดับ Pitch ไม่ให้ลอย
3. **In-Context Emotion Conditioning**:
   * อนุญาตให้ใส่ไฟล์เสียงอ้างอิงความยาว 3–5 วินาที (`ref_audio_path`) ของอารมณ์ที่ต้องการ (เช่น ตื่นเต้น, สุภาพ, ผ่อนคลาย)
   * ตัวโมเดลจะทำการคัดลอก Prosody และ Dynamics ของคลิปอ้างอิงนั้นมาใช้กับประโยคใหม่
4. **ONNX Runtime Engine**: รันผ่าน CPU หรือ CUDA Execution Provider คำนวณคลื่นเสียงดิบออกมา
5. **Acoustic Realism DSP Chain**: ส่งผ่านโมดูล Mastering 6 ขั้นตอนในหน่วยความจำ (RAM)

---

## 2. In-Memory Studio Mastering Chain (Spotify Pedalboard + NumPy DSP)

ความลับที่ทำให้เสียงของ ElevenLabs ฟังดูมีมิติ "แพงและเป็นธรรมชาติเหมือนมนุษย์จริง" ไม่ได้ขึ้นอยู่กับโมเดลอย่างเดียว แต่ขึ้นอยู่กับ **Intentional Imperfections & Vocal Mastering**:

แทนที่จะเรียกคำสั่ง FFmpeg ภายนอกซึ่งช้าและกิน Disk I/O ระบบใช้ **Spotify Pedalboard (C++ DSP Engine)** ร่วมกับ NumPy ทำงานบน RAM 100% ประมวลผลเร็วเพียง 20–70 มิลลิวินาที:

### แผนผังลำดับเอฟเฟกต์ (Acoustic Realism DSP Chain):
```
Raw Audio Array (24kHz)
          │
          ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Pre-EQ & De-Essing                                                  │
│   • Highpass Filter @ 80Hz: ตัด Sub-bass, ลมกระแทกไมค์ (Plosives)      │
│   • LowShelf Filter @ 220Hz (+1.5dB): เพิ่มความหนาและน้ำหนักเสียงพูด   │
│   • Peak De-esser @ 6.8kHz (-2.5dB, Q=1.2): ลดเสียงเสียดฟัน (ซ/ส)      │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Dynamic Speech Compressor                                           │
│   • Threshold -16.0 dB, Ratio 2.5:1, Attack 10ms, Release 100ms        │
│   • เกลี่ยระดับเสียงให้สม่ำเสมอ ดึงคำที่พูดเบา และคุมคำที่กระแทกดัง     │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Studio Space Simulation (Micro-Reverb Early Reflections)            │
│   • Room size: 0.08 (จำลองห้องอัด Vocal Booth ขนาดเล็ก)                │
│   • Wet Level: 0.04 (4%), Dry Level: 0.96 (96%), Damping: 0.6         │
│   • แก้ปัญหาเสียง "Dry ลอยติดหู" ให้มีมิติความลึก (Depth) เหมือนยืนพูด │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Analog Tube Warmth (Soft Saturation)                                │
│   • ผ่านสมการ Non-linear: y = tanh(drive * x) / tanh(drive)            │
│   • Drive = 1.12: เติมฮาร์โมนิกคู่/คี่อ่อนๆ ย่าน 1k-3kHz ให้เนื้อเสียงนุ่ม│
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. Comfort Noise Injection (Room Tone Ambient Masking)                 │
│   • สังเคราะห์ Pink Noise (1/f spectral curve) ผ่าน FFT ใน NumPy       │
│   • ระดับสัญญาณ -54.0 dBFS                                             │
│   • ลบปัญหา "Dead Digital Silence" (ช่วงเงียบ 0 dBFS ที่ทำให้ดูเป็น AI)│
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 6. True Peak Limiting & Normalization                                  │
│   • Limiter Threshold -1.0 dBFS                                        │
│   • Peak Scaling ล็อกเพดานสูงสุดที่ -1.0 dBFS ป้องกัน Digital Clipping │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         [Final Studio Wav]
```

---

## 3. รายละเอียดเชิงลึกของ 4 เทคนิคความสมจริง

### 3.1 Comfort Noise Injection (-54 dBFS Pink Noise)
* **ปัญหา**: AI TTS ทั่วไป ในช่องว่างระหว่างคำจะตัดค่าเป็น `0.000` สนิท (Dead Silence) ทำให้ผู้ฟังรู้สึกว่าเสียงขาดเป็นห้วงๆ หรือมีคนตัดแปะคลิป
* **การแก้**: สังเคราะห์ Pink Noise ด้วยสมการ $S(f) = \frac{1}{\sqrt{f}}$ ผ่าน Fast Fourier Transform (FFT) แล้วเกลี่ยเบาๆ ที่ความดัง $-54\text{ dBFS}$
* **ผลลัพธ์**: ช่องว่างระหว่างคำมีเสียงบรรยากาศห้องบางๆ เหมือนอัดในห้องจริง ผู้ฟังจะรู้สึกว่าการพูดลื่นไหลเป็นประโยคเดียวกัน

### 3.2 Analog Tube Warmth
* **ปัญหา**: สัญญาณเสียงดิจิทัลมีความแข็งกระด้าง (Sterile/Harsh)
* **การแก้**: ใช้ฟังก์ชัน Sigmoidal Hyperbolic Tangent:
  $$y = \frac{\tanh(\alpha \cdot x)}{\tanh(\alpha)} \quad (\alpha = 1.12)$$
* **ผลลัพธ์**: จำลองการทำงานของหลอดสุญญากาศ (Tube Preamp) เติม Harmonic Richness ให้เสียงนุ่ม อบอุ่น และฟังดูหรูหรา

### 3.3 Studio Micro-Reverb (Early Reflections)
* **ปัญหา**: เสียงดิบที่ออกจากโมเดล AI จะแห้งสนิท (Dry 100%) เหมือนเสียงดังอยู่ในหัวผู้ฟัง
* **การแก้**: ใส่ Reverb ขนาดห้องเล็กมาก (`room_size=0.08`, `wet=0.04`) เพื่อสร้าง Early Reflection ของผนังห้องอัด
* **ผลลัพธ์**: เสียงมีทิศทางและมิติพื้นที่เสมือนคนจริงนั่งพูดอยู่ตรงหน้า

### 3.4 Sibilance Taming (De-esser)
* **ปัญหา**: ภาษาไทยมีพยัญชนะเสียดแทรกเยอะ เช่น ส, ซ, ศ, ษ ซึ่งมักเกิดเสียงแหลมทิ่มหู (Piercing Highs) ในย่าน 6.5k–7.5kHz
* **การแก้**: ใช้ `PeakFilter` ตัดลดความถี่ $6,800\text{ Hz}$ ลง $-2.5\text{ dB}$ (Q=1.2) ช่วยให้เสียงนุ่มนวล ไม่บาดหูเมื่อเปิดฟังผ่านหูฟัง

---

## 4. Real-Time Factor (RTF) Benchmark
$$\text{RTF} = \frac{\text{Processing Time (วินาที)}}{\text{Audio Duration (วินาที)}}$$
* **เวลาประมวลผล DSP ทั้งหมด**: ~20 ถึง 75 มิลลิวินาที สำหรับเสียงยาว 5 วินาที
* **RTF ของกระบวนการ DSP**: $< 0.015$ (เร็วกว่าเวลาเล่นเสียงจริงกว่า 60 เท่า)
* **RTF รวมทั้งระบบ (ONNX INT8 + DSP)**: $\approx 0.25 - 0.35$ บน CPU มาตรฐาน

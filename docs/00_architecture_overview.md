# 00. Architecture Overview

ระบบ **satangthevalue-tts-custom** เป็นสถาปัตยกรรม Text-to-Speech (TTS) แบบแยกส่วน (Modular End-to-End Pipeline) สำหรับภาษาไทย มุ่งเน้นการสร้างเสียงที่มีความเป็นมนุษย์สูง (Human-like Prosody & Timbre) เทียบเท่า ElevenLabs (ระดับ 85–90% Parity) โดยมีเงื่อนไขและข้อจำกัดการทำงานคือ **ทำงานบน Google Colab Free Tier (T4 GPU 15GB VRAM), จัดเก็บข้อมูลบน Google Drive, จัดเก็บโค้ดบน GitHub และ Export เป็น ONNX เพื่อรันบน CPU สเปคต่ำได้**

---

## 1. High-Level Architecture Diagram

```
                 [Mobile Phone Audio (.m4a / .wav)]
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Restoration & Ingestion                                       │
│  - Resemble Enhance (Denoise + De-reverb + Bandwidth Ext 12k-24kHz)   │
│  - SNR Guard (> 25 dB)                                                 │
│  - Silero VAD (3.0s - 10.0s Slicing with Breath Pad 150ms/200ms)        │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: ASR & Thai Tonal G2P                                          │
│  - Faster-Whisper Large-v3 (Auto ASR, Confidence logprob > -0.5)       │
│  - PyThaiNLP: Number to Spoken Words + Text Normalization              │
│  - Tone Lock: Explicit Tonal Tags [0-4] + pypinyin loanword fallback   │
│  - Data Packaging: tarfile cache (แก้ปัญหา Google Drive FUSE I/O)      │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Memory-Safe LoRA Fine-Tuning (Colab T4 Free)                  │
│  - F5-TTS Backbone (Flow Matching Vector Field)                        │
│  - BitsAndBytes 8-bit AdamW (ลด VRAM 60%)                             │
│  - Hugging Face Accelerate (FP16 Mixed Precision)                      │
│  - Hard VRAM Guard (< 13.0 GB ceiling)                                 │
│  - Safe Checkpoint Sync: ส่งเข้า Google Drive ทุก 200 Steps            │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Model Optimization & Export                                   │
│  - Merge LoRA Weights into Base Graph                                  │
│  - PyTorch to ONNX Export (Opset 17, Dynamic Axes)                     │
│  - OnnxSlim Graph Pruning (ตัดโหนดซ้ำซ้อน)                             │
│  - Dynamic INT8 Quantization (ขนาดโมเดลรวม < 300 MB)                  │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Runtime Engine & Studio DSP Mastering                         │
│  - ONNX Runtime Inference (CPU Execution Provider)                     │
│  - In-Context Emotion Prompt Guide (3-5s Reference Audio)              │
│  - Spotify Pedalboard DSP in RAM:                                      │
│      • Highpass Filter 80Hz                                            │
│      • Transparent Speech Compressor                                   │
│      • True Peak Limiter (-1.0 dBFS)                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ทำไมสถาปัตยกรรมนี้ถึงแตะระดับ 85–90% ของ ElevenLabs?

1. **ไม่ปล่อยให้เสียงมือถือทำลายโมเดล**: เสียงอัดจากโทรศัพท์มี Reverb และ Noise สูง หากนำไปเทรนตรงๆ โมเดลจะจดจำเสียงสะท้อนเป็นเอกลักษณ์เสียง (Acoustic Smearing) การใช้ Resemble Enhance ช่วยลบ Reverb และขยายคลื่นความถี่สูง (Bandwidth Extension) ทำให้เสียงเปิดและชัดเหมือนไมค์สตูดิโอ
2. **การรักษาเสียงหายใจ (Breath Preservation)**: VAD ทั่วไปตัดเสียงเงียบทิ้งทั้งหมด ทำให้เสียงที่เจนออกมาพูดติดกันเป็นท่อนแข็ง การใส่ Padding หัวท้าย 150–200ms ทำให้โมเดลเรียนรู้จังหวะสูดลมหายใจจริง
3. **Explicit Tone Locking (G2P ภาษาไทย)**: ปัญหาหลักของ TTS ภาษาไทยคือเสียงสูงต่ำวรรณยุกต์ลอย การล็อกรหัสวรรณยุกต์ 0–4 ท้ายพยางค์ทำให้ Pitch Contour มีความเสถียรและแม่นยำ
4. **DSP Mastering ใน RAM**: เสียงดิบจากโมเดล AI มักมีเสียงลมกระแทก (Plosives) และเสียงเสียดฟัน (Sibilance) การส่งผ่าน Pedalboard DSP ช่วยยกระดับความนุ่มและความดัง (Loudness) สู่ระดับ Podcast/Studio ทันที

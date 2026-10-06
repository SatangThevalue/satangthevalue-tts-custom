# 08. Project KPIs, Metrics & Guardrails

## 1. ตารางตัวชี้วัดความสำเร็จของโปรเจกต์ (KPIs Table)

| หมวดหมู่ | ตัวชี้วัด (Metric) | เครื่องมือวัด | เกณฑ์ผ่านขั้นต่ำ (Pass Gate) | เป้าหมาย ElevenLabs Parity |
|---|---|---|---|---|
| **Data Quality** | Signal-to-Noise Ratio (SNR) | `src.utils.guards.calculate_snr` | > 25.0 dB | > 35.0 dB |
| | Audio Chunk Duration | `soundfile.info` | 3.0 – 10.0 วินาที | 4.0 – 8.0 วินาที |
| | Word/Tone Preservation | `validate_thai_tone` | Tone-tagged > 70% | Tone-tagged > 95% |
| | ASR Confidence Filter | Whisper `avg_logprob` | > -0.5 | > -0.2 |
| **Colab Stability** | VRAM Memory Usage | `check_vram_limit` | < 13.0 GB | < 11.5 GB |
| | Checkpoint Integrity | ตรวจสอบไฟล์บน Google Drive | มีไฟล์ทุก 200 steps | กู้คืน Resume ได้ 100% |
| **Acoustic Quality**| Speaker Similarity | CAM++ Cosine Similarity | > 0.80 | > 0.88 |
| | Tone Error Rate (TTER) | Human / Script Evaluation | < 4% | < 1% |
| | Naturalness (MOS) | UTMOS Score | > 3.8 / 5.0 | > 4.3 / 5.0 |
| **Deployment** | Model File Size | File size (`.onnx`) | < 400 MB | < 250 MB (INT8) |
| | Real-Time Factor (RTF) | `src.inference.engine` บน CPU | < 0.6 | < 0.3 |
| | Runtime RAM Footprint | `psutil` Memory RSS | < 3.0 GB | < 1.5 GB |

---

## 2. Hard Guardrails ฝังในโค้ด Python (`src/utils/guards.py`)

1. **`check_vram_limit(limit_gb=13.0)`**:
   * ตรวจสอบ VRAM ทุก Iteration หากเกินสั่ง `torch.cuda.empty_cache()` และ Raise Exception ทันที ป้องกัน Colab GPU Crash
2. **`validate_audio_chunk(wav_path, min_duration=3.0, max_duration=10.0)`**:
   * ลบคลิปเสียงทิ้งอัตโนมัติหากไม่อยู่ในช่วง 3–10 วินาที
3. **`validate_thai_tone(phonemes)`**:
   * ตรวจสอบว่าพยางค์ต้องลงท้ายด้วย `[0-4]` ห้ามมี Phoneme ไร้วรรณยุกต์ปนเข้าชุดเทรน
4. **`check_drive_mounted(target_dir)`**:
   * ตรวจสอบว่า Google Drive Mount ติดต่อได้จริงก่อนเริ่มบันทึก Checkpoint ป้องกันสถานการณ์ Train ไปหลายชั่วโมงแต่บันทึกไม่สำเร็จ

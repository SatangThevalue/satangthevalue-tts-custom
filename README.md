# satangthevalue-tts-custom

> **Production-grade Thai Custom TTS Pipeline targeting ElevenLabs Parity (85–90%)**  
> ออกแบบมาเพื่อเทรนบน **Google Colab Free Tier (GPU T4 15GB)**, เก็บข้อมูลบน **Google Drive**, บันทึกโค้ดบน **GitHub** และ Export เป็น **ONNX INT8** เพื่อนำไปรันบน CPU สเปคต่ำได้

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SatangThevalue/satangthevalue-tts-custom/blob/main/notebooks/colab_pipeline.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## สารบัญเอกสารเชิงลึก (Detailed Documentation)

| บทที่ | เอกสาร | คำอธิบาย |
|---|---|---|
| **00** | [Architecture Overview](docs/00_architecture_overview.md) | ภาพรวมสถาปัตยกรรมทั้งระบบ และการเปรียบเทียบกับ ElevenLabs |
| **01** | [Audio Ingestion & Restoration](docs/01_audio_ingestion_and_restoration.md) | การกู้เสียงมือถือด้วย Resemble Enhance (Denoise, De-reverb, Bandwidth Ext) |
| **02** | [Slicing & Breath Preservation](docs/02_slicing_and_breath_preservation.md) | Silero VAD สับเสียงช่วง 3–10s พร้อม Breath Padding 150ms/200ms |
| **03** | [ASR & Thai Tonal G2P](docs/03_asr_and_thai_tonal_g2p.md) | Faster-Whisper Large-v3, Text Normalization และ Tone Locking [0-4] |
| **04** | [Dataset Packaging & Drive I/O](docs/04_dataset_packaging_and_drive_io.md) | แก้ปัญหาคอขวด FUSE บน Google Drive ด้วย Tarfile Caching |
| **05** | [LoRA Fine-Tuning & VRAM Guards](docs/05_finetuning_and_vram_guards.md) | เทคนิคคุม VRAM < 13GB บน Colab T4 ด้วย Accelerate + BitsAndBytes 8-bit |
| **06** | [ONNX Export & INT8 Quantization](docs/06_onnx_export_and_quantization.md) | การแปลงโมเดล, OnnxSlim Pruning และ Dynamic INT8 Quantization |
| **07** | [DSP Mastering & Runtime](docs/07_dsp_mastering_and_runtime.md) | In-Memory Studio Mastering Chain ด้วย Spotify Pedalboard |
| **08** | [Metrics, KPIs & Guards](docs/08_metrics_kpis_and_guards.md) | ตารางตัวชี้วัดความสำเร็จและตัวดัก Error ในโค้ด Python |

---

## แผนภาพการทำงานของระบบ (Pipeline Flow)

```
[Mobile Phone Audio (.m4a/.wav)]
              │
              ▼
    [1. Resemble Enhance]     ──► ตัด Reverb/Noise + ขยายย่าน 12k-24kHz
              │
              ▼
    [2. Silero VAD Slice]     ──► หั่น 3.0s - 10.0s + เก็บเสียงหายใจ (Pad 150/200ms)
              │
              ▼
    [3. Faster-Whisper]       ──► ถอดเสียงไทยอัตโนมัติ (คัดกรอง logprob > -0.5)
              │
              ▼
    [4. Thai Tonal G2P]       ──► ล็อกวรรณยุกต์ 5 ระดับ [0-4] (PyThaiNLP + pypinyin)
              │
              ▼
    [5. LoRA Fine-Tuning]     ──► รันบน Colab T4 (FP16, 8-bit AdamW, VRAM < 13GB)
              │
              ▼
    [6. ONNX INT8 Export]     ──► แปลงเป็น ONNX (< 250MB) ตัดกราฟด้วย onnxslim
              │
              ▼
    [7. Pedalboard DSP]       ──► Highpass 80Hz + Compressor + Limiter ใน RAM
              │
              ▼
      [Final Studio Wav]
```

---

## โครงสร้างโปรเจกต์ (Repository Structure)

```
satangthevalue-tts-custom/
├── configs/
│   ├── base_config.yaml         # Audio sampling, paths, audio parameters
│   └── lora_config.yaml         # LoRA rank, alpha, learning rate, target modules
├── docs/                        # เอกสารอธิบายรายละเอียดเชิงลึกครบทั้ง 8 โมดูล
├── notebooks/
│   └── colab_pipeline.ipynb     # Interactive Jupyter Notebook สำหรับรันบน Colab
├── src/
│   ├── audio/
│   │   ├── enhance.py           # Resemble Enhance (Denoise + De-reverb)
│   │   ├── slicer.py            # Silero VAD Chunking + Breath Preservation
│   │   └── mastering.py         # Spotify Pedalboard Studio DSP Chain
│   ├── asr/
│   │   └── transcribe.py        # Faster-Whisper Large-v3 Pipeline
│   ├── g2p/
│   │   └── text_norm.py         # PyThaiNLP + Thai Tonal Locking [0-4]
│   ├── training/
│   │   ├── dataset.py           # PyTorch Dataset + Collator
│   │   └── finetune_lora.py     # Accelerate + BitsAndBytes 8-bit Training
│   ├── export/
│   │   ├── export_onnx.py       # PyTorch to ONNX Graph Exporter
│   │   └── optimize.py          # OnnxSlim Pruning + Dynamic INT8 Quantizer
│   ├── inference/
│   │   └── engine.py            # ONNX Runtime Engine + In-Context Emotion Guide
│   └── utils/
│       ├── guards.py            # Hard Error Guards (VRAM, Duration, SNR, Tone)
│       └── pack.py              # Tarfile Dataset Pack/Unpack (Drive I/O Fix)
├── pyproject.toml               # Poetry/UV Project Configuration
├── .gitignore                   # Ignore audio, weights, and caches
└── README.md
```

---

## การเริ่มต้นใช้งานบน Google Colab (Quickstart)

### 1. โครงสร้างโฟลเดอร์บน Google Drive
เตรียมโฟลเดอร์บน Google Drive ตามโครงสร้างนี้:
```
Google Drive: /MyDrive/tts-project/
├── 01_raw/                 # โยนไฟล์เสียงมือถือดิบ (.m4a, .wav, .mp3)
├── 02_processed/           # ไฟล์เสียงที่ตัดแล้วและ metadata.jsonl
├── 03_checkpoints/         # Weights ที่บันทึกอัตโนมัติทุก 200 steps
└── 04_onnx_exports/        # โมเดล .onnx พร้อมนำไปใช้งาน
```

### 2. รันบน Google Colab
เปิดไฟล์ [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb) บน Google Colab และรันทีละขั้นตอน:

```bash
# ติดตั้ง uv และ dependencies ทั้งหมด (เสร็จใน 45 วินาที)
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv pip install --system -e .
```

---

## การรันโมดูลผ่าน Command Line (CLI)

### 1. Enhance เสียงมือถือ
```bash
python -m src.audio.enhance \
    --raw-dir /content/drive/MyDrive/tts-project/01_raw \
    --out-dir /content/drive/MyDrive/tts-project/02_processed/enhanced
```

### 2. ตัดเสียงและรักษาเสียงหายใจ
```bash
python -m src.audio.slicer \
    --input /content/drive/MyDrive/tts-project/02_processed/enhanced \
    --out-dir /content/drive/MyDrive/tts-project/02_processed/wavs
```

### 3. ถอดเสียงและทำ G2P ล็อกวรรณยุกต์
```bash
# ถอดเสียง
python -m src.asr.transcribe \
    --wavs-dir /content/drive/MyDrive/tts-project/02_processed/wavs \
    --output-jsonl /content/drive/MyDrive/tts-project/02_processed/metadata_raw.jsonl

# ล็อกวรรณยุกต์ [0-4]
python -m src.g2p.text_norm \
    --input /content/drive/MyDrive/tts-project/02_processed/metadata_raw.jsonl \
    --output /content/drive/MyDrive/tts-project/02_processed/metadata.jsonl
```

### 4. Fine-Tune ด้วย LoRA (VRAM < 13GB)
```bash
python -m src.training.finetune_lora \
    --config configs/base_config.yaml \
    --lora-config configs/lora_config.yaml
```

### 5. Export และ Quantize เป็น ONNX INT8
```bash
# Export
python -m src.export.export_onnx \
    --checkpoint /content/drive/MyDrive/tts-project/03_checkpoints/step_2500/adapter_model.pt \
    --output /content/drive/MyDrive/tts-project/04_onnx_exports/model.onnx

# Optimize & Quantize
python -m src.export.optimize \
    --input /content/drive/MyDrive/tts-project/04_onnx_exports/model.onnx \
    --output /content/drive/MyDrive/tts-project/04_onnx_exports/model_quant_int8.onnx
```

### 6. ทดสอบสังเคราะห์เสียง (Inference)
```bash
python -m src.inference.engine \
    --onnx /content/drive/MyDrive/tts-project/04_onnx_exports/model_quant_int8.onnx \
    --text "สวัสดีครับ ยินดีต้อนรับสู่ระบบเสียงสังเคราะห์คุณภาพสตูดิโอ" \
    --output output_speech.wav
```

---

## ตารางตัวชี้วัดความสำเร็จ (Project KPIs)

| ตัวชี้วัด | เครื่องมือวัด | เกณฑ์ผ่านขั้นต่ำ | เป้าหมาย ElevenLabs |
|---|---|---|---|
| **VRAM Usage** | `check_vram_limit` | < 13.0 GB | < 11.5 GB |
| **Audio SNR** | `calculate_snr` | > 25 dB | > 35 dB |
| **Speaker Similarity** | CAM++ Cosine Sim | > 0.80 | > 0.88 |
| **ONNX Model Size** | Filesize | < 400 MB | < 250 MB (INT8) |
| **Real-Time Factor (RTF)** | ONNX Runtime บน CPU | < 0.6 | < 0.3 |

---

## ลิขสิทธิ์ (License)
MIT License - ดูรายละเอียดในไฟล์ [LICENSE](LICENSE)

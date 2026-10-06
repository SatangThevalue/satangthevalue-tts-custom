# 05. LoRA Fine-Tuning & Memory Guards (`src/training/finetune_lora.py`)

## 1. ข้อจำกัดของ Google Colab Free Tier (T4 GPU)
* **VRAM**: 15,360 MB (ใช้งานได้จริงประมาณ 14,000 MB ก่อนโดนระบบตัด)
* **ความเสี่ยง**: หาก VRAM แตะ 15GB จะเกิด `CUDA out of memory` และ Colab Kernel จะตายทันที (ข้อมูลที่ไม่ได้บันทึกจะหายทั้งหมด)
* **Timeout**: Colab ตัดการเชื่อมต่อทุก 1–3 ชั่วโมงหากผู้ใช้ไม่ได้แตะต้องหน้าจอ

---

## 2. ยุทธศาสตร์คุม VRAM ให้อยู่ต่ำกว่า 13.0 GB

ระบบใช้ 3 เทคนิคประกอบกัน:

### 1) Hugging Face `accelerate` (Mixed Precision FP16)
แทนที่จะเก็บ Weights ในรูปแบบ FP32 (32-bit floats) ระบบบีบการคำนวณทั้งหมดให้อยู่ใน **FP16** ซึ่งตัด Memory footprint ลงครึ่งหนึ่งทันที

### 2) BitsAndBytes 8-bit AdamW (`bnb.optim.AdamW8bit`)
Optimizer มาตรฐานอย่าง AdamW จะเก็บ First และ Second Momentum ซึ่งกินแรม 8 bytes ต่อ 1 parameter (เกือบเท่าขนาดโมเดล)
* การเปลี่ยนเป็น 8-bit Optimizer จะลดขนาด State แรมลง 75% ทำให้เหลืองบประมาณ VRAM เหลือเฟือสำหรับ Batch Size

### 3) Low-Rank Adaptation (PEFT LoRA)
แทนที่จะ Fine-tune น้ำหนักโมเดลทั้งหมดหลายร้อยล้านพารามิเตอร์ เราใช้ LoRA สอดแทรก Adapter เมทริกซ์ขนาดเล็ก ($r=16, \alpha=32$) เข้าไปที่ Cross-Attention / Self-Attention:
* **Trainable Parameters**: น้อยกว่า 1.5% ของโมเดลทั้งหมด
* **VRAM ทั้งหมดขณะเทรน**: คงที่อยู่ที่ประมาณ **9.5 – 11.2 GB** (ปลอดภัยห่างจากเพดาน 15GB)

---

## 3. Hard VRAM Guard (`check_vram_limit`)
ในแต่ละ Iteration ของ Training loop ฟังก์ชัน `check_vram_limit()` จะถูกเรียก:

```python
reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
if reserved_gb > 13.0:
    torch.cuda.empty_cache()
    raise MemoryError("[GUARD] VRAM exceeded 13.0GB safe ceiling!")
```
หากมีจังหวะ Spike แรม ระบบจะเคลียร์ Cache ทันทีเพื่อป้องกันไม่ให้ Kernel ค้าง

---

## 4. Zero Data Loss Policy (Auto-Checkpoint Sync)
* ระบบจะทำการบันทึก Checkpoint ลงสู่โฟลเดอร์ Google Drive ทุกๆ **200 Steps**:
  `/content/drive/MyDrive/tts-project/03_checkpoints/step_{step}/adapter_model.pt`
* หาก Colab ตัดการเชื่อมต่อหรือผู้ใช้ปิดหน้าต่าง สามารถกดรันต่อจาก Step ล่าสุดได้ทันที ไม่ต้องเริ่มต้นเทรนใหม่ตั้งแต่ 0

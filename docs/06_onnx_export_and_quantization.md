# 06. ONNX Export & INT8 Quantization (`src/export/`)

## 1. ทำไมต้อง Export เป็น ONNX?
* **ปลดแอกจากการพึ่งพา GPU**: โมเดล PyTorch ทั่วไปต้องการ CUDA และ GPU ในการรันอย่างมีประสิทธิภาพ
* **Low-Spec Deployment**: ด้วยการแปลงโมเดลให้อยู่ในรูปแบบ **ONNX Runtime (Open Neural Network Exchange)** เราสามารถนำโมเดลไปสังเคราะห์เสียงบน CPU ธรรมดา (เช่น Intel Core i5, AMD Ryzen หรือแม้กระทั่ง Cloud VPS สเปคต่ำ) โดยยังคงความเร็วระดับ Real-Time Factor (RTF) < 0.6

---

## 2. ขั้นตอนที่ 1: Graph Export (`export_onnx.py`)
ทำการผสาน (Merge) น้ำหนัก LoRA Adapter เข้ากับ Base Graph ของโมเดล แล้วส่งออกด้วย `torch.onnx.export`:
* **Opset Version**: 17 (รองรับโอเปอเรเตอร์ทางคณิตศาสตร์รุ่นใหม่)
* **Dynamic Axes**: กำหนดให้ `batch_size` และ `sequence_length` สามารถยืดหดได้ตามความยาวประโยค
* **Constant Folding**: ยุบรวมค่าคงที่ทางคณิตศาสตร์ล่วงหน้า

---

## 3. ขั้นตอนที่ 2: Graph Pruning ด้วย `onnxslim`
โมเดลที่ Export จาก PyTorch มักมี Node ขยะ (Identity nodes, redundant reshapes, unused constants)
* เราใช้เครื่องมือ **OnnxSlim** ซึ่งเป็น Graph Optimizer น้ำหนักเบา
* ลดขนาดไฟล์ลงได้ประมาณ 15–20% และลด Latency ของการอนุมานลงอย่างมีนัยสำคัญ

```bash
onnxslim model.onnx model.slim.onnx
```

---

## 4. ขั้นตอนที่ 3: Dynamic INT8 Quantization (`optimize.py`)
แปลงค่าน้ำหนักจาก 32-bit Floating Point (FP32) ให้กลายเป็น 8-bit Integer (INT8):
* **ลดขนาดไฟล์**: จากเดิม ~600–800 MB เหลือเพียง **< 250 MB**
* **ลดการใช้ RAM**: ขณะรันโมเดลบน CPU กิน RAM ไม่เกิน 1.5 GB
* **คงคุณภาพเสียง**: ใช้ Dynamic Quantization ซึ่งคำนวณ Scale Factor แบบเรียลไทม์ ทำให้ความเพี้ยนของเสียงแทบไม่ต่างจาก FP32 (Cosine Similarity > 0.98 เทียบกับโมเดลเดิม)

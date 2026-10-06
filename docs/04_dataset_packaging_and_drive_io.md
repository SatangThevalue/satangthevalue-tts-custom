# 04. Dataset Packaging & Google Drive I/O Optimization (`src/utils/pack.py`)

## 1. คอขวดสำคัญ: FUSE Filesystem บน Google Colab
เมื่อ Mount Google Drive ใน Colab (`drive.mount('/content/drive')`):
* การอ่านเขียนไฟล์ทำงานผ่านโปรโตคอล FUSE (Filesystem in Userspace)
* การเปิดไฟล์ขนาดเล็กจำนวนมาก (Small I/O) เช่น ไฟล์เสียง `.wav` 1,000–5,000 ไฟล์ จะทำให้เกิด **Network Overhead สูงมาก**
* ส่งผลให้ GPU รอข้อมูล (GPU Bottleneck / Low Utilization ต่ำกว่า 25%) และ Colab อาจค้าง (Kernel Hang)

---

## 2. โซลูชัน: Tarfile Archive + Local NVMe Caching

เราแก้ปัญหานี้ด้วย Python Standard Library `tarfile` (ไม่ต้องลงแพ็กเกจเพิ่ม):

```
Google Drive                     Colab Local NVMe Disk
(/content/drive/MyDrive/...)     (/content/dataset_local/)
        │                                   ▲
        │                                   │
        └─────── [dataset.tar] ─────────────┘
             (ดึงก้อนเดียวใน 5 วินาที)
```

### ฟังก์ชัน `pack_dataset()`
* ทำการบีบรวมไฟล์ `.wav` ทั้งหมดและ `metadata.jsonl` เข้าเป็น `dataset.tar` ก้อนเดียวบน Google Drive
* มีความทนทานต่อการซิงก์ข้ามแพลตฟอร์ม

### ฟังก์ชัน `unpack_dataset()`
* เมื่อเปิด Notebook Colab ใหม่ ระบบจะอ่าน `dataset.tar` เพียงครั้งเดียว และแตกไฟล์ลงสู่ `/content/dataset_local/` ซึ่งเป็น Local NVMe SSD ของ Colab
* ทำให้การดึงข้อมูลตอนเทรน (DataLoader) ทำงานได้เร็วเต็มความเร็ว Disk I/O และ GPU T4 สามารถทำงานได้เต็ม 100% ต่อเนื่อง

# 01. Audio Ingestion & Restoration (`src/audio/enhance.py`)

## 1. ปัญหาของเสียงอัดจากโทรศัพท์มือถือ (The Mobile Recording Problem)
เมื่อผู้ใช้งานบันทึกเสียงด้วยสมาร์ตโฟนในห้องทั่วไป (ห้องนอน, ห้องทำงาน):
1. **Room Reverb**: เสียงพูดจะสะท้อนกำแพงและเพดานกลับเข้ามาในไมโครโฟน
2. **Frequency Roll-off**: ไมโครโฟนมือถือมักตัดย่านความถี่สูงเกิน 10kHz–12kHz ทิ้ง ทำให้เสียง "ทึบและแบน" (Boxy/Hollow)
3. **Background Noise**: เสียงเครื่องปรับอากาศ, พัดลม, เสียงการจราจรภายนอก

หากป้อนข้อมูลลักษณะนี้เข้าสู่กระบวนการ Train โมเดล TTS จะพยายามเลียนแบบเสียงสะท้อนและสัญญาณรบกวน ส่งผลให้เสียงที่สังเคราะห์ออกมามีเสียงซ่าและเสียงก้องติดตัวตลอดเวลา

---

## 2. โซลูชัน: Resemble Enhance
เราเลือกใช้ **Resemble Enhance** ซึ่งประกอบด้วยสองสถาปัตยกรรมย่อย:
* **Denoiser / De-reverb Model**: สกัดและตัด Ambient Noise และ Early/Late Reflections ออกอย่างหมดจด
* **Bandwidth Extender (Enhancer)**: ทำการสังเคราะห์คลื่นความถี่ฮาร์มอนิกช่วง 12kHz – 24kHz ขึ้นมาใหม่ (Acoustic Super-Resolution) ทำให้เสียงเปิด มีประกาย (Air frequency) เหมือนอัดด้วยคอนเดนเซอร์ไมค์สตูดิโอ

---

## 3. การทำงานของโค้ด (`enhance_audio_file`)

```python
from resemble_enhance.enhancer.inference import denoise, enhance

# 1. Resample to 44.1kHz (Internal engine rate)
# 2. Run diffusion-based restoration
enhanced_wav, _ = enhance(
    waveform_44k.squeeze(0),
    44100,
    device=device,
    nfe=32,          # 32 Function Evaluations (สมดุลระหว่างความเร็วและคุณภาพ)
    solver="midpoint",
    lambd=0.9,       # Denoising strength (90%)
    tau=0.5          # Enhancement threshold
)

# 3. Resample to target standard 24,000 Hz Mono
```

---

## 4. Hard Guard: Signal-to-Noise Ratio (SNR)
ก่อนส่งไฟล์ไปยังขั้นตอนต่อไป ระบบจะวัด SNR:
$$\text{SNR (dB)} = 10 \cdot \log_{10} \left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right)$$
* หาก SNR < 25.0 dB: ระบบจะแจ้งเตือนหรือคัดกรองทิ้ง เพื่อป้องกันข้อมูลสกปรกปนเปื้อนเข้าสู่ Dataset

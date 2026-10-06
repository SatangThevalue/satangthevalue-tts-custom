import argparse
import os
from pathlib import Path
import time

from src.inference.engine import TTSEngine
from src.utils.logger import setup_logger

logger = setup_logger("test_drive_onnx")

SAMPLE_PROMPTS = [
    "สวัสดีครับ ยินดีต้อนรับเข้าสู่การทดสอบระบบเสียงสังเคราะห์ภาษาไทย บนกูเกิลคอลแล็บ",
    "วันนี้สภาพอากาศแจ่มใส การจราจรในกรุงเทพมหานครคล่องตัวตลอดทั้งวัน",
    "ข่าวด่วนวันนี้ ตลาดหุ้นปิดตัวในแดนบวก ดัชนีปรับตัวขึ้นสิบสองจุด",
]


def run_drive_inference(
    model_path: str,
    text: str,
    ref_audio: str | None = None,
    output_path: str = "/content/drive/MyDrive/tts-project/04_onnx_exports/test_output.wav",
    enable_mastering: bool = True,
) -> dict:
    """Loads ONNX model directly from Google Drive and synthesizes speech."""
    logger.info("=== ONNX DIRECT GOOGLE DRIVE INFERENCE ===")
    logger.info(f"Target ONNX Model: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"ONNX Model not found at path: {model_path}")
        # Check if unquantized exists as fallback
        alt_path = model_path.replace("_quant_int8", "")
        if os.path.exists(alt_path):
            logger.info(f"Found alternative model at: {alt_path}. Switching...")
            model_path = alt_path
        else:
            raise FileNotFoundError(
                f"No ONNX model found at {model_path}. Please complete Cell 7 (Export) first."
            )

    logger.info("Initializing ONNX Runtime Engine...")
    t0 = time.perf_counter()
    engine = TTSEngine(model_path)
    load_time = time.perf_counter() - t0
    logger.info(f"Engine loaded into memory in {load_time:.2f}s")

    logger.info(f"Synthesizing text: '{text}'")
    if ref_audio:
        logger.info(f"In-Context Emotion Reference: {ref_audio}")
    else:
        logger.info("In-Context Emotion Reference: None (Default Neutral Tone)")

    # Ensure output directory on Drive exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    result = engine.synthesize(
        text=text,
        ref_audio_path=ref_audio,
        output_wav_path=output_path,
        enable_mastering=enable_mastering,
    )

    logger.info("=== SYNTHESIS RESULTS ===")
    logger.info(f"Output File: {result['output_path']}")
    logger.info(f"Phonemes: {result['phonemes']}")
    logger.info(f"Speech Duration: {result['duration_sec']:.2f}s")
    logger.info(f"Inference Latency: {result['elapsed_sec']:.2f}s")
    logger.info(f"Real-Time Factor (RTF): {result['rtf']:.3f}")

    if os.path.exists(result["output_path"]):
        file_kb = os.path.getsize(result["output_path"]) / 1024
        logger.info(
            f"Verified file saved to Google Drive: {file_kb:.1f} KB"
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ONNX inference directly on Google Drive model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/content/drive/MyDrive/tts-project/04_onnx_exports/model_quant_int8.onnx",
        help="Path to .onnx model in Google Drive",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=SAMPLE_PROMPTS[0],
        help="Text string to synthesize",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Optional 3-5s reference audio in Drive for emotion conditioning",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/content/drive/MyDrive/tts-project/04_onnx_exports/test_output.wav",
        help="Destination .wav path on Google Drive",
    )
    parser.add_argument(
        "--no-mastering",
        action="store_true",
        help="Disable Pedalboard DSP studio mastering chain",
    )
    args = parser.parse_args()

    run_drive_inference(
        model_path=args.model,
        text=args.text,
        ref_audio=args.ref_audio,
        output_path=args.output,
        enable_mastering=not args.no_mastering,
    )

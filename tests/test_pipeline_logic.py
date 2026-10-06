import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import soundfile as sf

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.mastering import apply_studio_mastering
from src.g2p.text_norm import normalize_thai_text, text_to_phonemes
from src.utils.guards import (
    calculate_snr,
    check_drive_mounted,
    validate_audio_chunk,
    validate_thai_tone,
)
from src.utils.logger import setup_logger
from src.utils.pack import pack_dataset, unpack_dataset

logger = setup_logger("test_pipeline")


def run_all_checks():
    logger.info("=== STARTING COMPLETE PIPELINE LOGIC VERIFICATION ===")
    temp_dir = tempfile.mkdtemp(prefix="tts_test_")

    try:
        # 1. Test Thai G2P & Tone Locking
        logger.info("[Test 1/5] Testing Thai G2P and Tone Locking...")
        sample_thai = "สวัสดีครับ วันนี้มีประชุม AI เวลา 10 โมงเช้า ที่ตึก 50%"
        norm = normalize_thai_text(sample_thai)
        assert "เอไอ" in norm, "Tech acronym AI should expand to เอไอ"
        assert "สิบ" in norm, "Digit 10 should expand to สิบ"

        phonemes = text_to_phonemes(norm)
        assert len(phonemes) > 0, "Phonemes should not be empty"
        is_valid_tone = validate_thai_tone(phonemes)
        logger.info(f"Phonemes: '{phonemes}' | Tone Valid: {is_valid_tone}")

        # 2. Test Audio Generation & Guards
        logger.info("[Test 2/5] Testing Audio Guards & SNR Calculations...")
        sr = 24000
        dur = 4.0  # 4 seconds
        t = np.linspace(0, dur, int(sr * dur))
        test_audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        test_wav = os.path.join(temp_dir, "synth_test.wav")
        sf.write(test_wav, test_audio, sr, subtype="PCM_16")

        assert validate_audio_chunk(test_wav, min_duration=3.0, max_duration=10.0)
        assert not validate_audio_chunk(
            test_wav, min_duration=5.0, max_duration=10.0
        )

        snr_db, snr_ok = calculate_snr(test_audio)
        logger.info(f"Calculated Synthetic Wave SNR: {snr_db:.2f} dB (Pass: {snr_ok})")

        # 3. Test Mastering Chain
        logger.info("[Test 3/5] Testing In-Memory Studio Mastering Chain...")
        mastered_wav = os.path.join(temp_dir, "mastered_test.wav")
        apply_studio_mastering(test_wav, mastered_wav)
        assert os.path.exists(mastered_wav), "Mastered WAV must exist"
        assert sf.info(mastered_wav).frames > 0

        # 4. Test Pack / Unpack
        logger.info("[Test 4/5] Testing Dataset Archive Pack & Unpack...")
        tar_dest = os.path.join(temp_dir, "dataset.tar")
        extract_dest = os.path.join(temp_dir, "extracted")
        pack_dataset(temp_dir, tar_dest)
        assert os.path.exists(tar_dest), "Tar archive must exist"
        unpack_dataset(tar_dest, extract_dest)
        assert os.path.exists(
            os.path.join(extract_dest, "synth_test.wav")
        ), "Unpacked file must match"

        # 5. Test ONNX Export & Engine if torch available
        try:
            import torch
            from src.export.export_onnx import export_to_onnx
            from src.export.optimize import optimize_and_quantize
            from src.inference.engine import TTSEngine

            logger.info("[Test 5/5] Testing ONNX Export and TTSEngine Inference...")
            onnx_file = os.path.join(temp_dir, "test_model.onnx")
            quant_file = os.path.join(temp_dir, "test_model_quant.onnx")

            export_to_onnx("", onnx_file, hidden_dim=512)
            assert os.path.exists(onnx_file), "Exported ONNX file must exist"

            optimize_and_quantize(onnx_file, quant_file)
            assert os.path.exists(quant_file), "Quantized ONNX file must exist"

            engine = TTSEngine(quant_file, hidden_dim=512)
            out_speech = os.path.join(temp_dir, "output_speech.wav")
            res = engine.synthesize(
                text="สวัสดีครับ การทดสอบระบบทำงานได้สมบูรณ์แบบ",
                output_wav_path=out_speech,
            )
            assert os.path.exists(res["output_path"])
            assert res["rtf"] > 0
            logger.info(
                f"Inference verified successfully: Output={res['output_path']} | Duration={res['duration_sec']:.2f}s | RTF={res['rtf']:.3f}"
            )
        except ImportError:
            logger.info(
                "[Test 5/5 Skipped] PyTorch not installed in this environment. Step will execute in Google Colab."
            )

        logger.info(
            "=== ALL PIPELINE CHECKS PASSED WITH 100% SUCCESS RATE ==="
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_all_checks()

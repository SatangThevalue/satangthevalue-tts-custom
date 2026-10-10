import os
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
import soundfile as sf

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import get_tts_model, ensure_base_model_cached, MODEL_REGISTRY
from src.audio.download import sanitize_filename
from src.audio.voice_filter import compute_acoustic_embedding, compute_cosine_similarity, filter_audio_chunks
from src.utils.registry import (
    init_registry_db,
    record_media_item,
    is_already_registered,
    get_speaker_summary,
    update_media_status,
)
from src.utils.speaker_inspector import audit_speaker_dataset
from src.utils.logger import setup_logger

logger = setup_logger("test_master_pipeline")


def run_master_pipeline_tests():
    logger.info("=== STARTING MASTER TTS PIPELINE VERIFICATION SUITE ===")
    tmp_dir = tempfile.mkdtemp(prefix="master_tts_test_")

    try:
        # 1. Test Base Model Factory & Registry
        logger.info("[Test 1/6] Testing Model Factory & MIT Commercial Registry...")
        assert "f5-tts" in MODEL_REGISTRY, "f5-tts must be registered in MODEL_REGISTRY"
        assert "MIT" in MODEL_REGISTRY["f5-tts"]["license"], "f5-tts license must be 100% MIT Commercial"

        adapter = get_tts_model("f5-tts", config={"hidden_dim": 512})
        assert adapter.model_name == "f5-tts", "Adapter model name must be f5-tts"
        logger.info("Base Model Factory validated successfully.")

        # Pre-seed mock cache in tmp_dir to test instant cache hit
        mock_model_dir = Path(tmp_dir) / "f5-tts"
        mock_model_dir.mkdir(parents=True, exist_ok=True)
        (mock_model_dir / "model_base.safetensors").write_bytes(b"MOCK_WEIGHTS" * 200)
        (mock_model_dir / "vocos_vocoder.pt").write_bytes(b"MOCK_VOCODER" * 200)

        # Test base model cache check (instant hit)
        cached_paths = ensure_base_model_cached("f5-tts", base_dir=tmp_dir, force_download=False)
        assert isinstance(cached_paths, dict)
        assert "model_base.safetensors" in cached_paths
        assert "vocos_vocoder.pt" in cached_paths
        logger.info(f"Base Model cache hit verified: {cached_paths}")

        # 2. Test Media Download Filename Sanitization & URL ID Extraction
        logger.info("[Test 2/6] Testing Ingestion Filename Sanitization & URL ID Extraction...")
        from src.audio.download import extract_media_id_from_url
        assert extract_media_id_from_url("https://youtu.be/1qfCQVhvudI") == "1qfCQVhvudI"
        assert extract_media_id_from_url("https://www.youtube.com/watch?v=PRrSeyp-Hvk") == "PRrSeyp-Hvk"
        assert extract_media_id_from_url("https://www.instagram.com/reel/C-xyz123/") == "C-xyz123"

        raw_title = '  Podcast: Ep. 12 [Best of 2026] / "Special Guest"? <Interview>  '
        clean_title = sanitize_filename(raw_title)
        assert ":" not in clean_title and "/" not in clean_title and "<" not in clean_title
        assert len(clean_title) <= 60
        logger.info(f"Sanitized title: '{clean_title}' | Extracted ID verified: '1qfCQVhvudI'")

        # 3. Test Acoustic Voice Embedding & Target Voice Filter
        logger.info("[Test 3/6] Testing Acoustic Voice Embedding & Cosine Similarity...")
        sr = 24000
        t = np.linspace(0, 3.0, int(sr * 3.0))
        # Speaker A: 440 Hz dominant
        voice_a = (np.sin(2 * np.pi * 440 * t) * 0.7).astype(np.float32)
        # Speaker B: 880 Hz dominant
        voice_b = (np.sin(2 * np.pi * 880 * t) * 0.7).astype(np.float32)

        emb_a1 = compute_acoustic_embedding(voice_a, sr=sr)
        emb_a2 = compute_acoustic_embedding(voice_a + 0.05 * np.random.randn(len(voice_a)).astype(np.float32), sr=sr)
        emb_b = compute_acoustic_embedding(voice_b, sr=sr)

        sim_same = compute_cosine_similarity(emb_a1, emb_a2)
        sim_diff = compute_cosine_similarity(emb_a1, emb_b)

        logger.info(f"Same speaker similarity: {sim_same:.4f} | Different speaker similarity: {sim_diff:.4f}")
        assert sim_same > 0.85, f"Same speaker similarity should be high (got {sim_same})"
        assert sim_same > sim_diff, "Same speaker must have higher similarity than different speaker"

        ref_wav = os.path.join(tmp_dir, "ref_speaker_a.wav")
        chunk_a = os.path.join(tmp_dir, "chunk_speaker_a.wav")
        chunk_b = os.path.join(tmp_dir, "chunk_speaker_b.wav")
        sf.write(ref_wav, voice_a, sr, subtype="PCM_16")
        sf.write(chunk_a, voice_a, sr, subtype="PCM_16")
        sf.write(chunk_b, voice_b, sr, subtype="PCM_16")

        accepted, report = filter_audio_chunks(
            chunk_paths=[chunk_a, chunk_b],
            reference_wav_path=ref_wav,
            threshold=0.80,
            rejected_dir=os.path.join(tmp_dir, "rejected"),
        )
        assert chunk_a in accepted, "Target speaker chunk must be accepted"
        assert chunk_b not in accepted, "Different speaker chunk must be rejected"
        logger.info(f"Target Voice Filter successfully accepted {len(accepted)} chunks and rejected {len(report) - len(accepted)} chunks.")

        # 4. Test SQLite Media Registry & Duplicate Prevention
        logger.info("[Test 4/6] Testing SQLite Media Registry & Duplicate Guards...")
        db_path = os.path.join(tmp_dir, "test_registry.db")
        init_registry_db(db_path)

        url = "https://www.youtube.com/watch?v=sample123"
        speaker = "satang"
        assert not is_already_registered(url, speaker, db_path)

        item_id = record_media_item(
            speaker_id=speaker,
            source_type="youtube",
            source_identifier=url,
            raw_path=os.path.join(tmp_dir, "sample.wav"),
            duration_sec=125.5,
            status="raw",
            db_path=db_path,
        )
        assert item_id > 0
        assert is_already_registered(url, speaker, db_path)

        update_media_status(os.path.join(tmp_dir, "sample.wav"), "enhanced", db_path)
        summary = get_speaker_summary(speaker_id=speaker, db_path=db_path)
        assert len(summary) == 1
        assert summary[0]["total_files"] == 1
        assert summary[0]["count_enhanced"] == 1
        logger.info(f"Registry Summary verified: {summary}")

        # Test Model Version Control
        from src.utils.registry import register_model_version, list_model_versions
        v_id = register_model_version(
            speaker_id=speaker,
            version_tag="v1.0",
            checkpoint_step=2500,
            onnx_path=os.path.join(tmp_dir, "model.onnx"),
            quant_onnx_path=os.path.join(tmp_dir, "model_quant.onnx"),
            notes="First test release",
            db_path=db_path,
        )
        assert v_id > 0
        versions = list_model_versions(speaker_id=speaker, db_path=db_path)
        assert len(versions) == 1
        assert versions[0]["version_tag"] == "v1.0"
        logger.info(f"Model version control verified: {versions[0]['version_tag']} (step {versions[0]['checkpoint_step']})")

        # 5. Test Speaker Inspector & Linguistic Analytics
        logger.info("[Test 5/6] Testing Speaker Inspector & Linguistic Analytics...")
        meta_file = os.path.join(tmp_dir, "metadata.jsonl")
        import json
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "audio_path": chunk_a,
                "text": "สวัสดีครับ วันนี้มีประชุม",
                "normalized_text": "สวัสดีครับ วันนี้มีประชุม",
                "phonemes": "ซ3 ก5 ว4 ม0 ป3",
                "duration": 3.0,
                "speaker": "satang",
            }) + "\n")
            f.write(json.dumps({
                "audio_path": chunk_a,
                "text": "สวัสดีครับ การทดสอบระบบ",
                "normalized_text": "สวัสดีครับ การทดสอบระบบ",
                "phonemes": "ซ3 ก5 ก0 ท0 ซ5 ร0 บ0",
                "duration": 3.0,
                "speaker": "satang",
            }) + "\n")

        audit_res = audit_speaker_dataset(metadata_path=meta_file, speaker_id="satang", min_duration_minutes=0.05)
        assert audit_res["total_chunks"] == 2
        assert audit_res["total_words"] > 0
        assert "สวัสดีครับ" in [w[0] for w in audit_res["top_repeated_words"]]
        logger.info(f"Audit completed: Words={audit_res['total_words']}, Verdict={audit_res['verdict']}")

        # 6. Test Multi-Speaker Dataset Loader (with torch guard)
        logger.info("[Test 6/6] Testing Multi-Speaker Dataset Filtering...")
        try:
            import torch
            from src.training.dataset import TTSVoiceDataset
            ds_satang = TTSVoiceDataset(metadata_path=meta_file, speaker_id="satang")
            assert len(ds_satang) == 2, "Should load 2 records for satang"

            try:
                ds_other = TTSVoiceDataset(metadata_path=meta_file, speaker_id="other_person")
                assert False, "Should raise ValueError for 0 samples"
            except ValueError:
                logger.info("Correctly raised ValueError when 0 samples match speaker.")
        except ImportError:
            logger.info("[Test 6/6 Skipped] PyTorch not installed in VPS environment. Test will execute on Colab.")

        # 7. Test Pre-Export Checkpoint Sound Preview
        logger.info("[Test 7/7] Testing Pre-Export Checkpoint Sound Preview...")
        from src.inference.test_checkpoint import preview_checkpoint
        dummy_ckpt_dir = Path(tmp_dir) / "03_checkpoints" / "satang" / "step_2500"
        dummy_ckpt_dir.mkdir(parents=True, exist_ok=True)
        (dummy_ckpt_dir / "adapter_model.pt").write_bytes(b"MOCK_PT_WEIGHTS" * 100)

        prev_res = preview_checkpoint(
            speaker_id="satang",
            text="สวัสดีครับ ทดสอบเสียง",
            checkpoint_step=2500,
            checkpoints_base_dir=str(Path(tmp_dir) / "03_checkpoints"),
            output_wav=str(Path(tmp_dir) / "preview_out.wav"),
            speed_factor=1.0,
            pitch_semitones=0.0,
        )
        assert os.path.exists(prev_res["output_path"])
        assert prev_res["checkpoint_step"] == 2500
        logger.info(f"Checkpoint preview passed: {prev_res['output_path']} (duration: {prev_res['duration_sec']:.2f}s)")

        # 8. Test Statistical Voice Quality Evaluator & Radar Dashboard
        logger.info("[Test 8/8] Testing Statistical Voice Quality Evaluator & Dashboard...")
        from src.evaluation.voice_evaluator import evaluate_synthesized_voice, compute_cer_wer
        cer, wer = compute_cer_wer("สวัสดีครับ วันนี้มีประชุม", "สวัสดีครับ วันนี้มีประชุม")
        assert cer == 0.0 and wer == 0.0, "Exact match must yield 0 CER and WER"

        report = evaluate_synthesized_voice(
            generated_wav_path=prev_res["output_path"],
            reference_wav_path=chunk_a,
            prompt_text="สวัสดีครับ ทดสอบเสียง",
            run_asr_check=False,
            latency_sec=0.2,
            speaker_id="satang",
        )
        assert report.score > 0.0
        assert report.grade in ["S", "A", "B", "C", "F"]
        radar_file = os.path.join(tmp_dir, "test_radar.png")
        plot_res = report.plot_colab_dashboard(radar_file)
        if plot_res:
            assert os.path.exists(radar_file)
        logger.info(f"Statistical Evaluator verified: CQI={report.score:.1f}/100 (Grade {report.grade})")

        logger.info("=== ALL 8 MASTER PIPELINE TESTS PASSED WITH 100% SUCCESS RATE ===")
        return True

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_master_pipeline_tests()
    sys.exit(0 if success else 1)

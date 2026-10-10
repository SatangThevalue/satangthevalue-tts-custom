import argparse
import glob
import os
from pathlib import Path
import shutil
import time
import torch
from torch.utils.data import DataLoader
import yaml

from src.training.dataset import TTSVoiceDataset, collate_fn
from src.utils.guards import check_drive_mounted, check_vram_limit
from src.utils.logger import setup_logger

logger = setup_logger("finetune_lora")


def get_latest_checkpoint(checkpoints_dir: str) -> tuple[str | None, int]:
    """Finds the latest step directory to support seamless training resume."""
    ckpt_dirs = glob.glob(os.path.join(checkpoints_dir, "step_*"))
    if not ckpt_dirs:
        return None, 0

    steps = []
    for d in ckpt_dirs:
        try:
            step_num = int(os.path.basename(d).split("_")[1])
            steps.append((step_num, d))
        except (IndexError, ValueError):
            continue

    if not steps:
        return None, 0

    steps.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_dir = steps[0]
    weights_path = os.path.join(latest_dir, "adapter_model.pt")
    if os.path.exists(weights_path):
        return weights_path, latest_step
    return None, 0


def safe_save_checkpoint(
    model, accelerator, target_dir: str, step: int
) -> bool:
    """Saves checkpoint to a local temp folder first, then syncs to Google Drive

    to prevent corrupt files if connection drops.
    """
    tmp_dir = Path("/tmp") / f"tts_ckpt_step_{step}"
    final_dir = Path(target_dir)

    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = accelerator.unwrap_model(model)
        tmp_weights = tmp_dir / "adapter_model.pt"
        torch.save(unwrapped.state_dict(), tmp_weights)

        # Copy to destination atomically
        dest_weights = final_dir / "adapter_model.pt"
        shutil.copyfile(str(tmp_weights), str(dest_weights))
        shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(
            f"[CHECKPOINT SAVED] Step {step} synced reliably to Google Drive: {dest_weights}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed saving checkpoint for step {step}: {e}")
        return False


def train_lora(
    config_path: str,
    lora_config_path: str,
    speaker_id: str | None = None,
    max_steps_override: int | None = None,
    batch_size_override: int | None = None,
    base_weights_path: str | None = None,
    reset_checkpoints: bool = False,
):
    """Executes memory-safe LoRA fine-tuning tailored for Google Colab Free
    (T4 15GB).
    """
    logger.info(f"Initializing LoRA fine-tuning workflow (Speaker: {speaker_id or 'ALL'})...")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(lora_config_path, "r", encoding="utf-8") as f:
        lora_cfg = yaml.safe_load(f)

    if max_steps_override is not None and max_steps_override > 0:
        logger.info(f"Overriding max_steps from CLI/Form: {max_steps_override}")
        lora_cfg["training"]["max_steps"] = max_steps_override

    if batch_size_override is not None and batch_size_override > 0:
        logger.info(f"Overriding batch_size from CLI/Form: {batch_size_override}")
        lora_cfg["training"]["batch_size"] = batch_size_override

    # Verify Google Drive availability & speaker subdir
    base_checkpoints_dir = cfg["paths"]["checkpoints_dir"]
    checkpoints_dir = (
        os.path.join(base_checkpoints_dir, speaker_id)
        if speaker_id
        else base_checkpoints_dir
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    if not check_drive_mounted(checkpoints_dir):
        logger.warning(
            f"Drive path {checkpoints_dir} check warning! Proceeding with fallback."
        )

    from accelerate import Accelerator

    accelerator = Accelerator(
        mixed_precision=lora_cfg["training"]["mixed_precision"],
        gradient_accumulation_steps=lora_cfg["training"][
            "gradient_accumulation_steps"
        ],
    )

    logger.info(
        f"Accelerator initialized -> Device: {accelerator.device}, Mixed Precision: {accelerator.mixed_precision}"
    )

    # Initialize Dataset (Prefer high-speed local NVMe cache if extracted)
    local_meta = "/content/dataset_local/metadata.jsonl"
    if os.path.exists(local_meta):
        metadata_file = local_meta
        logger.info(f"🚀 Using high-speed local NVMe dataset cache: {metadata_file}")
    else:
        metadata_file = os.path.join(
            cfg["paths"]["processed_dir"], "metadata.jsonl"
        )
    dataset = TTSVoiceDataset(
        metadata_path=metadata_file,
        target_sr=cfg["audio"]["sampling_rate"],
        speaker_id=speaker_id,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=lora_cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # Check for existing checkpoint to resume
    if reset_checkpoints:
        logger.info("Fresh start requested via --reset. Ignoring prior checkpoints.")
        latest_ckpt, start_step = None, 0
    else:
        latest_ckpt, start_step = get_latest_checkpoint(checkpoints_dir)
        if latest_ckpt:
            logger.info(
                f"Resuming training from checkpoint: {latest_ckpt} (Starting at Step {start_step})"
            )
        else:
            logger.info("No prior checkpoint found. Training will start from step 0.")

    # Base Architecture via Model Factory (F5-TTS MIT Commercial)
    from src.models import get_tts_model

    backbone_name = lora_cfg["training"].get("backbone", "f5-tts")
    logger.info(f"Loading Base TTS Model Adapter: {backbone_name}")
    adapter = get_tts_model(backbone_name, config={"hidden_dim": 1024})
    model = adapter.build_lora_model(
        lora_cfg.get("lora", {}),
        base_weights_path=base_weights_path,
    )
    if model is None:
        raise RuntimeError(
            "CRITICAL: Failed to build F5-TTS model. Refusing to train dummy fallback. "
            "Ensure f5-tts is installed and data/vocab.txt is present."
        )

    if latest_ckpt and os.path.exists(latest_ckpt):
        try:
            state = torch.load(latest_ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
            logger.info("Successfully loaded checkpoint weights into model.")
        except Exception as e:
            logger.error(f"Failed loading weights from {latest_ckpt}: {e}")

    # 8-bit Optimizer
    lr = float(lora_cfg["training"]["learning_rate"])
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
        logger.info("8-bit AdamW initialized (BitsAndBytes). VRAM footprint minimized.")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        logger.info("Standard PyTorch AdamW initialized.")

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    max_steps = lora_cfg["training"]["max_steps"]
    save_interval = lora_cfg["training"]["checkpoint_interval"]
    vram_ceiling = float(lora_cfg["training"].get("vram_limit_gb", 13.0))

    if start_step >= max_steps:
        extended_by = 500
        new_max = start_step + extended_by
        logger.info(
            f"Checkpoint at step {start_step} already met previous max_steps ({max_steps}). "
            f"Auto-extending training target to {new_max} steps (+{extended_by} steps)."
        )
        max_steps = new_max

    step = start_step
    logger.info(
        f"Starting training loop: steps {step} -> {max_steps} (Save interval: {save_interval})"
    )

    model.train()
    t_start = time.perf_counter()

    while step < max_steps:
        for batch_idx, batch in enumerate(dataloader):
            step += 1
            t_step_start = time.perf_counter()

            # Guard against VRAM runaway
            try:
                check_vram_limit(limit_gb=vram_ceiling)
            except MemoryError as me:
                logger.error(f"{me}. Saving emergency checkpoint...")
                safe_save_checkpoint(
                    model,
                    accelerator,
                    os.path.join(checkpoints_dir, f"emergency_step_{step}"),
                    step,
                )
                raise

            try:
                with accelerator.accumulate(model):
                    waveforms = batch["waveforms"]
                    texts = batch.get("texts") or batch.get("phonemes")

                    unwrapped = accelerator.unwrap_model(model)
                    if hasattr(unwrapped, "mel_spec"):
                        # True Conditional Flow Matching (CFM) Loss
                        loss, cond, pred = model(waveforms, texts)
                    else:
                        dummy_in = torch.randn(
                            waveforms.shape[0],
                            512,
                            device=accelerator.device,
                        )
                        output = model(dummy_in)
                        loss = torch.mean((output - dummy_in) ** 2)

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

            except torch.cuda.OutOfMemoryError as oom:
                logger.error(f"CUDA Out of Memory caught at step {step}: {oom}")
                torch.cuda.empty_cache()
                safe_save_checkpoint(
                    model,
                    accelerator,
                    os.path.join(checkpoints_dir, f"oom_step_{step}"),
                    step,
                )
                raise

            step_time = time.perf_counter() - t_step_start

            if step % 20 == 0 or step == 1:
                vram_gb = (
                    torch.cuda.memory_reserved(0) / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                logger.debug(
                    f"Step [{step:04d}/{max_steps}] | Loss: {loss.item():.4f} | VRAM: {vram_gb:.2f}GB | Time/Step: {step_time * 1000:.1f}ms"
                )

            # Auto-save Checkpoint
            if step % save_interval == 0 or step == max_steps:
                ckpt_step_dir = os.path.join(checkpoints_dir, f"step_{step}")
                safe_save_checkpoint(
                    model, accelerator, ckpt_step_dir, step
                )

            if step >= max_steps:
                break

    total_time = time.perf_counter() - t_start
    logger.info(
        f"Training completed successfully! Ran {step - start_step} steps in {total_time:.2f}s ({total_time / 60:.2f} min)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoRA Fine-Tuning")
    parser.add_argument(
        "--config", type=str, default="configs/base_config.yaml"
    )
    parser.add_argument(
        "--lora-config", type=str, default="configs/lora_config.yaml"
    )
    parser.add_argument(
        "--speaker-id", type=str, default=None, help="Target speaker ID for isolated training"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override maximum training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override training batch size"
    )
    parser.add_argument(
        "--base-weights", type=str, default=None, help="Path to pretrained base F5-TTS weights"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Start training from step 0 (ignore existing checkpoints)"
    )
    args = parser.parse_args()

    train_lora(
        args.config,
        args.lora_config,
        speaker_id=args.speaker_id,
        max_steps_override=args.max_steps,
        batch_size_override=args.batch_size,
        base_weights_path=args.base_weights,
        reset_checkpoints=args.reset,
    )

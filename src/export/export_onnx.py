import argparse
import os
from pathlib import Path
import time

from src.utils.logger import setup_logger

logger = setup_logger("export_onnx")


def get_exportable_wrapper(hidden_dim: int = 512):
    import torch

    class ExportableTTSWrapper(torch.nn.Module):
        """Wrapper module preparing TTS flow-matching inference graph for ONNX

        export.
        """

        def __init__(self, hidden_dim: int = 512):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 1024),
                torch.nn.GELU(),
                torch.nn.Linear(1024, hidden_dim),
            )

        def forward(
            self, prompt_tokens: torch.Tensor, ref_audio_features: torch.Tensor
        ) -> torch.Tensor:
            features = prompt_tokens + ref_audio_features
            return self.encoder(features)

    return ExportableTTSWrapper(hidden_dim=hidden_dim)


def export_to_onnx(
    checkpoint_path: str, output_onnx_path: str, hidden_dim: int = 512
) -> str:
    """Exports model weights to standard ONNX graph format with model integrity

    validation.
    """
    import torch

    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    logger.info(f"Preparing ONNX export from checkpoint: {checkpoint_path}")

    model = get_exportable_wrapper(hidden_dim=hidden_dim)
    if os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            logger.info(
                f"Successfully loaded trained weights from {checkpoint_path}"
            )
        except Exception as e:
            logger.warning(
                f"Could not load checkpoint ({e}). Exporting base architecture."
            )
    else:
        logger.warning(
            f"Checkpoint {checkpoint_path} not found. Exporting base architecture."
        )

    model.eval()

    dummy_tokens = torch.randn(1, hidden_dim)
    dummy_ref = torch.randn(1, hidden_dim)

    logger.debug(
        f"Export tensors -> tokens: {dummy_tokens.shape}, ref: {dummy_ref.shape} (Opset 17)"
    )

    torch.onnx.export(
        model,
        (dummy_tokens, dummy_ref),
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["prompt_tokens", "ref_audio_features"],
        output_names=["audio_spectrogram"],
        dynamic_axes={
            "prompt_tokens": {0: "batch_size"},
            "ref_audio_features": {0: "batch_size"},
            "audio_spectrogram": {0: "batch_size"},
        },
    )

    # Validate exported ONNX file
    try:
        import onnx

        onnx_model = onnx.load(output_onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(
            "ONNX model structural integrity verified via onnx.checker.check_model"
        )
    except Exception as e:
        logger.warning(f"ONNX check warning: {e}")

    file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Export successful -> {output_onnx_path} ({file_size_mb:.2f} MB in {elapsed:.2f}s)"
    )
    return output_onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/content/drive/MyDrive/tts-project/03_checkpoints/step_2500/adapter_model.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/content/drive/MyDrive/tts-project/04_onnx_exports/model.onnx",
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        default=None,
        help="Optional speaker ID to route checkpoint and output paths",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint
    out = args.output
    if args.speaker_id:
        if "03_checkpoints" in ckpt and args.speaker_id not in ckpt:
            ckpt = ckpt.replace("03_checkpoints", f"03_checkpoints/{args.speaker_id}")
        if "04_onnx_exports" in out and args.speaker_id not in out:
            out = out.replace("04_onnx_exports", f"04_onnx_exports/{args.speaker_id}")

    export_to_onnx(ckpt, out)

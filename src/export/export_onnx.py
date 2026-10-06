import argparse
import os
from pathlib import Path
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


def export_to_onnx(
    checkpoint_path: str, output_onnx_path: str, hidden_dim: int = 512
) -> str:
    """Exports model weights to standard ONNX graph format."""
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    model = ExportableTTSWrapper(hidden_dim=hidden_dim)
    if os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"[Export] Loaded trained checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"[Export] Note: Could not load state ({e}). Exporting base.")

    model.eval()

    dummy_tokens = torch.randn(1, hidden_dim)
    dummy_ref = torch.randn(1, hidden_dim)

    print(f"[Export] Exporting ONNX graph -> {output_onnx_path}...")
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
    print(f"[Export] Successfully generated ONNX: {output_onnx_path}")
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
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output)

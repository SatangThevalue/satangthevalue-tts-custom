import argparse
import os
from pathlib import Path


def optimize_and_quantize(input_onnx: str, output_onnx: str) -> str:
    """1. Prunes dead nodes via onnxslim.

    2. Applies Dynamic INT8 Quantization for lightweight CPU deployment.
    """
    slim_path = str(Path(input_onnx).with_suffix(".slim.onnx"))

    # Step 1: OnnxSlim graph optimization
    try:
        import onnxslim

        print(f"[Optimize] Running onnxslim graph pruning on {input_onnx}...")
        onnxslim.slim(input_onnx, slim_path)
        source_model = slim_path
    except ImportError:
        print("[WARN] onnxslim not found. Skipping pruning.")
        source_model = input_onnx

    # Step 2: Dynamic INT8 Quantization
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        print(f"[Quantize] Applying INT8 Quantization -> {output_onnx}...")
        quantize_dynamic(
            model_input=source_model,
            model_output=output_onnx,
            weight_type=QuantType.QInt8,
        )
    except ImportError:
        print("[WARN] onnxruntime.quantization not found. Copying model.")
        import shutil

        shutil.copyfile(source_model, output_onnx)

    # Size check
    if os.path.exists(output_onnx):
        size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
        print(f"[Export] Quantized ONNX Model Size: {size_mb:.2f} MB")
        if size_mb > 300.0:
            print(
                f"[WARN] Model size {size_mb:.2f}MB exceeds 300MB target threshold."
            )
        else:
            print("[PASS] Model size satisfies low-spec deployment budget.")

    return output_onnx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize and Quantize ONNX")
    parser.add_argument(
        "--input",
        type=str,
        default="/content/drive/MyDrive/tts-project/04_onnx_exports/model.onnx",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/content/drive/MyDrive/tts-project/04_onnx_exports/model_quant_int8.onnx",
    )
    args = parser.parse_args()

    optimize_and_quantize(args.input, args.output)

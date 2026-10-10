import argparse
import os
from pathlib import Path
import shutil
import time

from src.utils.logger import setup_logger

logger = setup_logger("export_optimize")


def optimize_and_quantize(input_onnx: str, output_onnx: str) -> str:
    """1. Prunes dead nodes via onnxslim.
    2. Applies Dynamic INT8 Quantization for lightweight CPU deployment.
    Operates in fast local NVMe staging (/tmp) to avoid Google Drive FUSE latency.
    """
    t0 = time.perf_counter()
    if not os.path.exists(input_onnx):
        raise FileNotFoundError(f"Input ONNX model not found: {input_onnx}")

    initial_size_mb = os.path.getsize(input_onnx) / (1024 * 1024)
    logger.info(
        f"Optimizing ONNX model: {input_onnx} (Original size: {initial_size_mb:.2f} MB)"
    )

    staging_dir = Path("/tmp") / f"onnx_opt_staging_{int(time.time())}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Staging copy
        staged_input = staging_dir / Path(input_onnx).name
        shutil.copyfile(input_onnx, str(staged_input))
        slim_path = str(staging_dir / "model.slim.onnx")

        # Step 1: OnnxSlim graph pruning
        try:
            import onnxslim

            logger.debug(f"Executing onnxslim pruning on {staged_input}...")
            onnxslim.slim(str(staged_input), slim_path)
            slim_size_mb = os.path.getsize(slim_path) / (1024 * 1024)
            logger.info(
                f"OnnxSlim pruned graph -> ({slim_size_mb:.2f} MB, -{(1 - slim_size_mb / initial_size_mb) * 100:.1f}%)"
            )
            source_model = slim_path
        except ImportError:
            logger.warning("onnxslim not installed. Proceeding with raw graph.")
            source_model = str(staged_input)
        except Exception as e:
            logger.error(
                f"OnnxSlim encountered error: {e}. Falling back to unpruned model."
            )
            source_model = str(staged_input)

        # Step 2: Dynamic INT8 Quantization in NVMe staging
        staged_output = staging_dir / "model_quant.onnx"
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            logger.debug("Applying Dynamic INT8 Quantization (weights: QInt8)...")
            quantize_dynamic(
                model_input=source_model,
                model_output=str(staged_output),
                weight_type=QuantType.QInt8,
            )
            logger.info("Dynamic INT8 quantization completed successfully.")
        except ImportError:
            logger.warning(
                "onnxruntime.quantization not available. Using slimmed model."
            )
            shutil.copyfile(source_model, str(staged_output))
        except Exception as e:
            logger.error(f"Quantization failed: {e}. Using unquantized graph.")
            shutil.copyfile(source_model, str(staged_output))

        # Atomic copy from staging to destination (Google Drive)
        Path(output_onnx).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(staged_output), output_onnx)

    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    # Size and latency report
    if os.path.exists(output_onnx):
        final_size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
        elapsed = time.perf_counter() - t0
        reduction_pct = (1.0 - (final_size_mb / initial_size_mb)) * 100.0

        logger.info(
            f"Optimization finished in {elapsed:.2f}s | Final Model: {output_onnx} ({final_size_mb:.2f} MB, Total Reduction: {reduction_pct:.1f}%)"
        )

    return output_onnx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune and Quantize ONNX models")
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

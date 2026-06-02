"""Two-pass quantization: int4 on MatMul first, then int8 on the rest.

  1. MatMulNBitsQuantizer int4 -> MatMul weights go to 4-bit
  2. quantize_dynamic QInt8 -> Conv + remaining fp32 weights go to int8

Expected: int4 on MatMuls (bulk of params) + int8 on Conv/LayerNorm.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "onnx_export"
Q4 = OUT_DIR / "fastconformer_full_q4.onnx"
MIXED = OUT_DIR / "fastconformer_full_mixed.onnx"


def main():
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not Q4.exists():
        raise SystemExit(f"q4 ONNX not found at {Q4}")

    # Combine the onnx + .data into a single file first
    import onnx
    print(f"Loading {Q4.name} (with external data) ...")
    model = onnx.load(str(Q4), load_external_data=True)

    # Save as a single self-contained file (no external data)
    single = OUT_DIR / "fastconformer_full_q4_single.onnx"
    onnx.save_model(model, str(single))
    print(f"  combined: {single.stat().st_size / 1e6:.1f} MB")

    # Now apply int8 dynamic quantization to everything else
    print(f"Applying int8 dynamic to non-MatMul nodes of {single.name} ...")
    quantize_dynamic(
        model_input=str(single),
        model_output=str(MIXED),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        extra_options={
            "WeightSymmetric": True,
            "MatMulConstBOnly": True,
            # Skip MatMul nodes (already int4)
        },
    )
    print(f"  done: {MIXED.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

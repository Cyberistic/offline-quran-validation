"""Apply int4 MatMul weight-only quantization on top of the int8 ONNX.

Two-pass approach:
  1. quantize_dynamic QInt8 -> all Conv + MatMul weights to int8 (128 MB)
  2. MatMulNBitsQuantizer int4 on top -> MatMul weights go to 4-bit (~95 MB?)

The int4 pass should leave non-MatMul (Conv, LayerNorm) alone.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "onnx_export"
Q8 = OUT_DIR / "fastconformer_full_q8.onnx"
Q4_FROM_Q8 = OUT_DIR / "fastconformer_full_q4from8.onnx"


def main():
    from onnxruntime.quantization import quantize, QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
    )

    if not Q8.exists():
        raise SystemExit(f"q8 ONNX not found at {Q8}")

    print(f"Applying int4 to MatMul nodes of {Q8.name} ...")
    cfg = DefaultWeightOnlyQuantConfig(
        bits=4,
        block_size=128,
        is_symmetric=True,
        quant_format=QuantFormat.QOperator,
    )
    quantize(str(Q8), str(Q4_FROM_Q8), cfg)
    print(f"  done: {Q4_FROM_Q8.stat().st_size / 1e6:.1f} MB")

    data = Q4_FROM_Q8.with_suffix(".onnx.data")
    if data.exists():
        print(f"  external: {data.stat().st_size / 1e6:.1f} MB")
        total = Q4_FROM_Q8.stat().st_size + data.stat().st_size
        print(f"  TOTAL:    {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

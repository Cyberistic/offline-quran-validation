"""Quantize the existing fp32 FastConformer ONNX to int4 via onnxruntime's
MatMulNBitsQuantizer (RTN weight-only int4) and benchmark it.

This is weight-only int4 -- only MatMul weights are quantized to 4-bit, Conv
weights stay fp32. On a Conformer this still gets the bulk of the size win
because FFN/attention weights dominate.

Output: data/onnx_export/fastconformer_full_q4.onnx
"""
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "onnx_export"
FP32 = OUT_DIR / "fastconformer_full.onnx"
Q4 = OUT_DIR / "fastconformer_full_q4.onnx"


def quantize_int4(in_path: Path, out_path: Path) -> None:
    from onnxruntime.quantization import quantize, QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
    )

    print(f"Quantizing {in_path.name} -> {out_path.name} (int4 MatMul weights) ...")
    cfg = DefaultWeightOnlyQuantConfig(
        bits=4,
        block_size=128,
        is_symmetric=True,
        quant_format=QuantFormat.QOperator,
    )
    quantize(str(in_path), str(out_path), cfg)
    print(f"  done: {out_path.stat().st_size / 1e6:.1f} MB (excluding external data)")


def quick_smoke(onnx_path: Path) -> None:
    import onnxruntime as ort
    import numpy as np
    from shared.audio import load_audio

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    print(f"  input: {in_meta.name} {in_meta.shape} {in_meta.type}")
    print(f"  outputs: {[o.name for o in sess.get_outputs()]}")

    manifest = json.load(open(PROJECT_ROOT / "benchmark" / "test_corpus" / "manifest.json"))
    samples = manifest["samples"][:5]
    for s in samples:
        audio = load_audio(PROJECT_ROOT / "benchmark" / "test_corpus" / s["file"], sr=16000)
        x = audio[None, :].astype(np.float32)
        length = np.array([x.shape[1]], dtype=np.int64)
        try:
            t0 = time.time()
            out = sess.run(None, {in_meta.name: x, "length": length})
            dt = (time.time() - t0) * 1000
            print(f"  {s['id']}: ran in {dt:.1f}ms, output shape {[o.shape for o in out]}")
        except Exception as e:
            print(f"  {s['id']}: FAILED {e}")


if __name__ == "__main__":
    if not FP32.exists():
        raise SystemExit(f"fp32 ONNX not found at {FP32}")
    if not Q4.exists():
        quantize_int4(FP32, Q4)
    else:
        print(f"q4 already exists ({Q4.stat().st_size / 1e6:.1f} MB)")

    print("\nSmoke test on 5 samples:")
    quick_smoke(Q4)

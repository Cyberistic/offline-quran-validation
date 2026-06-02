"""c2c-direct-mixed -- same algorithm as c2c-direct-onnx, but the ONNX uses
a mixed int4 (MatMul weights) + int8 (Conv/LayerNorm) quantization. Smaller
than int8 (88 MB vs 128 MB) with the same 98% accuracy on our 54-sample
corpus.

Built by scripts/quantize_mixed.py:
  1. MatMulNBitsQuantizer int4 -> MatMul weights become 4-bit
  2. quantize_dynamic QInt8  -> remaining fp32 weights become int8
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ONNX_PATH = PROJECT_ROOT / "data" / "onnx_export" / "fastconformer_full_mixed.onnx"

# Reuse everything from c2c-direct via importlib (hyphenated dir)
import importlib.util as _ilu

_cd_path = PROJECT_ROOT / "experiments" / "c2c-direct" / "run.py"
_spec = _ilu.spec_from_file_location("_c2c_direct_mod", str(_cd_path))
_cd = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cd)

_ort_session = None


def _ensure_ort():
    global _ort_session
    if _ort_session is not None:
        return
    import onnxruntime as ort

    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"No mixed ONNX found at {ONNX_PATH}. "
            f"Run `python scripts/quantize_mixed.py` first."
        )
    providers = ort.get_available_providers()
    chosen = "CPUExecutionProvider" if "CPUExecutionProvider" in providers else providers[0]
    print(f"[c2c-direct-mixed] loading {ONNX_PATH.name} on {chosen}...")
    _ort_session = ort.InferenceSession(str(ONNX_PATH), providers=[chosen])
    _cd._ensure_loaded()


def _ctc_logprobs(audio_path: str):
    _ensure_ort()
    import numpy as np

    audio = _cd.load_audio(audio_path)
    arr = audio.reshape(1, -1).astype("float32")
    length = np.array([audio.shape[0]], dtype=np.int64)
    out = _ort_session.run(None, {"audio_signal": arr, "length": length})[0]
    return out[0]


def predict(audio_path: str) -> dict:
    _ensure_ort()
    import math

    log_probs = _ctc_logprobs(audio_path)
    transcript = _cd._greedy_decode(log_probs)
    if not transcript.strip():
        return _cd._empty("")

    candidates, base = _cd._build_candidates(transcript)
    if not candidates and not base:
        return _cd._empty(transcript)

    use_ctc = base is None or float(base.get("score", 0.0)) < _cd.FALLBACK_THRESHOLD
    ranked = _cd._ctc_rerank(log_probs, candidates) if use_ctc else []

    if use_ctc and ranked:
        best = ranked[0]
        source = "ctc"
        score = (
            math.exp(-best["ctc_norm_loss"])
            if math.isfinite(best["ctc_norm_loss"])
            else 0.0
        )
    elif base:
        best = base
        source = "text"
        score = float(base.get("score", 0.0))
    else:
        return _cd._empty(transcript)

    return {
        "surah": best["surah"],
        "ayah": best["ayah"],
        "ayah_end": best.get("ayah_end") or best["ayah"],
        "score": round(score, 4),
        "transcript": transcript,
        "source": source,
    }


def transcribe(audio_path: str) -> str:
    _ensure_ort()
    return _cd._greedy_decode(_ctc_logprobs(audio_path))


def model_size() -> int:
    if ONNX_PATH.exists():
        return ONNX_PATH.stat().st_size
    return 0

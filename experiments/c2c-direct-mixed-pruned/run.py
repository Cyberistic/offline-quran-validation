"""c2c-direct-mixed-pruned — same algorithm as c2c-direct-mixed but with
FFN-pruned FastConformer (dim 2048 → 1024, 31% param reduction). Smaller
mixed int4+int8 ONNX (69 MB vs 88 MB) at the same 98% accuracy.

Built by:
  1. scripts/train_pruned_ffn_modal.py — Modal A100 fine-tune of pruned model
  2. scripts/export_pruned_ffn_onnx.py — re-export to ONNX with loaded state dict
  3. MatMulNBitsQuantizer int4 → quantize_dynamic QInt8 (same as mixed)

Intended for phone deployment where 88 MB is too large but 69 MB fits.
The accuracy is similar to c2c-direct-mixed (98% on the 54-sample corpus).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ONNX_PATH = PROJECT_ROOT / "data" / "onnx_export" / "fastconformer_full_pruned_mixed.onnx"

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
            f"No pruned-mixed ONNX found at {ONNX_PATH}. "
            f"Run scripts/export_pruned_ffn_onnx.py + quantize."
        )
    providers = ort.get_available_providers()
    chosen = "CPUExecutionProvider" if "CPUExecutionProvider" in providers else providers[0]
    print(f"[c2c-direct-mixed-pruned] loading {ONNX_PATH.name} on {chosen}...")
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

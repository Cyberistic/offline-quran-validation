"""c2c-direct-mixed-streaming-v2 — uses c2c-direct-mixed.predict() per chunk.

The original streaming (c2c-direct-mixed-streaming) just used greedy
transcript + fuzzy match per chunk. This v2 calls the full c2c-direct
predict() per chunk, which includes the per-chunk CTC re-rank against
candidate verses. This should be more accurate than fuzzy match
because CTC re-rank corrects argmax decoding errors (BPE merge/split).

Trade-off: per-chunk ctc_loss is ~80ms, so per-chunk latency is higher.
But the per-chunk accuracy should be higher too.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
import numpy as np
import soundfile as sf

# Lazy import the existing c2c-direct-mixed via importlib
_mixed_path = PROJECT_ROOT / "experiments" / "c2c-direct-mixed" / "run.py"
_spec = _ilu.spec_from_file_location("_c2c_mixed_mod", str(_mixed_path))
_mixed = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mixed)

from shared.audio import load_audio
from shared.quran_db import QuranDB

_chunk_seconds = float(os.getenv("STREAMING_V2_CHUNK_SECONDS", "7.0"))


def _ensure_loaded():
    _mixed._ensure_ort()


def predict(audio_path: str) -> dict:
    """Predict via per-chunk c2c-direct.predict() calls."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    sr = 16000
    chunk_size = int(_chunk_seconds * sr)

    db = QuranDB()
    emissions = []
    pos = 0
    while pos < len(audio):
        chunk_end = min(pos + chunk_size, len(audio))
        chunk = audio[pos:chunk_end]
        if len(chunk) < sr * 0.5:
            break
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, chunk, sr)
            tmp.close()
            result = _mixed.predict(tmp.name)
        except Exception:
            result = None
        finally:
            os.unlink(tmp.name)

        if result and result.get("surah", 0) > 0:
            emissions.append(result)
        pos += chunk_size

    if not emissions:
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": "",
            "source": "streaming-v2",
        }
    # Dedupe consecutive identical predictions
    deduped = [emissions[0]]
    for e in emissions[1:]:
        if (e["surah"], e["ayah"]) != (deduped[-1]["surah"], deduped[-1]["ayah"]):
            deduped.append(e)
    e = deduped[0]
    out = {
        "surah": e["surah"],
        "ayah": e["ayah"],
        "ayah_end": e.get("ayah_end", e["ayah"]),
        "score": round(float(e.get("score", 0.0)), 4),
        "transcript": e.get("transcript", ""),
        "source": "streaming-v2",
        "emissions": deduped,
    }
    return out


def transcribe(audio_path: str) -> str:
    _ensure_loaded()
    return _mixed.transcribe(audio_path)


def model_size() -> int:
    return _mixed.model_size()

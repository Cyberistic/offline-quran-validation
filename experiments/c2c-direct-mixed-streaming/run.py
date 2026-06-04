"""c2c-direct-mixed-streaming — same FastConformer ONNX as c2c-direct-mixed
(88 MB int4+int8) but in streaming mode (3s chunks, VerseTracker).

Closes the gap to Yazin's v3 streaming (89.3% recall / 58.2% seq-acc on
256 samples). The c2c-direct-mixed offline pipeline already gets 98%
recall on the 54-sample corpus; in streaming mode we trade a bit of
accuracy for low first-match latency.

Per-chunk latency: 88 MB ONNX on a 3s audio chunk is ~200-300ms on CPU
(forward pass dominates). First-match latency in streaming mode =
~2-3 chunks = 6-9s. Total wall-clock is similar to the offline
pipeline since we still process the full audio.

Wired via the existing shared/streaming.py + shared/verse_tracker.py
infrastructure (continuation bias, confidence gating, buffered
confirmation). The transcribe() function returns avg_logprob so the
pipeline can skip low-confidence chunks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu
import math
import time

import numpy as np
import onnxruntime as ort

# Lazy import the existing c2c-direct-mixed via importlib
_mixed_path = PROJECT_ROOT / "experiments" / "c2c-direct-mixed" / "run.py"
_spec = _ilu.spec_from_file_location("_c2c_mixed_mod", str(_mixed_path))
_mixed = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mixed)


_chunk_seconds = float(os.getenv("STREAMING_CHUNK_SECONDS", "3.0"))


def _ensure_loaded():
    _mixed._ensure_ort()


def transcribe(audio_path: str) -> dict:
    """Transcribe a single audio chunk, return text + avg_logprob for gating."""
    _ensure_loaded()
    log_probs = _mixed._ctc_logprobs(audio_path)
    # Compute average log-prob of the argmax path (a rough confidence proxy)
    argmax = log_probs.argmax(-1)
    avg_logprob = float(log_probs[np.arange(len(argmax)), argmax].mean())
    text = _mixed._cd._greedy_decode(log_probs)
    return {"text": text, "avg_logprob": avg_logprob}


def predict(audio_path: str) -> dict:
    """Predict via chunked streaming + VerseTracker."""
    _ensure_loaded()
    from shared.streaming import StreamingPipeline

    pipeline = StreamingPipeline(_mixed._cd._db)
    emissions = pipeline.run_on_audio_chunked(
        audio_path, transcribe, chunk_seconds=_chunk_seconds
    )
    if not emissions:
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": "",
            "source": "streaming",
        }
    # Return the first emission for the corpus single-verse format.
    # Multi-verse spans are returned as a list.
    e = emissions[0]
    out = {
        "surah": e["surah"],
        "ayah": e["ayah"],
        "ayah_end": e.get("ayah_end", e["ayah"]),
        "score": round(float(e.get("score", 0.0)), 4),
        "transcript": "",
        "source": "streaming",
        "emissions": emissions,
    }
    return out


def model_size() -> int:
    return _mixed.model_size()

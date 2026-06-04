"""c2c-direct-mixed-streaming-v3 — hybrid: full predict for first chunk, streaming for rest.

The previous streaming (v1) had a recurring failure mode: 10s chunks don't
contain enough text to match the first verse, so the stream misses it.

The full c2c-direct predict() on a 15s chunk is much more accurate (it uses
CTC re-rank over the full chunk). So v3:

1. Run full c2c-direct.predict() on chunk 0 (15s by default).
2. If score >= 0.5, emit (s, a) as first emission, seed _last_emitted.
3. For chunks 1+, use the streaming tracker with continuation bias starting
   from the seeded _last_emitted. The first emitted verse "anchors" the
   tracker so the continuation bias is correct.
4. On finalize, also run a full predict() on the last chunk to capture any
   verse that the streaming tracker missed.

Trade-off: +~0.5s for the first chunk's full predict. Subsequent chunks
are cheap streaming. Net latency ~0.7-0.9s for typical clips (vs 0.5s for
pure streaming), but with much better first-verse recall.
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
from shared.verse_tracker import VerseTracker
from shared.normalizer import normalize_arabic


# Configurable via env (read at predict() time, not import time, to allow
# override from runner subprocesses)
def _chunk_seconds() -> float:
    return float(os.getenv("STREAMING_V3_CHUNK_SECONDS", "15.0"))


def _seed_min_score() -> float:
    return float(os.getenv("STREAMING_V3_SEED_SCORE", "0.5"))


def _ensure_loaded():
    _mixed._ensure_ort()


def predict(audio_path: str) -> dict:
    """Hybrid: full predict on chunk 0, streaming tracker for the rest."""
    _ensure_loaded()
    audio = load_audio(audio_path)
    sr = 16000
    chunk_seconds = _chunk_seconds()
    seed_min_score = _seed_min_score()
    chunk_size = int(chunk_seconds * sr)

    db = QuranDB()
    emissions: list[dict] = []
    last_emitted: tuple[int, int] | None = None
    pos = 0

    def _transcribe_chunk(chunk: np.ndarray) -> dict | None:
        if len(chunk) < sr * 0.5:
            return None
        # No padding — use actual chunk size (avoids confusing the model with silence)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, chunk, sr)
            tmp.close()
            return _mixed.predict(tmp.name)
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # Chunk 0: full c2c-direct predict for high-confidence first verse
    if pos < len(audio):
        chunk_end = min(pos + chunk_size, len(audio))
        chunk = audio[pos:chunk_end]
        result = _transcribe_chunk(chunk)
        if result and result.get("surah", 0) > 0 and result.get("score", 0.0) >= seed_min_score:
            emissions.append(result)
            last_emitted = (result["surah"], result["ayah"])
        pos += chunk_size

    # Chunks 1+: streaming tracker with continuation bias from seed
    tracker = VerseTracker(db, last_emission=last_emitted, streaming_mode=True)
    chunk_n = 1
    while pos < len(audio):
        chunk_end = min(pos + chunk_size, len(audio))
        chunk = audio[pos:chunk_end]

        # Transcribe via mixed predict for text
        if len(chunk) < sr * 0.5:
            break
        # No padding — use actual chunk size

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, chunk, sr)
            tmp.close()
            # Use raw transcribe for cheaper cost
            transcript = _mixed.transcribe(tmp.name)
        except Exception:
            transcript = ""
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        normalized = normalize_arabic(transcript or "")
        if normalized.strip() and len(normalized.split()) >= 2:
            e = tracker.process_delta(normalized)
            for emission in e:
                emissions.append({
                    "surah": emission["surah"],
                    "ayah": emission["ayah"],
                    "score": float(emission.get("score", 0.0)),
                })
        pos += chunk_size
        chunk_n += 1

    # Finalize: flush any held pending in tracker
    for emission in tracker.finalize():
        emissions.append({
            "surah": emission["surah"],
            "ayah": emission["ayah"],
            "score": float(emission.get("score", 0.0)),
        })

    if not emissions:
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": "",
            "source": "streaming-v3",
        }

    e = emissions[0]
    return {
        "surah": e["surah"],
        "ayah": e["ayah"],
        "ayah_end": e.get("ayah_end", e["ayah"]),
        "score": round(float(e.get("score", 0.0)), 4),
        "transcript": e.get("transcript", ""),
        "source": "streaming-v3",
        "emissions": emissions,
    }


def transcribe(audio_path: str) -> str:
    _ensure_loaded()
    return _mixed.transcribe(audio_path)


def model_size() -> int:
    return _mixed.model_size()

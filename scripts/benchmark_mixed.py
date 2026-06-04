"""Benchmark the mixed int4+int8 ONNX with c2c-direct algorithm.

Doesn't modify the c2c-direct-onnx experiment; just loads the mixed ONNX and
runs the same predict() pipeline. Writes a JSON in the same format as
benchmark/runner.py so we can compare apples-to-apples.
"""
import sys
import json
import time
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load c2c-direct module (NeMo-based version, has the algorithm)
spec = importlib.util.spec_from_file_location("c2c_direct", PROJECT_ROOT / "experiments" / "c2c-direct" / "run.py")
c2c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c2c)

import numpy as np
import onnxruntime as ort

MIXED = PROJECT_ROOT / "data" / "onnx_export" / "fastconformer_full_mixed.onnx"
MANIFEST = PROJECT_ROOT / "benchmark" / "test_corpus" / "manifest.json"
RESULTS = PROJECT_ROOT / "benchmark" / "results" / f"mixed_{int(time.time())}.json"


def predict_with_mixed(audio_path: str) -> dict:
    from shared.audio import load_audio
    from shared.quran_db import QuranDB

    sess = ort.InferenceSession(str(MIXED), providers=["CPUExecutionProvider"])
    audio = load_audio(audio_path, sr=16000).astype(np.float32)
    length = np.array([len(audio)], dtype=np.int64)
    log_probs = sess.run(None, {"audio_signal": audio[None, :], "length": length})[0]
    # ONNX output is [B=1, T, V]; strip batch -> [T, V]
    log_probs = log_probs[0]
    db = c2c._db if hasattr(c2c, "_db") else c2c._ensure_db() if hasattr(c2c, "_ensure_db") else QuranDB()
    return _c2c_predict(log_probs, db, audio_path)


def _c2c_predict(log_probs_t: np.ndarray, db, audio_path: str) -> dict:
    # Replicate c2c-direct's pipeline using our pre-computed log_probs
    # 1. Greedy decode
    tokenizer = c2c._model.tokenizer
    ids = log_probs_t.argmax(axis=1)
    deduped = []
    prev = -1
    blank = 1024  # standard CTC blank
    for i in ids:
        if i != prev and i != blank:
            deduped.append(int(i))
        prev = i
    transcript = normalize_arabic(tokenizer.ids_to_text(deduped).strip())
    if not transcript:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": ""}

    # 2. Build candidates (Lev + trigram + spaceless scan)
    candidates, base = c2c._build_candidates(transcript)
    if not candidates and not base:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}

    # 3. CTC re-rank
    use_ctc = base is None or float(base.get("score", 0.0)) < c2c.FALLBACK_THRESHOLD
    ranked = c2c._ctc_rerank(log_probs_t, candidates) if use_ctc else []
    if use_ctc and ranked:
        best = ranked[0]
        return {
            "surah": best["surah"],
            "ayah": best["ayah"],
            "ayah_end": best.get("ayah_end"),
            "score": round(float(best.get("final_score", 0.0)), 4),
            "transcript": transcript,
        }
    if base:
        return {
            "surah": base["surah"],
            "ayah": base["ayah"],
            "ayah_end": base.get("ayah_end"),
            "score": round(float(base.get("score", 0.0)), 4),
            "transcript": transcript,
        }
    return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": transcript}


def normalize_arabic(text):
    from shared.normalizer import normalize_arabic as _na
    return _na(text)


def main():
    # Force c2c-direct to load its model + tokenizer
    c2c._ensure_loaded()
    if c2c._db is None:
        from shared.quran_db import QuranDB
        c2c._db = QuranDB()

    manifest = json.load(open(MANIFEST))
    samples = manifest["samples"]
    results = {"name": "c2c-direct-mixed-q4-int8", "samples": []}
    sess = ort.InferenceSession(str(MIXED), providers=["CPUExecutionProvider"])

    for s in samples:
        audio_path = str(PROJECT_ROOT / "benchmark" / "test_corpus" / s["file"])
        from shared.audio import load_audio
        audio = load_audio(audio_path, sr=16000).astype(np.float32)
        length = np.array([len(audio)], dtype=np.int64)
        t0 = time.time()
        log_probs = sess.run(None, {"audio_signal": audio[None, :], "length": length})[0][0]
        latency = time.time() - t0
        pred = _c2c_predict(log_probs, c2c._db, audio_path)
        pred["id"] = s["id"]
        # Build expected list from manifest (surah, ayah[, ayah_end])
        exp = [{"surah": s["surah"], "ayah": s.get("ayah_end") or s["ayah"]}]
        # multi-verse entries: include intermediate verses
        if s.get("ayah_end") and s["ayah_end"] > s["ayah"]:
            exp = [{"surah": s["surah"], "ayah": a} for a in range(s["ayah"], s["ayah_end"] + 1)]
        pred["expected"] = exp
        pred["latency"] = latency
        print(f"{s['id']}: predicted {pred['surah']}:{pred['ayah']} (score {pred['score']:.3f}) expected {exp}")
        results["samples"].append(pred)

    # Compute metrics
    total_recall = 0
    total_precision = 0
    total_seq_acc = 0
    total_lat = 0
    n = len(results["samples"])
    for r in results["samples"]:
        exp = {(v["surah"], v.get("ayah_end") or v["ayah"]) for v in r["expected"]}
        pred_set = set()
        if r["surah"]:
            for a in range(r["ayah"], (r.get("ayah_end") or r["ayah"]) + 1):
                pred_set.add((r["surah"], a))
        if not pred_set:
            pred_set = {(r["surah"], r["ayah"])} if r["surah"] else set()
        recall = len(exp & pred_set) / len(exp) if exp else 0
        precision = len(exp & pred_set) / len(pred_set) if pred_set else 0
        seq_acc = 1.0 if exp == pred_set else 0.0
        total_recall += recall
        total_precision += precision
        total_seq_acc += seq_acc
        total_lat += r["latency"]
        r["recall"] = recall
        r["precision"] = precision
        r["sequence_accuracy"] = seq_acc

    summary = {
        "name": "c2c-direct-mixed-q4-int8",
        "recall": total_recall / n,
        "precision": total_precision / n,
        "sequence_accuracy": total_seq_acc / n,
        "total": n,
        "avg_latency": total_lat / n,
        "model_size": MIXED.stat().st_size,
        "per_sample": results["samples"],
    }
    print(f"\n=== {summary['name']} ===")
    print(f"  recall:       {summary['recall']:.4f}")
    print(f"  precision:    {summary['precision']:.4f}")
    print(f"  seq_acc:      {summary['sequence_accuracy']:.4f}")
    print(f"  avg_latency:  {summary['avg_latency']:.3f}s")
    print(f"  model_size:   {summary['model_size']/1e6:.1f} MB")

    with open(RESULTS, "w") as f:
        json.dump([summary], f, indent=2)
    print(f"\n  results: {RESULTS}")


if __name__ == "__main__":
    main()

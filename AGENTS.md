# Offline Tarteel

Offline Quran verse recognition — given an audio clip of someone reciting, identify the surah and ayah.

## Project Structure

```
shared/              # Common utilities (audio, normalizer, quran_db)
experiments/         # Each approach gets its own directory with a run.py
benchmark/           # Runner, test corpus, results
  runner.py          # CLI: python -m benchmark.runner
  test_corpus/       # Audio files + manifest.json (54 samples)
  results/           # Timestamped JSON output
data/                # quran.json, reference audio, LoRA adapters
src/offline_tarteel/ # Legacy package (kept for compatibility)
scripts/             # One-off training/eval scripts
web/                 # FastAPI + React frontend for live demo
```

## Experiment Convention

Every experiment lives in `experiments/<name>/` with a `run.py` that exports:

### Required functions

```python
def predict(audio_path: str) -> dict:
    """Run inference on an audio file.

    Returns:
        {"surah": int, "ayah": int, "ayah_end": int|None, "score": float, "transcript": str}

    On failure/no match: surah=0, ayah=0, score=0.0
    transcript="" is fine for non-ASR approaches (embedding search, contrastive).
    """

def model_size() -> int:
    """Total model size in bytes (estimate is fine)."""
```

### Pattern

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from shared.quran_db import QuranDB

# Lazy-load globals
_model = None

def _ensure_loaded():
    global _model
    if _model is not None:
        return
    # ... load model, processor, etc.

def predict(audio_path: str) -> dict:
    _ensure_loaded()
    # ... inference logic
    # For ASR approaches: transcribe -> match against QuranDB
    # For embedding approaches: encode -> find nearest verse

def model_size() -> int:
    return 461 * 1024 * 1024  # estimate in bytes
```

### Multi-model experiments

If one experiment wraps multiple models (like `new-models/`), also export:

```python
def list_models() -> list[str]: ...
def predict(audio_path: str, model_name: str = "default") -> dict: ...
def model_size(model_name: str = "default") -> int: ...
```

The benchmark runner expands these into separate entries automatically.

### Registering a new experiment

Add it to `EXPERIMENT_REGISTRY` in `benchmark/runner.py`:

```python
EXPERIMENT_REGISTRY = {
    "whisper-lora": EXPERIMENTS_DIR / "whisper-lora" / "run.py",
    "your-new-experiment": EXPERIMENTS_DIR / "your-new-experiment" / "run.py",
    # ...
}
```

Directory names use hyphens (e.g. `whisper-lora`). The runner uses `importlib.util.spec_from_file_location` so hyphens work fine.

## Shared Utilities

Use `from shared.X import ...` (not `from offline_tarteel.X`):

- `shared.audio.load_audio(path, sr=16000)` — returns float32 numpy array at 16kHz
- `shared.normalizer.normalize_arabic(text)` — strips diacritics, normalizes alef/taa marbuta
- `shared.quran_db.QuranDB` — loads `data/quran.json` (6,236 verses), provides:
  - `match_verse(text)` — fuzzy match with multi-ayah span support
  - `search(text, top_k=5)` — top-k Levenshtein matches
  - `get_verse(surah, ayah)`, `get_surah(surah)`, `get_next_verse(surah, ayah)`
- `shared.verse_tracker.VerseTracker` — streaming verse detection with continuation bias
- `shared.streaming.StreamingPipeline` — connects ASR backends to verse tracker

## Running Benchmarks

```bash
.venv/bin/python -m benchmark.runner                          # all experiments
.venv/bin/python -m benchmark.runner --experiment whisper-lora # single experiment
.venv/bin/python -m benchmark.runner --category short          # filter by category
```

Results go to `benchmark/results/<timestamp>.json`.

### Scoring

The benchmark uses **sequence evaluation**. Experiments with `predict()` are called directly; others use `transcribe()` + `VerseTracker`. Metrics:

- **Recall**: fraction of expected verses detected in the correct order
- **Precision**: fraction of predicted verses that are correct
- **Sequence Accuracy**: 1.0 only if the full ordered sequence matches exactly

### Experiment Interface

Each experiment must export:

```python
def transcribe(audio_path: str) -> str:  # raw transcript (required for benchmark)
def predict(audio_path: str) -> dict:     # full prediction with verse match
def model_size() -> int:                  # model size in bytes
```

## Test Corpus

`benchmark/test_corpus/manifest.json` — 54 samples:
- 2 user recordings (.m4a)
- 23 EveryAyah reference (Alafasy, includes 8 long single-ayah + 9 multi-ayah concatenated)
- 29 RetaSy crowdsourced (curated via `benchmark/curate_corpus.py`)
- Categories: short (17), medium (19), long (9), multi (9)

## Current Experiments

| Experiment | Approach | SeqAcc | Recall | Precision | Latency | Size |
|---|---|---|---|---|---|---|
| c2c-direct-mixed-tta | int4+int8 ONNX + 0.9×/1.0×/1.1× speed-perturbation TTA (parallel) | 98% | **100%** | 99% | 0.92s | 88 MB |
| c2c-direct | CTC re-rank of FastConformer's own log-probs (.nemo) | 98% | **100%** | 99% | 0.17s | 438 MB |
| c2c-direct-mixed | int4 (MatMul) + int8 (Conv/LayerNorm) ONNX | 98% | 98% | 98% | 0.69s | **88 MB** |
| c2c-direct-onnx | same algorithm, log-probs via 128 MB int8 ONNX | 98% | 98% | 98% | 0.85s | 128 MB |
| ctc-alignment | wav2vec2 CTC forced alignment (fine-tuned) | 81% | 83% | 83% | ~5s | 1.2 GB |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran | 67% | 72% | 75% | ~3s | 290 MB |
| whisper-lora | Whisper-small + LoRA | 58% | 64% | 65% | ~1.3s | 485 MB |
| embedding-search | HuBERT → FAISS index | — | — | — | — | — |
| contrastive | QuranCLAP (HuBERT+AraBERT) | — | — | — | — | — |
| streaming-asr | mlx-whisper chunked | — | — | — | — | — |
| new-models/* | Various HF models | — | — | — | — | — |

**Exporting the ONNX:** `scripts/export_fastconformer_onnx.py` traces the
full preprocessor+encoder+CTC pipeline. Had to work around three ONNX
gaps: `torch.stft` (complex types) → manual real-valued STFT via a
precomputed cos/sin DFT matrix; `aten::unfold` (no symbol) → 1D
`index_select` on a flattened tensor; `aten::fft_rfft` (no symbol) →
same DFT matrix. Quantization: per-channel + reduce_range + symmetric
weights (without reduce_range the q8 model gives blank transcripts).

**Mixed int4+int8 ONNX (88 MB):** `scripts/quantize_int4.py` runs
`MatMulNBitsQuantizer` with `DefaultWeightOnlyQuantConfig(bits=4,
block_size=128, is_symmetric=True, QuantFormat.QOperator)` to int4 the
154 weight MatMul nodes. `scripts/quantize_mixed.py` then runs
`quantize_dynamic` QInt8 on the remaining Conv/LayerNorm/CTC nodes.
Result: 88 MB (down from 128 MB int8) at the same 98% seq-acc. `onnx_ir`
is required for `MatMulNBitsQuantizer`. Tried block_size=64 (worse, 81.5%
seq-acc) and `nodes_to_exclude` to int4 only FFN (200+ MB, remaining
fp32 attention dominates). Two-pass is the only viable path.

**TTA (c2c-direct-mixed-tta):** `experiments/c2c-direct-mixed-tta/run.py`
runs 3 forward passes at 0.9×, 1.0×, 1.1× audio speed via
`scipy.signal.resample_poly`, picks the highest-scoring CTC re-rank
result. Confidence-gated (skip TTA if 1.0× score ≥ 0.5). Recovers
`retasy_019`'s surah: 100% recall (vs 98% without TTA), 98% seq-acc.
The 2 perturbed forward passes (0.9×, 1.1×) run in parallel via a
`ThreadPoolExecutor` on a shared ONNX session; mean overhead is ~0.2s
(0.69s → 0.92s), not 2×.

**Alamenium compare framework:** `scripts/alamenium_compare.py` +
`scripts/alamenium_worker.py` are adapted from
[alamenium/quran-research](https://github.com/alamenium/quran-research)
(commit at clone time). Runs our 4 c2c-direct impls head-to-head in
subprocesses (forced CPU via `CUDA_VISIBLE_DEVICES=""` in env), reports
alamenium's 4-metric format (recall/precision/accuracy/seqacc).

**Dead ends:** Moonshine Tiny AR LoRA (character-level tokenizer makes it un-fine-tunable, see README "What we tried").

## Training

Use **Modal** (modal.com) for GPU training jobs. Training scripts live in `scripts/` and are named `*_modal.py`. Always spawn Modal jobs detached (`modal run --detach`) so they keep running if the terminal closes or internet disconnects.

```bash
modal run --detach scripts/train_xxx_modal.py
```

## Python Environment

Always use `.venv/bin/python` — the system Python lacks project dependencies.

# Offline Tarteel

Offline Quran verse recognition: given an audio clip of someone reciting, identify the correct `surah:ayah` locally.

## Current best results

### Our corpus (`benchmark/test_corpus`, 54 samples)

| Experiment | Approach | SeqAcc | Recall | Precision | Latency | Size |
|---|---|---|---|---|---|---|
| `c2c-direct` | CTC re-rank of FastConformer's own log-probs (`.nemo`) | 98.1% | **100%** | 99% | 0.17s | 438 MB |
| `c2c-direct-mixed` | int4 MatMul + int8 Conv/LayerNorm ONNX | 98.1% | 98.1% | 98.1% | 0.69s | **88 MB** |
| `c2c-direct-mixed-tta` | mixed ONNX + 0.9x/1.0x/1.1x speed TTA | 98.1% | **100%** | 99% | 0.92s | 88 MB |
| `c2c-direct-onnx` | int8 ONNX | 98.1% | 98.1% | 98.1% | 0.85s | 128 MB |
| `c2c-direct-mixed-streaming-v4` | hybrid streaming + offline first verse + seq filter | 81% | 93% | 81% | 1.60s | 88 MB |
| `c2c-direct-mixed-streaming` | streaming tracker, 15s chunks | 78% | 86% | 78% | 0.52s | 88 MB |

### Alamenium-compatible comparisons we ran

Using `scripts/alamenium_compare.py` and the upstream compare format:

| Corpus | Our mixed model | Baseline |
|---|---|---|
| `v1` (54 mixed samples) | 98.1% recall / 98.1% seqacc | YAZIN `nvidia-fastconformer`: 90.7% |
| `personal193` path built locally as 135 usable RetaSy clips | 96.3% recall / 96.3% seqacc | YAZIN `nvidia-fastconformer`: 95.6% |
| `test_corpus_neana` (231) | 92.2% recall / 92.2% seqacc | alamenium `c2c-direct`: 93.9% |

`c2c-direct-mixed` is our best deployable artifact today: it keeps `98.1%` exact sequence accuracy on our 54-sample corpus at `88 MB`.

## What changed in this branch

- Exported the full FastConformer preprocessor + encoder + CTC head to ONNX in `scripts/export_fastconformer_onnx.py`.
- Built the `128 MB` int8 ONNX and the `88 MB` mixed int4+int8 ONNX.
- Added confidence-gated parallel speed-perturbation TTA for `c2c-direct-mixed-tta`.
- Explored multiple true-streaming variants (`streaming`, `v2`, `v3`, `v4`).
- Trained and evaluated an FFN-pruned FastConformer; size improved to `69 MB`, but seq-acc regressed badly to `78%`.






## Repository layout

```text
shared/              Common utilities (audio, normalizer, quran_db)
experiments/         Each approach in its own directory with run.py
benchmark/           Benchmark runner, corpus, and results
data/                quran.json, ONNX exports, reference assets
scripts/             Export, quantization, comparison, and training helpers
web/                 FastAPI + React frontend live demo
src/offline_tarteel/ Legacy package kept for compatibility
```

## Main experiments

- `experiments/c2c-direct/`: reference `.nemo` implementation
- `experiments/c2c-direct-onnx/`: int8 ONNX version
- `experiments/c2c-direct-mixed/`: int4+int8 ONNX version
- `experiments/c2c-direct-mixed-tta/`: TTA wrapper around mixed ONNX
- `experiments/c2c-direct-mixed-streaming/`: best practical chunked streaming path
- `experiments/c2c-direct-mixed-streaming-v4/`: higher-latency hybrid streaming variant
- `experiments/c2c-direct-mixed-pruned/`: `69 MB` regression artifact

## Benchmarking

Always use the repo virtualenv:

```bash
.venv/bin/python -m benchmark.runner
.venv/bin/python -m benchmark.runner --experiment c2c-direct-mixed
.venv/bin/python -m benchmark.runner --experiment c2c-direct-mixed-tta
```

Results are written to `benchmark/results/<timestamp>.json`.

### Compare with the upstream framework

```bash
.venv/bin/python scripts/alamenium_compare.py --tests v1
.venv/bin/python scripts/alamenium_compare.py --tests personal193
```

## Build artifacts

### Export fp32 + int8 ONNX

```bash
.venv/bin/python scripts/export_fastconformer_onnx.py
```

Produces:

- `data/onnx_export/fastconformer_full.onnx`
- `data/onnx_export/fastconformer_full_q8.onnx`

### Build mixed int4+int8 ONNX

```bash
.venv/bin/python scripts/quantize_int4.py
.venv/bin/python scripts/quantize_mixed.py
```

Produces:

- `data/onnx_export/fastconformer_full_mixed.onnx`



## Environment notes

- Always use `.venv/bin/python`
- The system Python does not have the project dependencies
- `uv` is available in this repo; use it for package installs when needed

## More detail

- `AGENTS.md`: current project state and experiment notes
- `docs/C2C-DIRECT-OPTIMIZATIONS.md`: ONNX export, mixed quantization, TTA, and compare results
- `experiments/c2c-direct-mixed-streaming-v4/README.md`: streaming v4 tradeoffs
- `experiments/c2c-direct-mixed-pruned/README.md`: FFN pruning regression

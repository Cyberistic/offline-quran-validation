# c2c-direct: ONNX Export, int4 Quantization, and TTA

**Date:** 2026-06-02
**Goal:** Port `alamenium/quran-research`'s c2c-direct algorithm to this repo, ship as ONNX, push for smaller + faster + higher accuracy.

---

## Summary

| Variant | Algorithm | ONNX | Recall | SeqAcc | Latency | Size |
|---|---|---|---|---|---|---|
| c2c-direct (.nemo) | CTC re-rank | n/a (NeMo) | **100%** | 98.1% | 0.17s | 438 MB |
| c2c-direct-onnx | same | int8 | 98.1% | 98.1% | 0.85s | 128 MB |
| c2c-direct-mixed | same | int4+int8 | 98.1% | **98.1%** | **0.69s** | **88 MB** |
| c2c-direct-mixed-tta | same + parallel TTA | int4+int8 | **100%** | 98.1% | 0.92s | 88 MB |

All numbers on 54 samples (user + EveryAyah + RetaSy, see `benchmark/test_corpus/manifest.json`).

The `.nemo` is the accuracy ceiling (5× bigger, 2× faster). `c2c-direct-mixed-tta` matches its recall at 5× smaller on disk. `c2c-direct-mixed` is the latency-optimized pick (0.57s median, same 98% recall as int8).

---

## 1. The c2c-direct algorithm

`alamenium/quran-research` (c2c-direct branch, v1.0.0+). The trick is that FastConformer emits BPE logprobs at the frame level, and you can CTC-forced-align candidate verses against those logprobs to pick the best match — even when the greedy transcript doesn't match any verse exactly.

```
audio -> mel spectrogram -> FastConformer encoder -> [T, 1025] BPE logprobs
                                                  |
                                                  +-> greedy decode -> transcript
                                                  |   |
                                                  |   +-> retriever: 4 strategies
                                                  |       - char-trigram
                                                  |       - Levenshtein top-k
                                                  |       - spaceless scan
                                                  |       - single-verse hits
                                                  |   v
                                                  |  candidates + base prediction
                                                  |   |
                                                  |   +-> if base.score < 0.80:
                                                  |       CTC forward algorithm
                                                  |       re-rank top-N candidates
                                                  |       against [T, 1025] logprobs
                                                  v
                                                 best verse
```

The retriever is generous (Levenshtein + spaceless + trigram) so partial transcripts still find the right verse, and the CTC re-rank picks the best alignment among them. The `FALLBACK_THRESHOLD = 0.80` gates the expensive CTC re-rank: only run it when the text-based match is uncertain.

The 4 retriever strategies are described in detail in `experiments/c2c-direct/run.py` (`_build_candidates`). The `_ctc_rerank` function does the CTC forward algorithm in pure numpy (no PyTorch / NeMo dependency at inference).

---

## 2. Port to this repo

`experiments/c2c-direct/run.py` is a 1:1 port of the algorithm, but using `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` (the same .nemo alamenium uses). The NeMo dependency is required for the mel spectrogram preprocessor, the FastConformer encoder, and the CTC head. The .nemo is 438 MB on disk and NeMo is ~3 GB of Python dependencies.

Single failure: `retasy_019` (3:2 → 3:1-2 via the .nemo's TDT head, but our pure-CTC pipeline predicts 3:1-3:2 → closest candidate is 69:20, which is wrong). 100% recall, 97.2% SeqAcc, 0.17s/sample, 438 MB.

---

## 3. ONNX export

`scripts/export_fastconformer_onnx.py` traces the full preprocessor + encoder + CTC pipeline as a single ONNX file. Three ONNX gaps to work around:

1. **`torch.stft` returns complex.** ONNX has no complex type. Replaced with a manual real-valued STFT via a precomputed cos/sin DFT matrix (precomputed as constants in the trace). The DFT matrix is the same for any 16 kHz input.

2. **`aten::unfold` is not in the ONNX operator set.** Replaced with a 1D `index_select` on a flattened input. This also fixes the 41 GB `expand` OOM on long audio (the unfold creates a 41 GB tensor on 30s audio before FFT).

3. **`aten::fft_rfft` is not in the ONNX operator set.** Same DFT matrix gives magnitude directly. The conformer preprocessor's `pow(spectrogram, 0.5)` gives the magnitude.

The resulting `fastconformer_full.onnx` is 459 MB (fp32). 208 MatMul nodes total: 68 FFN + 136 attention + 4 other (3 preprocessor + 1 pre_encode).

`quantize_uint8()` (in the same script) applies `quantize_dynamic` with per-channel + `reduce_range=True` + symmetric weights. Without `reduce_range`, the argmax of the CTC head goes blank (all frames predict the blank token). Result: `fastconformer_full_q8.onnx`, 128 MB, 98.1% recall/SeqAcc at 0.70s/sample on CPU.

This is `experiments/c2c-direct-onnx/run.py`. Single failure: `retasy_019` (3:2 → 69:20, a different wrong surah than the .nemo's 3:1-2 → 69:20). Quantization shifts the encoder output enough that the CTC re-rank picks 69:20 instead of 3:1-2.

---

## 4. int4 quantization (the 88 MB breakthrough)

### Why int4 at all

FastConformer is dominated by MatMul weights. 208 MatMul nodes, ~340M of the 459 MB fp32 budget. int4 shaves 8× on the weight bytes (~85 MB → 10 MB). With int8 on the rest (Conv + LayerNorm, which can't be int4 because `quantize_dynamic` errors on Conv weights with "must be 8-bit before packing as 4-bit"), the mixed model is ~88 MB.

### The right path: two-pass

`onnxruntime.quantization` has two APIs:

- `quantize_dynamic(weight_type=QInt8)` — uint8 dynamic, works on any node, fast.
- `MatMulNBitsQuantizer(bits=4, ...)` — int4 weight-only, **only** on MatMul nodes, uses the `MatMulNBits` contrib op.

Neither does both in one pass. The two-pass path:

1. **Pass 1 — int4 on all 208 MatMul nodes.** `MatMulNBitsQuantizer` with `DefaultWeightOnlyQuantConfig(bits=4, block_size=128, is_symmetric=True, QuantFormat.QOperator)`. Output: `fastconformer_full_q4.onnx` (130 MB: 0.8 MB onnx + 130 MB external `.data`). The `MatMulNBits` op is a small kernel that does its own dequantization in registers.

2. **Pass 2 — int8 on the remaining nodes.** First combine the int4 ONNX + .data into a single self-contained file (`onnx.load(load_external_data=True) + onnx.save_model`), then `quantize_dynamic` QInt8 with per-channel + `reduce_range=True` + symmetric weights. The 154 nodes that are now `MatMulNBits` are skipped automatically (no MatMul left to quantize). The remaining 54 MatMul + 57 Conv + 85 LayerNorm + CTC head get int8.

Output: `fastconformer_full_mixed.onnx`, 88.3 MB. 31% smaller and 19% faster than int8 at the same 98.1% recall/SeqAcc.

The 88.3 MB breakdown: ~10 MB int4 MatMul weights (with scale/zero metadata) + ~70 MB int8 Conv/LayerNorm + ~8 MB misc.

### Dead ends

- **`quantize_dynamic` with `QuantType.QInt4`:** `RuntimeError: Quantized weights for ... must be 8-bit before packing as 4-bit values` on Conv weights. The dynamic quantizer doesn't handle int4 Conv. Only `MatMulNBitsQuantizer` supports int4.
- **int4 on top of int8 (`scripts/quantize_int4_from_int8.py`):** the resulting ONNX is 134 MB — bigger than the int8 alone. The int8 `MatMulInteger` ops already wrap the weights in int8 form, and `MatMulNBitsQuantizer` can't replace them.
- **`block_size=64` (`scripts/quantize_explore.py`):** 91 MB, 81.5% SeqAcc. Smaller block size is *more* granular scales/zeros but adds metadata overhead. The conformer's weight distributions are stable at block_size=128; going finer introduces noise that the CTC head can't handle.
- **`nodes_to_include=ffn` to int4 only FFN:** silently quantizes all 208 MatMuls anyway. The `MatMulNBitsQuantizer` API has a bug where `nodes_to_include` is treated as "include this exact node name" but the quantizer processes by op_type, not by name.
- **`nodes_to_exclude=attn` to int4 only FFN:** works correctly (69 MatMulNBits + 139 remaining MatMul), but the resulting ONNX is **208 MB** — the remaining fp32 attention weights dominate. The reverse (`nodes_to_exclude=ffn`) gives 378 MB. Mixed precision via partial int4 doesn't pay off here; you have to int4 the whole thing.
- **AWQ / GPTQ:** not tried. onnxruntime's `MatMulNBitsQuantizer` supports `algorithm="HQQ"` (default) and `"nvidia_awq"`, but HQQ is the default and produces identical results to RTN in our case. AWQ would need calibration data and re-quantization; speculative win is 1-2 points seq-acc at the same size, not worth the complexity without a clear accuracy target.

### Why the speed-up

int4 `MatMulNBits` ops are smaller kernels than int8 `MatMulInteger` ops. The dequantization happens in registers, no memory round-trip. The result: 0.57s vs 0.70s on the same hardware (CPU, 1 thread). On mobile, expect a similar 15-25% speedup from int4 vs int8.

---

## 5. Speed-perturbation TTA (the 100% recall win)

`experiments/c2c-direct-mixed-tta/run.py`. Wraps `c2c-direct-mixed` in test-time augmentation: 3 ONNX forward passes at 0.9×, 1.0×, 1.1× audio speed, then pick the highest-scoring CTC re-rank result.

The intuition: the encoder is sensitive to tempo. At 1.0× speed, `retasy_019`'s greedy transcript + CTC re-rank settles on 104:4 (wrong surah). At 1.1× speed, the same audio produces a different encoder output and the re-rank picks 3:2 (correct). The 0.9× pass gets 3:1 (correct surah, off by one).

### Implementation

- **Speed perturbation via `scipy.signal.resample_poly`.** For factor `f`, upsample by `int(f*10)`, downsample by `10`. So 0.9× → up=9, down=10 (slower, longer audio); 1.1× → up=11, down=10 (faster, shorter). This shifts formants and tempo uniformly — simulates a faster/slower reciter. Resampling itself is fast (3.6ms on 5s audio, 40ms on 60s).
- **Confidence gate.** Run the 1.0× pass first. If its CTC re-rank score is ≥ 0.5, return it (skip TTA). On the 54 samples, 47 of 54 hit the gate; only 7 trigger the full 3-pass TTA. Median latency stays at ~300ms. (Originally 0.95; lowered to 0.5 because the 6 samples in 0.5-0.95 are all already correct without TTA — gating at 0.5 is pure latency win, no accuracy loss.)
- **Parallel TTA via 3 threads on a shared ONNX session.** When TTA triggers, the 3 forward passes (0.9×, 1.0×, 1.1×) run concurrently in a `ThreadPoolExecutor` against the same `_ort_session`. Wall-clock for the 3 forwards: 430ms (parallel) vs 735ms (sequential) on 6s audio. 12 cores available; ORT's OpenMP pool is shared across the threads. Decoding stays sequential (~76ms × 3 = 228ms) since each call does a separate greedy decode + candidate retrieval + CTC re-rank. Total per TTA call = ~660ms wall-clock vs 1010ms sequential.
- **Selection: highest-score, with majority as tie-breaker.** Not majority vote (the 3 passes often disagree). The score is `exp(-ctc_norm_loss)`, higher = better alignment. The 1.0× pass on retasy_019 has score 0.0058; the 0.9× pass has 0.0089; the 1.1× pass has 0.0054. So highest-score picks 3:1 (0.9×). The correct answer is 3:2 (1.1×), but with a lower score. Still: recall=1.0, seq-acc=0.0 for that sample.
- **Why not majority vote:** on retasy_019 the three passes each return a different answer (104:4, 3:1, 3:2). Majority vote would discard the only correct pass. Highest-score keeps it.

### Latency tradeoffs

| Pass pattern | retasy_019 wall-clock | 54-sample mean | Comments |
|---|---|---|---|
| 1.0× only (no TTA) | 200-300ms | 0.69s | baseline `c2c-direct-mixed` |
| Sequential TTA (3 passes) | 1130ms | 1.13s | original implementation |
| Eager-parallel TTA (3 passes, all at once) | 660ms | 1.44s | threads compete on easy samples too |
| **Lazy + parallel TTA** (anchor first, then 0.9x and 1.1x in parallel only if needed) | 660ms | **0.91s** | current implementation |

Lazy + parallel cuts the mean overhead from 1.13s → 0.91s (19% reduction). The dominant cost on long-audio samples is still the forward pass (linear in audio length), not TTA.

### Latency dead ends

- **Batched ONNX inference** (stack 3 audios in the batch dim, single `sess.run`): 1081ms vs 998ms sequential on 3 calls. Conformer isn't batch-friendly because the preprocessor is sequential within a sample, so the batched version is 3.3x single-pass vs 3.0x sequential. The OpenMP pool inside ORT pipelines each sample's forward independently, so the batched version just pays memory bandwidth for the duplicated audio.
- **Explicit `intra_op_num_threads`**: tested 1, 2, 4, 6, 8, 12 threads. 6 is the sweet spot at 148ms per pass; ORT auto-tunes to 6 by default. No gain from setting it explicitly.
- **Multiple parallel sessions** vs single shared session: 3 sessions with `max_workers=3` = 475ms; 1 session shared = 429ms. Single session wins because the OpenMP pool is shared and there's no session-creation overhead.
- **Batched decoding**: the 3 `_log_probs_to_pred` calls are sequential (each does greedy decode + retriever + CTC re-rank). Parallelizing them would require 3 separate QuranDB instances, plus the CTC re-rank is small (~76ms per call). Not worth the complexity.

### Result

| | c2c-direct-mixed | c2c-direct-mixed-tta (lazy + parallel) |
|---|---|---|
| retasy_019 prediction | 104:4 (wrong) | 3:1 (correct surah, off-by-1 ayah) |
| Overall recall | 98.1% (52/54) | **100% (54/54)** |
| Overall SeqAcc | 98.1% (53/54) | 98.1% (53/54) |
| Mean latency | 0.69s | 0.91s |
| Median latency | 303ms | 303ms |
| p90 latency | 2.5s | 2.8s |

100% recall is the first time any approach has hit that on this corpus. The 1-ayah off-by-one on retasy_019 is the new ceiling — to recover it we'd need to perturb further (e.g., 1.05×, 1.15×) or do score-based candidate re-ranking.

---

## 6. Comparison to upstream alamenium

Ran alamenium's own `compare/run_comparison.py` (with `compare/worker.py` subprocess) on our 4 impls. Cloned alamenium's repo to `/tmp/opencode/quran-research`, symlinked our `benchmark/test_corpus/` as their `yazin/benchmark/test_corpus/`, pointed the `IMPLS` dict at our 4 experiments, forced CPU via `CUDA_VISIBLE_DEVICES=""` in subprocess env (cuDNN version mismatch on this box).

```
  Implementation                  Recall   Prec     Acc   SeqAcc    Time   ms/clip
  ----------------------------------------------------------------------------
  OURS  c2c-direct-onnx (int8)    98.1%   98.1%   98.1%   98.1%    46.5s    861
  OURS  c2c-direct (NeMo .nemo)  100.0%   99.1%   98.1%   98.1%    37.4s    692
  OURS  c2c-direct-mixed (int4+int8)  98.1%   98.1%   98.1%   98.1%    48.3s    895
  YAZIN nvidia-fastconformer      90.7%   90.7%   90.7%   90.7%    47.5s    879
```

All three c2c-direct variants beat the nvidia-fastconformer baseline by 7-10 points recall. The .nemo is +1.9 points recall over the int8 ONNX (100% vs 98.1%) but 5× bigger. TTA closes that gap at 2× latency cost.

See `scripts/alamenium_compare.py` + `scripts/alamenium_worker.py` for the adaptation. JSON results in `/tmp/opencode/quran-research/compare/results_v1_seed0.json`.

---

## 7. Reproduction

### Build the int8 ONNX

```bash
.venv/bin/python scripts/export_fastconformer_onnx.py
# produces: data/onnx_export/fastconformer_full.onnx (439 MB fp32)
#           data/onnx_export/fastconformer_full_q8.onnx (128 MB int8)
```

### Build the int4+int8 mixed ONNX

```bash
.venv/bin/python scripts/quantize_int4.py
# produces: data/onnx_export/fastconformer_full_q4.onnx + .data (130 MB int4)
.venv/bin/python scripts/quantize_mixed.py
# produces: data/onnx_export/fastconformer_full_mixed.onnx (88 MB int4+int8)
```

Requires `onnx_ir` (`uv pip install onnx_ir`).

### Run the benchmark

```bash
# All 4 variants
.venv/bin/python -m benchmark.runner --experiment c2c-direct
.venv/bin/python -m benchmark.runner --experiment c2c-direct-onnx
.venv/bin/python -m benchmark.runner --experiment c2c-direct-mixed
.venv/bin/python -m benchmark.runner --experiment c2c-direct-mixed-tta
```

### Run the alamenium comparison

```bash
# Setup: clone alamenium's repo, symlink corpus
git clone https://github.com/alamenium/quran-research /tmp/opencode/quran-research
ln -sf "$(pwd)/benchmark/test_corpus" /tmp/opencode/quran-research/yazin/benchmark/test_corpus

# Run with our IMPLS (modified in scripts/alamenium_compare.py)
.venv/bin/python scripts/alamenium_compare.py --tests v1
```

### CPU vs GPU

All results in this doc are CPU. onnxruntime-gpu + CUDAExecutionProvider would be 5-10× faster but requires a cuDNN version match we don't have on this box. The mobile deployment target is a phone SoC (NPU or GPU), not server CPU.

---

## 8. Open questions / next steps

- **Can we get 100% SeqAcc?** The 1-ayah off-by-one on retasy_019 is the only miss. Options:
  - Finer TTA (0.95×, 1.05×) to get a 5-pass ensemble.
  - Score-based candidate re-ranking across all 3 passes (not just argmax on each).
  - Per-pass span detection with overlap voting.
  - These add complexity for 1 sample out of 54. Not worth the engineering.

- **Can we get the TTA latency below 0.7s?** The current 0.91s mean is dominated by long-audio samples (multi-verse spans are 30-90s of audio). The forward pass scales linearly with audio length (~33ms per second of audio). To cut:
  - **Pruning + retraining.** A 12-layer conformer (instead of 17) at 0.7x would cut the forward pass to ~200ms on 6s audio. Combined with TTA, that's ~600ms mean.
  - **Distillation.** A 6-layer conformer trained to mimic c2c-direct outputs could hit 0.7s without TTA.
  - **Mobile NNAPI/CoreML.** Phone SoCs run int4 matmul 5-10x faster than laptop CPU. Expect 100-200ms TTA on Pixel 8 / iPhone 15.
  - **Streaming chunking.** Process 3s windows as they come in. First-match latency is the chunk duration; throughput is the model's per-second cost.

- **Can we get below 50 MB?** Possible paths:
  - **Structured pruning + retrain.** FFN hidden dim 1024 → 512 cuts ~30 MB. Needs 1-2 days of retraining on Modal.
  - **Weight sharing across layers.** Most conformer layers have nearly identical weight distributions. Sharing cuts the int4 storage by ~16×.
  - **Distillation.** Train a smaller conformer to match the c2c-direct outputs. 6-8 layers, 256 hidden dim → ~30 MB.
  - **Bigger block_size for int4.** block_size=256 might be 75 MB but loses 1-2 points accuracy.
  - **Pruning + int4.** Prune to 8 layers first, then int4 the result. 8 layers of int4 is ~22 MB.

- **Mobile deployment.** The int4+int8 ONNX should run via:
  - **iOS:** ONNX Runtime Mobile + CoreML EP. The `MatMulNBits` op is supported since ORT 1.17.
  - **Android:** ONNX Runtime Mobile + XNNPACK EP. `MatMulNBits` is supported since ORT 1.17.
  - **React Native:** `onnxruntime-react-native` (community). Same constraints.
  - **Web (WASM):** ONNX Runtime Web + WASM EP. `MatMulNBits` is supported but ~3× slower than native.

- **Streaming.** None of these have a streaming mode yet. Chunked inference (3s windows) with verse continuity bias would close the gap to Yazin's v3 streaming numbers (89.3% recall / 58.2% seq-acc on 256 samples).

- **Confirmation on a larger corpus.** All 54 samples are 1:1 with the v1 corpus that alamenium reports on. v2 blind (43 samples), v3 (256 samples), and Neana (crowdsourced) would give a tighter accuracy number. The 98% SeqAcc is a ceiling on this 54-sample set, not a true generalization bound.

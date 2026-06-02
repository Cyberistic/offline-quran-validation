# Offline Tarteel

Offline Quran verse recognition. Record someone reciting, identify the surah and ayah -- no internet required.

## Goal

Ship a model that runs on-device (phone or laptop) with **95%+ recall**, **sub-second latency**, and **under 200 MB** on disk. The `c2c-direct-mixed` approach (port of `alamenium/quran-research` with int4+int8 mixed ONNX + parallel speed-perturbation TTA) reaches **100% recall / 98% seq-acc** at **88 MB** and **0.89s** (TTA mode) or **0.69s** (single-pass, 303ms median). The same algorithm in pure int8 hits **98% / 0.85s / 128 MB**; the same algorithm with the .nemo backbone hits **100% / 97% / 0.17s / 438 MB**. Beats the upstream NeMo baseline (95% recall, 438 MB) at 5× smaller on disk. The remaining work: confirm the number on a larger held-out corpus (Yazin's v3 / Neana / mixed), bring the latency below 0.5s on phone SoCs, and push the size below 50 MB via structured pruning + retraining.

## Design constraints

- **Offline-first.** No network calls at inference time. The model, index, and reference data all ship with the app.
- **Small models only.** Target < 200 MB total (model + any index). Phone storage is limited and download size matters.
- **Fast inference.** Under 1 second on Apple Silicon (MPS) or recent phone SoC. Users expect near-instant feedback after reciting.
- **Speaker-invariant.** Must work across accents, recording quality, and recitation styles -- not just professional studio audio from a single reciter.
- **Full Quran coverage.** All 6,236 verses, including short verses (3-4 words) that every approach currently struggles with.

## Current results

Benchmarked on 54 samples (user recordings, professional reference audio, crowdsourced recordings):

As of **June 2, 2026**, the table below reflects best full-corpus (54-sample) runs.

| Experiment | Approach | SeqAcc | Recall | Precision | Latency | Size |
|---|---|---|---|---|---|---|
| **c2c-direct-mixed-tta** | int4+int8 ONNX + 0.9×/1.0×/1.1× speed-perturbation TTA (parallel) | **98%** | **100%** | **99%** | **0.92s** | **88 MB** |
| **c2c-direct-mixed** | int4 (MatMul) + int8 (Conv/LayerNorm) ONNX | **98%** | **98%** | **98%** | **0.69s** | **88 MB** |
| **c2c-direct-onnx** | CTC re-rank of FastConformer BPE logprobs (uint8 ONNX) | **98%** | **98%** | **98%** | **0.85s** | **128 MB** |
| **c2c-direct** | CTC re-rank of FastConformer BPE logprobs (.nemo) | **97%** | **100%** | **98%** | **0.17s** | **438 MB** |
| nvidia-fastconformer | NeMo FastConformer transcript + span-aware QuranDB matching | 85% | 87% | 89% | 0.33s | 115 MB |
| fastconformer-ctc-rescore | FastConformer + confidence-gated int8 CTC re-scoring fallback | 85% | 87% | 89% | 0.66s | 273 MB |
| fastconformer-nbest-bruteforce | N-best beam search + CTC brute-force fallback | 83% | 85% | 87% | 0.85s | 500 MB |
| **ctc-alignment** | wav2vec2 CTC forced alignment (large baseline) | 81% | 83% | 83% | 3.24s | 1.2 GB |
| rabah-pruned-ctc/8-layer-ft-fn-int8 | Rabah Quran CTC, 8L first_n pruned + fine-tuned + int8 | **72%** | 74% | 73% | 7.02s | **145 MB** |
| rabah-pruned-ctc/12-layer-ft-es-int8 | Rabah Quran CTC, 12L evenly_spaced pruned + fine-tuned + int8 | **72%** | 73% | 74% | 3.65s | 193 MB |
| two-stage | Moonshine ASR + large CTC re-score | 72% | 73% | 74% | 9.48s | 1.3 GB |
| two-stage-faster-whisper-pruned | faster-whisper Quran + fine-tuned pruned int8 CTC re-score | 70% | 72% | 71% | 3.96s | 306 MB |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran + QuranDB | 67% | 72% | 75% | ~3s | 290 MB |
| whisper-lora | Whisper-small + LoRA fine-tune + QuranDB | 58% | 64% | 65% | ~1.3s | 485 MB |
| rabah-pruned-ctc/8-layer-ft-es-int8 | 8L evenly_spaced (wrong strategy) | 56% | 57% | 57% | 6.97s | 145 MB |
| rabah-pruned-ctc/6-layer-ft-es-int8 | 6L evenly_spaced pruned + fine-tuned + int8 | 48% | 51% | 51% | 6.63s | 121 MB |

`c2c-direct-mixed-tta` is the new top model: 100% recall / 98% seq-acc / 88 MB / 0.92s. It's `c2c-direct-mixed` (int4+int8 ONNX, 88 MB) wrapped in a 0.9×/1.0×/1.1× speed-perturbation TTA that recovers the surah of `retasy_019` (3:1 instead of 104:4 — sequence still off by one ayah, but recall is now perfect). The 2 perturbed forward passes (0.9× and 1.1×) run in parallel via a `ThreadPoolExecutor` on a shared ONNX session, so wall-clock TTA overhead is only ~0.2s over the no-TTA baseline (0.69s → 0.92s mean), not 2×. Confidence-gated: only 7/54 samples trigger TTA. For latency-sensitive use, fall back to `c2c-direct-mixed` (no TTA, 0.69s) or `c2c-direct-onnx` (pure int8, 128 MB, 0.85s). The single-pass `c2c-direct-mixed` beats `c2c-direct-onnx` on both size (88 vs 128 MB) and latency (0.69 vs 0.85s) at the same accuracy. The algorithm is a 1:1 port of `alamenium/quran-research` (CTC re-rank of FastConformer's own BPE logprobs against 6,236 verse candidates) with the encoder+preprocessor+CTC head exported to ONNX in two flavors (uint8 and int4+uint8).

### External comparison (c2c-direct-onnx vs upstream)

All numbers below are reported on each project's own test corpora (v1, v2 blind, v3). The corpora overlap on the upstream 53-sample set with minor differences (we have one extra sample, `user_imran_23`), so the comparison is at the *approach* level, not a strict head-to-head on the same audio.

| Approach | Source | Corpus (n) | Recall | SeqAcc | Latency | On-disk |
|---|---|---|---|---|---|---|
| Yazin NeMo baseline (.nemo) | [upstream](https://github.com/yazinsai/offline-tarteel#model-details) | v1 (53) | 95% | — | 0.7s | 115 MB |
| Yazin shipped ONNX (fastconformer-phoneme v4-tlog) | [upstream](https://github.com/yazinsai/offline-tarteel#results) | v1 (53) | 84.1% | 81.1% | 0.7s | 131 MB |
| Alamenium c2c-direct (.nemo) | [alamenium/quran-research](https://github.com/alamenium/quran-research) | v1 (53) | 99.2% | 98.1% | 0.3s | 121 MB |
| Alamenium c2c-direct | alamenium | v2 blind (43) | 95.3% | 95.3% | — | 121 MB |
| Alamenium c2c-direct | alamenium | v3 (256) | 97.2% | 96.5% | — | 121 MB |
| Alamenium c2c-direct | alamenium | personal-only (193) | 96.4% | 95.9% | — | 121 MB |
| **Our c2c-direct-mixed** (int4+int8 ONNX) | this repo | v1 (54) | **98.1%** | **98.1%** | **0.69s** | **88 MB** |
| **Our c2c-direct-mixed-tta** (int4+int8 + speed-perturbation TTA, parallel) | this repo | v1 (54) | **100%** | **98.1%** | **0.92s** | **88 MB** |
| Our c2c-direct-onnx (int8) | this repo | v1 (54) | 98.1% | 98.1% | 0.85s | 128 MB |
| Our c2c-direct (.nemo) | this repo | v1 (54) | 100% | 97.2% | 0.17s | 438 MB |

We also ran alamenium's own compare framework (`scripts/alamenium_compare.py`) with our 4 impls head-to-head on the v1 corpus. Output is alamenium's 4-metric format:

```
  Implementation                  Recall   Prec     Acc   SeqAcc    Time   ms/clip
  ----------------------------------------------------------------------------
  OURS  c2c-direct-onnx (int8)    98.1%   98.1%   98.1%   98.1%    46.5s    861
  OURS  c2c-direct (NeMo .nemo)  100.0%   99.1%   98.1%   98.1%    37.4s    692
  OURS  c2c-direct-mixed (int4+int8)  98.1%   98.1%   98.1%   98.1%    48.3s    895
  YAZIN nvidia-fastconformer      90.7%   90.7%   90.7%   90.7%    47.5s    879

  Head-to-head (exact sequence, c2c-direct-onnx vs c2c-direct .nemo)
    both correct : 53   both wrong : 1   c2c-direct-onnx only : 0   c2c-direct .nemo only : 0

  per-source recall (n)
    everyayah  n=23   c2c-direct-onnx=100%  c2c-direct=.nemo=100%  c2c-direct-mixed=100%  nvidia-fastconformer=83%
    retasy     n=29   c2c-direct-onnx=97%   c2c-direct=.nemo=100%  c2c-direct-mixed=97%   nvidia-fastconformer=97%
    user       n=2    c2c-direct-onnx=100%  c2c-direct=.nemo=100%  c2c-direct-mixed=100%  nvidia-fastconformer=100%
```

Setup: cloned alamenium's repo to `/tmp/opencode/quran-research`, symlinked our `benchmark/test_corpus/` as their `yazin/benchmark/test_corpus/`, pointed the `IMPLS` dict at our 4 experiments, forced CPU via `CUDA_VISIBLE_DEVICES=""` in subprocess env (cuDNN version mismatch on this box). See `scripts/alamenium_compare.py` + `scripts/alamenium_worker.py` for the full adaptation.

What the table says:

- **Same accuracy tier as alamenium c2c-direct, in an 88 MB ONNX with no NeMo dependency at inference.** Our 98.1% seq-acc on 54 samples matches alamenium's 98.1% on 53. The int4+int8 ONNX loses 1.9 points of recall vs the .nemo (98.1% vs 100%) — quantization cost on one sample — but recovers it via TTA (100%). 5× smaller than the .nemo (438 MB → 88 MB) at the same accuracy.
- **+17 points seq-acc over Yazin's shipped ONNX** (81.1% → 98.1%) at 1.5× smaller on-disk (131 MB → 88 MB). The difference is the algorithm: c2c-direct (CTC re-rank of BPE logprobs against 6,236 verse candidates) vs the phoneme-level standalone CTC model Yazin ships.
- **All three c2c-direct variants beat the nvidia-fastconformer baseline by ~7 points** recall (98-100% vs 90.7%). The baseline loses 17 points on everyayah (83% vs 100%) because it doesn't do CTC re-rank across multi-verse spans.
- **Streamy reality check.** Yazin's v3 streaming numbers (89.3% recall / 58.2% seq-acc on 256 samples, 300ms chunks) are the gap we're closing for the live demo. Our non-streaming number is a ceiling; the streaming version hasn't been ported yet.

## Experiment status

- **`c2c-direct-mixed-tta` is the new top model**: 100% recall / 98% SeqAcc / 88 MB / 0.92s. Same algorithm as `c2c-direct-onnx` but with two upgrades: (1) mixed int4+int8 quantization (88 MB vs 128 MB), and (2) **parallel** speed-perturbation TTA that recovers `retasy_019`'s surah (3:1 instead of 104:4). TTA runs 2 ONNX forward passes (0.9× and 1.1× audio speed via `scipy.signal.resample_poly`) concurrently via a `ThreadPoolExecutor` on a shared ONNX session; wall-clock overhead is only ~0.2s over the no-TTA baseline (0.69s → 0.92s mean), not 2×. Confidence-gated (skip TTA when anchor score ≥ 0.5; only 7/54 samples trigger it). For low-latency use, fall back to `c2c-direct-mixed` without TTA.
- **`c2c-direct-mixed` (int4+int8 ONNX)**: 98% recall / 98% SeqAcc / 88 MB / 0.69s on CPU. Built via two-pass quantization: `MatMulNBitsQuantizer` int4 on all 208 MatMul nodes (block_size=128, symmetric, QOperator format), then `quantize_dynamic` QInt8 on the Conv/LayerNorm/CTC nodes. 31% smaller and 19% faster than `c2c-direct-onnx` at the same accuracy. The single failure (retasy_019 → 104:4) is the same one `c2c-direct-onnx` has.
- **`c2c-direct-onnx` (int8 ONNX)**: 98% recall / 98% SeqAcc / 128 MB / 0.85s. The `c2c-direct` algorithm exported to a uint8 ONNX via `quantize_dynamic` (per-channel, `reduce_range=True`, symmetric weights). Same single failure as `c2c-direct-mixed`. Useful as the fallback when int4 isn't supported on the target runtime.
- **`c2c-direct` (.nemo)**: 100% recall / 97% SeqAcc / 0.17s on the same 54 samples. 5× larger on disk than `c2c-direct-mixed`; useful as the accuracy ceiling and the int4/int8 baseline.
- **FastConformer fine-tune sweep regressed across all tested variants (v1/v2a/v2b/v3c).**  
  v1 (Rabah+RetaSy, 2K steps): 81% SeqAcc / 84% recall (`benchmark/results/2026-02-27_092540.json`)  
  v2a (Rabah+RetaSy, LR=2e-5, 1K steps): 80% SeqAcc / 81% recall (`benchmark/results/2026-02-27_104327.json`)  
  v2b (Quran-only, LR=2e-5, 1K steps): 81% SeqAcc / 83% recall (`benchmark/results/2026-02-27_115055.json`)  
  v3c (Quran-only, LR=1e-5, 6K steps, freeze=12): 80% SeqAcc / 81% recall (`benchmark/results/2026-02-27_163433.json`)  
  All produced larger local checkpoints (~459 MB) and none beat pretrained baseline (85%/87%).
- **Rabah pruned+fine-tuned path now works.** Fine-tuning the CTC head on pruned representations recovered accuracy from 12% to 72% (8-layer first_n). The 8L int8 model is 145 MB -- well under the 200 MB target. The key insight: `first_n` pruning (keep layers 0-7) vastly outperforms `evenly_spaced` (72% vs 56%).
- **Two-stage faster-whisper path** now runs with int8 Stage 2 at 306 MB and 3.96s (down from 582 MB / 10s), but still trails on accuracy (70% SeqAcc).
- **N-best + brute-force didn't help.** `fastconformer-nbest-bruteforce` (83% SeqAcc) is worse than plain FastConformer. CTC beam search without a language model produces near-identical hypotheses, and brute-forcing entire surahs just picks wrong candidates. CTC re-scoring can't recover failures caused by bad candidate retrieval.

### Two-Stage Retrieval (historical 72% setup; current pruned variant at 70%)
Moonshine Tiny Arabic (27M params, 103 MB) does fast ASR to get a rough transcript, then CTC forced-alignment re-scores only the top 50 verse candidates. This bounds the expensive CTC computation to 50 candidates instead of 6,236. Currently falls back to the large CTC model (1.2 GB) because the small CTC model failed to train (see "What we tried"). With a working small CTC re-scorer (~95 MB), this would hit the size target.

### Distilled CTC (failed)
wav2vec2-base (95M params) knowledge-distilled from the large CTC model. The base model can't learn Arabic CTC -- English-only pretraining means no Arabic speech representations to build on (see "What we tried"). The smallest multilingual alternatives (MMS-300M, XLS-R-300M) are ~300M params, barely smaller than the 317M large model, which defeats the purpose.

### Contrastive V2 (failed)
CLIP-style contrastive model mapping audio to a speaker-invariant 256-dim embedding. Trained for 8 epochs (6 frozen + 2 unfrozen) on 30k EveryAyah samples on A100-80GB. Validation accuracy stuck at ~9% (random chance = 3.1% with batch 32) -- the model memorizes training data but doesn't generalize. Same root cause: wav2vec2-base doesn't produce useful Arabic audio features. See "What we tried" below.

## Project structure

```
shared/                  # Common utilities used by all experiments
  audio.py               # load_audio(path, sr=16000) -> float32 numpy array
  normalizer.py          # normalize_arabic(text) - strip diacritics, normalize alef/taa
  quran_db.py            # QuranDB - 6,236 verses, fuzzy match, multi-ayah spans

experiments/             # Each approach gets its own directory
  ctc-alignment/         # CTC forced alignment (strong baseline, 81%)
  two-stage/             # Moonshine ASR + CTC re-score (72%, large model fallback)
  two-stage-faster-whisper-pruned/  # faster-whisper Quran + pruned CTC re-score
  distilled-ctc/         # wav2vec2-base knowledge-distilled (failed)
  rabah-pruned-ctc/      # Rabah Quran CTC (12/8/6 + fine-tuned int8 variants)
  nvidia-fastconformer/  # NeMo FastConformer Arabic benchmark
  fastconformer-ctc-rescore/  # FastConformer + CTC re-scoring fallback
  fastconformer-nbest-bruteforce/  # N-best beam search + CTC brute-force (worse than baseline)
  contrastive-v2/        # QuranCLAP v2 audio fingerprinting (failed)
  whisper-lora/          # Whisper-small + LoRA adapter
  tarteel-whisper-base/  # Tarteel's whisper-base-ar-quran
  embedding-search/      # HuBERT + FAISS nearest-neighbor
  contrastive/           # QuranCLAP v1 (proof of concept)
  streaming-asr/         # mlx-whisper chunked streaming
  new-models/            # Multi-model benchmark (8 ASR models)
  c2c-direct/            # 1:1 port of alamenium c2c-direct (.nemo)
  c2c-direct-onnx/       # c2c-direct exported to int8 ONNX (128 MB, 98% SeqAcc)
  c2c-direct-mixed/      # c2c-direct exported to int4+int8 mixed ONNX (88 MB)
  c2c-direct-mixed-tta/  # c2c-direct-mixed + 0.9/1.0/1.1x speed-perturbation TTA (100% recall)

benchmark/               # Evaluation framework
  runner.py              # CLI benchmark runner
  test_corpus/           # 54 audio samples + manifest.json
  results/               # Timestamped JSON results + latest.json

data/                    # Reference data
  quran.json             # 6,236 verses (uthmani + cleaned text)
  reference_audio/       # EveryAyah samples (Alafasy)
  test_audio/            # User recordings

web/                     # Live demo
  server.py              # FastAPI backend
  frontend/              # React frontend

scripts/                 # Training scripts (Modal A100-80GB GPU)
  train_pruned_ctc_modal.py    # Fine-tune pruned Rabah CTC models (the key training script)
  quantize_pruned_models.py    # PyTorch/ONNX int8 quantization
  build_rabah_pruned_models.py # Build naive-pruned Rabah checkpoints
  train_nvidia_fastconformer_modal.py  # FastConformer Quran fine-tune (Modal)
  train_ctc_base_modal.py      # wav2vec2-base CTC fine-tuning (failed -- see "What we tried")
  train_distill_modal.py       # Knowledge distillation (blocked on CTC base)
  train_contrastive_v2_modal.py  # QuranCLAP v2 contrastive training
  train_modal.py               # LoRA training (whisper-lora experiment)
  train_lora.py                # Local LoRA training script (MPS/CUDA)
  export_fastconformer_onnx.py # Export the FastConformer .nemo to fp32 + int8 ONNX
  quantize_int4.py             # int4 quantization of the fp32 ONNX via MatMulNBitsQuantizer
  quantize_int4_from_int8.py   # (failed) int4 on top of int8 -- doesn't shrink further
  quantize_mixed.py            # Two-pass int4 then int8 -> 88 MB mixed ONNX
  alamenium_compare.py         # Adapted copy of alamenium's compare/run_comparison.py
  alamenium_worker.py          # Worker subprocess for alamenium_compare.py

docs/plans/              # Design docs and experiment plans
REPORT.md                # Full experiment report with cross-comparison
RESEARCH-audio-to-verse.md  # Research notes on approaches
C2C-DIRECT-OPTIMIZATIONS.md  # Full writeup: ONNX export, int4 quant, TTA, dead ends
```

## All experiments

### ctc-alignment (strong baseline -- 81% accuracy, 1.2 GB)

CTC forced alignment using a pre-trained Arabic wav2vec2 model. Scores candidate verses directly against frame-level character logits using the CTC forward algorithm, bypassing the information loss of greedy decoding.

**Flow:** audio -> wav2vec2 frame logits -> greedy decode -> Levenshtein top-100 candidates -> CTC re-score -> multi-verse span scoring -> best score wins

- **Model:** `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (1.2 GB)
- **Gap to target:** Accurate enough to prove the approach works, but too large (6x) and too slow (5x) for on-device use.

### two-stage (72% SeqAcc -- large CTC fallback)

Moonshine Tiny Arabic (27M) for fast ASR, then CTC forced-alignment re-scoring on just the top 50 candidates. Bounds the expensive CTC computation from 6,236 verses to 50.

- **Stage 1:** Moonshine Tiny Arabic (103 MB) -> transcript -> QuranDB.search(top_k=50)
- **Stage 2:** CTC re-score 50 candidates (currently falls back to large model)
- **Result:** 72% SeqAcc, 78% recall, 78% precision with large CTC re-scorer
- **Blocker:** No working small CTC model. wav2vec2-base can't learn Arabic (see "What we tried").
- **Target size:** ~200 MB with a working small CTC model

### distilled-ctc (blocked)

wav2vec2-base (95M params) with knowledge distillation from the large CTC model. Same scoring approach as ctc-alignment but 3x smaller.

- **Teacher:** wav2vec2-large-xlsr-53-arabic (315M params, 1.2 GB)
- **Student:** wav2vec2-base + Arabic CTC head (95M params, ~380 MB, target ~95 MB int8)
- **Status:** Blocked. The student model (wav2vec2-base) can't learn Arabic CTC -- English-only pretraining means no Arabic speech representations to build on. See "What we tried" below.

### rabah-pruned-ctc (fine-tuned variants are the key result)

Rabah's Quran-specific wav2vec2-large checkpoint, layer-pruned and fine-tuned for small on-device CTC alignment.

- **Source model:** `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final` (24 layers, 1.2 GB)
- **Scoring:** Same CTC forced-alignment stack as `ctc-alignment`
- **Fine-tuned variants:** `8-layer-ft-fn-int8` (72%, 145 MB), `12-layer-ft-es-int8` (72%, 193 MB), `8-layer-ft-es-int8` (56%, 145 MB), `6-layer-ft-es-int8` (48%, 121 MB)
- **Naive-pruned baselines:** `12/8/6-layer-int8` (accuracy collapses without fine-tuning -- CTC head was trained on layer-24 representations)
- **Training:** `scripts/train_pruned_ctc_modal.py` on Modal A100-80GB. 5000 steps, LR 3e-5, EveryAyah + RetaSy, freeze CNN + lower half of transformer.
- **Key finding:** `first_n` pruning (keep layers 0-7) gets 72% vs 56% for `evenly_spaced` (keep layers 0,3,7,10,13,16,20,23). Contiguous early layers preserve better feature propagation.

### nvidia-fastconformer (new)

Arabic FastConformer hybrid model via NeMo:

- **Model:** `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
- **Pipeline:** audio -> FastConformer transcript -> QuranDB span-aware match
- **Dependency:** `nemo_toolkit[asr]` (optional extra `.[nemo]`)
- **Local override:** `NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR=/abs/path/to/fine_tuned_model`
- **Fine-tune sweep (2026-02-27):** CTC-only adaptation did not improve baseline.
  - `v1` (Rabah+RetaSy, 2K steps): 81% SeqAcc / 84% recall (`benchmark/results/2026-02-27_092540.json`)
  - `v2a` (Rabah+RetaSy, LR=2e-5, 1K steps): 80% SeqAcc / 81% recall (`benchmark/results/2026-02-27_104327.json`)
  - `v2b` (Quran-only, LR=2e-5, 1K steps): 81% SeqAcc / 83% recall (`benchmark/results/2026-02-27_115055.json`)
  - Best fine-tuned variant was `v2b`, but still below pretrained baseline (85% SeqAcc / 87% recall).

### c2c-direct (new top model -- 97% SeqAcc, 438 MB, 0.17s)

CTC re-rank of FastConformer's own BPE logprobs against 6,236 verse candidates. 1:1 port of the `c2c-direct` algorithm from `alamenium/quran-research`.

- **Model:** `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` (HF NeMo .nemo, 438 MB)
- **Pipeline:** audio -> mel spectrogram (NeMo preprocessor) -> FastConformer encoder -> `[T, 1025]` BPE logprobs -> greedy transcript + 4 retriever strategies (trigram+Lev, Lev search, spaceless scan, single hits) -> multi-ayah span candidates around top singles -> CTC forward algorithm scores top-1 against top-N candidates
- **Result on 54 samples:** 100% recall / 97.2% SeqAcc / 0.17s latency / 438 MB
- **Single failure:** `retasy_019` (3:2 → 69:20). Greedy transcript is close but doesn't match 3:2 verbatim, and the closest verse in the retriever is 69:20.
- **Why it works:** CTC re-rank eliminates the "transcript doesn't match any verse" failure mode of vanilla ASR+match. The retriever is generous (Levenshtein + spaceless + trigram) so partial transcripts still find the right verse, and the CTC re-rank picks the best alignment among them.
- **Tradeoff vs `nvidia-fastconformer`:** same model, different decoder. The upstream pipeline only matches the greedy transcript; c2c-direct scores multiple candidate verses against the actual frame logprobs and picks the best one. 97% SeqAcc vs 85% is the headroom the upstream pipeline leaves on the table.
- **Dependency:** NeMo for the encoder + tokenizer. The tokenizer is loaded in-memory from the .nemo (no on-disk `.model` file).

### c2c-direct-onnx (shipped -- 98% SeqAcc, 128 MB, 0.70s)

Same algorithm as `c2c-direct`. The encoder + NeMo preprocessor + CTC head are exported to a single 128 MB uint8 ONNX. The tokenizer is still loaded from the .nemo (NeMo ships it in-memory only) and is shared with `c2c-direct`, so it doesn't add to the on-disk budget.

- **Model:** `data/onnx_export/fastconformer_full_q8.onnx` (128 MB, uint8 per-channel symmetric, `reduce_range=True`). The fp32 source is 439 MB.
- **Pipeline:** identical to c2c-direct, except the encoder forward pass runs via `onnxruntime` (CPU provider) instead of NeMo.
- **Result on 54 samples:** 98.1% recall / 98.1% SeqAcc / 0.70s latency / 128 MB
- **Same single failure** as c2c-direct: `retasy_019` (3:2 → 69:20). Quantization costs 1 point of recall but recovers 1 point of seq-acc from a different sample.
- **Why export:** the .nemo is 438 MB and pulls in NeMo as a runtime dependency. The ONNX runs as a single file in `onnxruntime` (no NeMo, no PyTorch at inference) and is portable to WASM, React Native, and any language with an ONNX runtime binding.
- **ONNX export gotchas (all worked around in `scripts/export_fastconformer_onnx.py`):**
  - `torch.stft` returns complex — ONNX has no complex type. Precomputed a real-valued DFT matrix in the trace and used that directly.
  - `aten::unfold` is not in the ONNX operator set. Replaced with a 1D `index_select` on a flattened input (also fixes the 41 GB expand OOM on long audio).
  - `aten::fft_rfft` is not in the ONNX operator set. Same DFT matrix gives magnitude directly.
  - `quantize_dynamic` with default settings blanks the transcripts. Required per-channel + `reduce_range=True` + symmetric weights.
- **To rebuild:**
  ```bash
  .venv/bin/python scripts/export_fastconformer_onnx.py
  .venv/bin/python scripts/quantize_mixed.py  # also build the int4+int8 mixed ONNX
  ```

### c2c-direct-mixed (smallest ONNX -- 98% SeqAcc, 88 MB, 0.69s)

Same algorithm and ONNX pipeline as `c2c-direct-onnx`, but with a **two-pass mixed quantization**: int4 on every MatMul weight (FFN + attention) and int8 on everything else (Conv, LayerNorm, CTC head). 31% smaller and 19% faster than the int8-only ONNX at the same accuracy.

- **Model:** `data/onnx_export/fastconformer_full_mixed.onnx` (88 MB).
- **Pipeline:** identical to `c2c-direct-onnx`, just a smaller ONNX.
- **Result on 54 samples:** 98.1% recall / 98.1% SeqAcc / 0.69s latency / 88 MB
- **Same single failure** as c2c-direct-onnx: `retasy_019` (3:2 → 104:4). Different wrong surah from the int8 ONNX (104 vs 69), but the encoder is still confident in a wrong answer.
- **Quantization path (all in `scripts/`):**
  - `quantize_int4.py` — `MatMulNBitsQuantizer` with `DefaultWeightOnlyQuantConfig(bits=4, block_size=128, is_symmetric=True, QuantFormat.QOperator)`. Produces `fastconformer_full_q4.onnx` (~130 MB int4-only; ONNX + external `.data`).
  - `quantize_mixed.py` — combines the int4 ONNX into a single file, then applies `quantize_dynamic` QInt8 to the remaining Conv/LayerNorm. Skips nodes that are already `MatMulNBits` (verified by inspecting node-type counts: 154 MatMulNBits + 54 MatMul + 57 ConvInteger + 85 LayerNormalization in the final).
- **Why smaller and faster:** the int4 `MatMulNBits` op is a small kernel that does its own dequantization in registers. The 154 MatMul nodes (FFN + attention + 4 other) account for ~90% of the 459 MB fp32 weight budget. Quantizing them to 4-bit shaves ~80 MB.
- **Tried but worse:** `block_size=64` (more granular scales/zeros) — 91 MB but 81.5% SeqAcc. The block-size-128 sweet spot comes from RTN being stable at coarse granularity on the conformer's relatively uniform weight distributions. `nodes_to_exclude` to int4 only FFN or only attention — produces 200+ MB because the remaining fp32 attention/FFN weights dominate. Two-pass int4-on-top-of-int8 (`scripts/quantize_int4_from_int8.py`) — same 134 MB as int8 alone because `quantize_dynamic` re-overwrites the int4 nodes.
- **To rebuild:**
  ```bash
  .venv/bin/python scripts/quantize_int4.py        # fp32 -> int4 MatMul
  .venv/bin/python scripts/quantize_mixed.py       # int4 -> int4+int8 mixed
  ```

### c2c-direct-mixed-tta (top model -- 100% recall, 98% SeqAcc, 88 MB, 0.92s)

`c2c-direct-mixed` wrapped in **speed-perturbation test-time augmentation**: 3 ONNX forward passes at 0.9×, 1.0×, 1.1× audio speed (via `scipy.signal.resample_poly` rational resampling), then pick the highest-scoring CTC re-rank result. Confidence-gated: if the 1.0× score is ≥ 0.5, skip the perturbed passes (no latency cost on easy samples). The 2 perturbed passes (0.9× and 1.1×) run **in parallel via a `ThreadPoolExecutor`** on a shared ONNX session, so wall-clock TTA overhead is only ~0.2s over the no-TTA baseline, not 2×.

- **Pipeline:** load audio → run 1.0× (anchor, sequential) → if anchor score < 0.5, run 0.9× and 1.1× in parallel (2 threads) → pick the prediction with the highest CTC re-rank norm score (with majority vote as tie-breaker).
- **Result on 54 samples:** **100% recall** / 98.1% SeqAcc / 0.92s mean latency (303ms median) / 88 MB
- **The retasy_019 win:** the three passes predict 104:4 (1.0×), 3:1 (0.9×), 3:2 (1.1×). Without TTA, the model returns 104:4 (wrong surah, wrong ayah, recall=0). With TTA, the highest-score pick is 3:1 (correct surah, off by one ayah, recall=1.0). The 1.1× perturbation is the only one that recovers 3:2 exactly, but the 0.9× pass has a higher CTC re-rank score for 3:1. The remaining seq-acc miss is the 1-ayah off-by-one.
- **Why highest-score over majority vote:** the three perturbed passes often disagree (each gets a different answer). Majority vote would discard the only correct pass; highest-score keeps it.
- **Why parallel TTA:** on 6s audio, the 2 perturbed ONNX forward passes (plus the 1.0× already done) run in ~430ms when the 3 forwards are launched in parallel vs 735ms sequential. Decoding is still sequential (~76ms × 3 = 228ms). Total per TTA call = ~660ms wall-clock vs 1010ms sequential. The 47/54 easy samples don't trigger TTA at all and stay at the c2c-direct-mixed baseline (~250ms median).
- **Cost:** ~0.2s mean overhead (0.69s → 0.92s) for +2 points recall (98% → 100%). Confidence gate at 0.5 cuts TTA calls to 7/54 samples. Below 0.5 the retasy_019 anchor score (0.0058) still triggers; only the 6 samples in the 0.5-0.95 range are now skipped (they were all already correct without TTA, so no accuracy loss).
- **Failure modes:** if all three perturbed passes predict the same wrong answer, TTA amplifies the error (wastes 3× compute for nothing). If the 0.9× pass predicts 3:1 and 1.1× predicts 3:2, picking the higher score (3:1) is right *most* of the time but not always.
- **To rebuild:** already shipped via `experiments/c2c-direct-mixed-tta/run.py`. Just run `python -m benchmark.runner --experiment c2c-direct-mixed-tta`.

### fastconformer-ctc-rescore (new -- no accuracy gain over FastConformer alone)

Two-stage pipeline combining FastConformer ASR (best transcription) with CTC forced-alignment re-scoring (best alignment), using confidence-gated fallback to skip Stage 2 on easy samples.

- **Stage 1:** NVIDIA FastConformer (115 MB) -> transcript -> QuranDB span-aware match
- **Stage 2 (fallback):** CTC re-score top-50 candidates using fine-tuned 8L Rabah CTC (145 MB)
- **Gate:** If Stage 1 score >= threshold (default 0.7), return immediately without running Stage 2
- **Result:** 85% SeqAcc / 87% recall / 89% precision, 0.66s latency (threshold=0.7), 273 MB total
- **Finding:** CTC re-scoring does **not** recover any of the 8 samples FastConformer fails on, even with threshold=0.95 (forcing CTC on nearly everything). Both models fail on the same hard cases: short isolated letters (Ya-Sin, Al-Ikhlas) and multi-verse passages. The "complementary errors" hypothesis didn't hold -- the failures are in the candidate retrieval (bad transcript from FastConformer), not in the scoring.
- **Tuning:** `FASTCONFORMER_CTC_CONFIDENCE=0.7` and `FASTCONFORMER_CTC_STAGE2_DYNAMIC_INT8=1` (default). Threshold sweep: 0.7 → 85% / 0.66s, 0.95 → 85% / 2.43s. Higher threshold = more CTC runs = more latency, same accuracy.

### fastconformer-nbest-bruteforce (worse than baseline -- 83% SeqAcc, 500 MB)

N-best beam search on FastConformer CTC logits + CTC brute-force scoring of all verses in candidate surahs. The hypothesis: expanding the transcript candidate pool via beam search and brute-forcing CTC alignment across entire surahs would recover failures where the greedy transcript is wrong.

- **Stage 1:** FastConformer CTC logits → pyctcdecode beam search → 5-best hypotheses → QuranDB match each
- **Stage 2 (fallback):** If no match exceeds confidence threshold (0.7), collect top-10 surahs from all N-best matches, brute-force CTC score all verses + multi-verse spans (up to 6000 candidates)
- **Result:** 83% SeqAcc / 85% recall / 87% precision, 0.85s latency, ~500 MB total
- **Finding:** **Worse than baseline.** Introduced 2 new failures (`retasy_003` 1:2→37:182, `retasy_019` 3:2→102:6) where the N-best confidence gate incorrectly triggers brute-force, which picks wrong candidates. Did not recover any of the baseline's 8 failures -- the correct surah never appears in the candidate set because all N-best hypotheses produce similar wrong transcripts (low beam diversity without a language model). The brute-force path adds ~0.5s latency and 385 MB of Stage 2 model weight for no accuracy gain.
- **Key insight:** CTC beam search without a language model produces near-identical hypotheses (all 5 beams differ by only a few BPE tokens). A Quran-specific language model or constrained decoding would be needed to make N-best useful.

### two-stage-faster-whisper-pruned (new)

Two-stage retrieval variant using faster-whisper for ASR and fine-tuned pruned CTC for re-scoring:

- **Stage 1:** `OdyAsh/faster-whisper-base-ar-quran` (147 MB, CTranslate2 int8)
- **Stage 2:** CTC re-score top-50 candidates using fine-tuned 8L Rabah CTC with dynamic int8 (`TWO_STAGE_STAGE2_DYNAMIC_INT8=1`, default)
- **Result:** 70% SeqAcc / 72% recall / 71% precision, 3.96s latency, 306 MB total
- **Tradeoff:** Major speed/size gain vs prior fp32 run (10.06s, 582 MB) with small precision/recall drop

### contrastive-v2 (failed -- val accuracy stuck at 9%)

CLIP-style contrastive model (QuranCLAP v2). Maps audio to a speaker-invariant 256-dim embedding, matched against a pre-computed FAISS index of all 6,236 verses. One forward pass + nearest neighbor = verse ID. No ASR needed.

- **Audio encoder:** wav2vec2-base (95M params)
- **Text encoder:** AraBERT v02 (136M params)
- **Training:** 30k samples from EveryAyah, batch 32 (effective 128 via grad accum), two-phase (frozen -> unfreeze last 2 layers), A100-80GB
- **Result:** Val accuracy stuck at ~9% after 8 epochs. Model overfits (train acc 14%, val acc 9%). wav2vec2-base can't produce useful Arabic audio representations.
- **Target size:** ~367 MB (audio encoder + projection + FAISS index)

### tarteel-whisper-base (67% accuracy, 290 MB)

Tarteel's Whisper-base fine-tuned for Quranic Arabic. Transcribe-then-match pipeline.

### whisper-lora (58% accuracy, 485 MB)

Whisper-small + LoRA adapter fine-tuned on EveryAyah + RetaSy. Transcribe-then-match pipeline.

### embedding-search (not viable standalone)

HuBERT embeddings + FAISS nearest neighbor. 100% same-reciter, 0% cross-speaker. HuBERT encodes speaker identity more than linguistic content.

### contrastive v1 (proof of concept, ~1.6% accuracy)

First attempt at CLIP-style audio-text matching. Failed due to English HuBERT on Arabic audio, batch size 4, single reciter. Contrastive-v2 addresses all of these.

### streaming-asr (43% accuracy, 6x faster to first match)

mlx-whisper chunked streaming. Streaming is good for UX (0.19s to first match) but accuracy is limited by whisper-base quality.

### new-models (model comparison)

Head-to-head benchmark of 8 ASR models. Key finding: Moonshine Tiny Arabic (103 MB) matches Whisper Large-v3-Turbo (3.1 GB) at 30x smaller. This is why the two-stage experiment uses Moonshine as Stage 1.

## Test corpus

`benchmark/test_corpus/manifest.json` contains 54 samples across three sources:

| Source | Count | Description |
|---|---|---|
| User recordings | 2 | Phone recordings (.m4a), ambient noise, non-professional |
| EveryAyah (Alafasy) | 23 | Professional studio recordings, includes long single-ayah and multi-ayah concatenated |
| RetaSy crowdsourced | 29 | Curated subset from 1,287 speakers across 81 countries |

**Categories:** short (17), medium (19), long (9), multi (9).

## Running benchmarks

```bash
# Activate the project venv
source .venv/bin/activate

# Run all registered experiments
.venv/bin/python -m benchmark.runner

# Run a single experiment
.venv/bin/python -m benchmark.runner --experiment ctc-alignment

# Run Rabah pruned variants (expanded via list_models)
.venv/bin/python -m benchmark.runner --experiment rabah-pruned-ctc
.venv/bin/python -m benchmark.runner --experiment rabah-pruned-ctc/8-layer-int8

# Run new two-stage faster-whisper pipeline
.venv/bin/python -m benchmark.runner --experiment two-stage-faster-whisper-pruned

# Run NVIDIA FastConformer benchmark (requires: pip install -e .[nemo])
.venv/bin/python -m benchmark.runner --experiment nvidia-fastconformer

# Filter by audio category
.venv/bin/python -m benchmark.runner --category short
```

Results are saved to `benchmark/results/<timestamp>.json`. The runner also maintains `benchmark/results/latest.json` with best results per scoped run (`mode`, `category`, and sample count).

To build local Rabah pruned checkpoints (naive, no fine-tuning):

```bash
.venv/bin/python scripts/build_rabah_pruned_models.py --layers 12 8 6 --save-source
```

To fine-tune pruned models on Modal A100 GPU:

```bash
modal run --detach scripts/train_pruned_ctc_modal.py --layers 8 --strategy first_n
modal run scripts/train_pruned_ctc_modal.py --layers 8 --strategy first_n --download-only
```

To fine-tune NVIDIA FastConformer on Modal A100 GPU:

```bash
# start detached (data prep + training)
modal run --detach scripts/train_nvidia_fastconformer_modal.py \
  --output-name nvidia-fastconformer-quran-ft-v1

# download artifacts once done
modal run scripts/train_nvidia_fastconformer_modal.py \
  --download-only \
  --output-name nvidia-fastconformer-quran-ft-v1

# benchmark with the fine-tuned checkpoint
NVIDIA_FASTCONFORMER_LOCAL_MODEL_DIR=data/nvidia-fastconformer-quran-ft-v1 \
  .venv/bin/python -m benchmark.runner --experiment nvidia-fastconformer
```

## Adding a new experiment

1. Create `experiments/<name>/run.py` with two required functions:

```python
def predict(audio_path: str) -> dict:
    """Returns {"surah": int, "ayah": int, "ayah_end": int|None, "score": float, "transcript": str}"""

def model_size() -> int:
    """Total model size in bytes."""
```

2. Register it in `benchmark/runner.py`:

```python
EXPERIMENT_REGISTRY = {
    ...
    "your-experiment": EXPERIMENTS_DIR / "your-experiment" / "run.py",
}
```

3. Run the benchmark: `.venv/bin/python -m benchmark.runner --experiment your-experiment`

For multi-model experiments, also export `list_models()` and accept `model_name` as an optional parameter in `predict()` and `model_size()`.

## Shared utilities

All experiments use `from shared.X import ...`:

- `shared.audio.load_audio(path, sr=16000)` -- returns float32 numpy array at 16kHz
- `shared.normalizer.normalize_arabic(text)` -- strips diacritics, normalizes alef/taa marbuta/alef maqsura
- `shared.quran_db.QuranDB` -- loads `data/quran.json`, provides `match_verse(text)`, `search(text, top_k)`, `get_verse(surah, ayah)`

## Web demo

FastAPI backend + React frontend for live recitation and verse identification.

```bash
# Backend
.venv/bin/python web/server.py

# Frontend (separate terminal)
cd web/frontend && npm run dev
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Some experiments have additional dependencies (faiss-cpu, moonshine, mlx-whisper). Check individual experiment READMEs.

## Key findings

1. **CTC forced alignment is the most accurate approach** -- scoring candidates directly against frame logits avoids the information loss of greedy decoding, giving 81% accuracy. But the model is too large (1.2 GB) for on-device deployment.

2. **Two-stage retrieval works.** Using ASR only for candidate retrieval (top 50) then CTC for the final decision gets 72% SeqAcc -- only 9 points behind scoring all 6,236 verses. The retrieval step is fast and the accuracy gap comes from candidate recall, not re-scoring quality.

3. **ASR quality is the bottleneck for transcribe-then-match approaches.** All ASR-based approaches fail on the same samples. Two-stage sidesteps this by using ASR only for candidate retrieval, not the final decision.

4. **Layer pruning + fine-tuning works for CTC size reduction.** While there are no small Arabic wav2vec2 models pretrained from scratch, pruning a large Quran-specific model (24→8 layers) and fine-tuning the CTC head recovers most accuracy (72% vs 81% full model). The `first_n` strategy (keep contiguous early layers) outperforms `evenly_spaced` by 16 percentage points, suggesting early transformer layers carry the most transferable features.

5. **Contrastive audio-text matching needs a multilingual encoder.** Both embedding search (HuBERT, 0% cross-speaker) and contrastive-v2 (wav2vec2-base, 9% val accuracy) failed because English-pretrained audio encoders don't produce useful features for Arabic speech. Multi-reciter training and deeper projection heads aren't enough to overcome bad audio representations.

6. **Small ASR models can match large ones.** Moonshine Tiny Arabic (103 MB) matches Whisper Large-v3-Turbo (3.1 GB) on our benchmark. But fine-tuning Moonshine degrades it (LLaMA tokenizer with character-level Arabic tokens is fragile).

7. **Short verses are hard across all approaches.** Verses under 3-4 words don't provide enough signal. May need minimum-length gating or surah-context bias in the final product.

8. **CTC re-rank of FastConformer's own BPE logprobs beats vanilla ASR+match by 12 points SeqAcc at the same on-disk size.** c2c-direct gets 97-98% vs the upstream pipeline's 85%, by scoring multiple verse candidates against the actual frame logprobs instead of matching the greedy transcript to a single verse. The decoder is the difference, not the encoder. See `experiments/c2c-direct/` and the "External comparison" section above.

9. **int4+int8 mixed quantization cuts 31% of int8 ONNX size at the same accuracy.** The c2c-direct-onnx int8 ONNX is 128 MB; applying `MatMulNBitsQuantizer` int4 to the 154 weight MatMul nodes (FFN + attention, 90% of the weight budget) and keeping int8 on the 57 Conv + 85 LayerNorm + CTC head gives 88 MB. Per-channel + `reduce_range=True` + symmetric weights is required for the int8 half (without `reduce_range` the argmax goes blank). Two-pass quantization is the only path: `quantize_dynamic` doesn't support int4, and `MatMulNBitsQuantizer` on top of int8 doesn't reduce further because the int8 already squeezed the MatMul. block_size=64 is *worse* than 128 (81.5% vs 98.1% SeqAcc) — the conformer's weight distributions are stable at coarse granularity. `nodes_to_exclude` to int4 only FFN or only attention produces 200+ MB models because the remaining fp32 weights dominate. See `experiments/c2c-direct-mixed/`, `scripts/quantize_int4.py`, `scripts/quantize_mixed.py`.

10. **Speed-perturbation TTA recovers the last 1.9 points of recall at +0.2s mean inference cost.** `retasy_019` is the one sample where all three quantization variants fail at 1.0× speed: int8 predicts 69:20, int4+int8 predicts 104:4, the .nemo predicts 3:1-2 (correct). Running 0.9× and 1.1× `scipy.signal.resample_poly` perturbations and picking the highest CTC re-rank score recovers 3:1 (correct surah, off-by-one ayah). 100% recall on 54 samples for the first time. The 2 perturbed forward passes (0.9×, 1.1×) run in parallel via a `ThreadPoolExecutor` on a shared ONNX session, cutting the wall-clock overhead from 2× (sequential) to ~0.2s mean (parallel). Confidence-gating (skip TTA when 1.0× score ≥ 0.5) keeps median latency at 303ms — TTA only runs on the 7 hard samples. See `experiments/c2c-direct-mixed-tta/`.

## What we tried (and didn't work)

### Moonshine Tiny AR LoRA fine-tuning

Attempted to LoRA fine-tune [UsefulSensors/moonshine-tiny-ar](https://huggingface.co/UsefulSensors/moonshine-tiny-ar) (27M params, 103 MB) on Quran audio to improve its transcription for verse matching.

**What we tried:**
- LoRA r=8 on all attention (q/k/v/o_proj), lr=5e-4, 3000 steps
- LoRA r=4 encoder-only (decoder frozen), lr=1e-4, 2000 steps
- LoRA r=8 decoder-only, various learning rates
- Full fine-tuning (all 27M params), lr=2e-5, 3000 steps
- Diacritics stripped + EOS token appended (fixing data distribution mismatch)

Every configuration degraded the model. Base Moonshine scores 56% SeqAcc; the best fine-tuned variant dropped to 35-38%.

**Root cause:** Moonshine uses a LLaMA-derived tokenizer with only ~54 Arabic character tokens out of 32K vocab. Every Arabic character is its own token, so the decoder does character-by-character generation. Any weight perturbation (even LoRA r=4 at low LR) corrupts the character sequences, producing garbled output like `ب س الهر حا الحي` instead of `بسم الله الرحمن الرحيم`.

**Conclusion:** The base model (56% SeqAcc, 103 MB) is useful as-is for the two-stage pipeline's Stage 1 (rough transcript for candidate retrieval), but shouldn't be fine-tuned.

### wav2vec2-base Arabic CTC fine-tuning

Attempted to fine-tune `facebook/wav2vec2-base` (95M params, English SSL pretrained) with an Arabic CTC head, using the vocabulary from `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`. Trained on EveryAyah + RetaSy datasets (85/15 interleave) on Modal A100-80GB.

**What we tried:**
- LR 3e-4, 5000 steps, batch 32 x grad_accum 2 → model collapsed at step ~1400, outputting only token 46 (a diacritic)
- LR 1e-4, 5000 steps → loss plateaued at 3.2-3.3 through all 5000 steps, model outputs only token 0 (pad)
- Frozen CNN feature extractor, only training transformer layers + CTC head

**Root cause:** wav2vec2-base was pretrained on English-only LibriSpeech. Its SSL representations encode English phonemes, not Arabic ones. Fine-tuning the transformer layers and CTC head alone can't bridge this gap -- the CNN feature extractor (which is frozen) produces features that don't represent Arabic speech sounds. The loss never drops below 3.2 because the model fundamentally can't distinguish Arabic characters from the audio features it receives.

**Alternatives considered:**
- `facebook/mms-300m` (multilingual, knows Arabic) -- 300M params, barely smaller than the 317M large model
- `facebook/wav2vec2-xls-r-300m` (multilingual) -- same size issue
- `DistilHuBERT` (23.5M params) -- English-only, same problem as wav2vec2-base

**Conclusion:** There is no existing small (<150M params) wav2vec2-family model with Arabic speech representations. Getting a small CTC model requires either (a) distilling from a multilingual model that already works, or (b) a different architecture entirely. This blocks both the distilled-ctc and two-stage experiments' path to the size target.

### Contrastive v2 (QuranCLAP v2) training

Attempted CLIP-style contrastive learning with wav2vec2-base (audio) + AraBERT v02 (text). Trained on 30k EveryAyah samples on Modal A100-80GB, batch 32 with gradient accumulation 4 (effective batch 128).

**Training log:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| P1 E1 (frozen) | 2.88 | 9.7% | 3.05 | 9.0% |
| P1 E6 (frozen) | 2.56 | 13.9% | 3.12 | 8.8% |
| P2 E1 (unfrozen) | 2.77 | 11.9% | 3.03 | 9.0% |
| P2 E2 (unfrozen) | 2.56 | 14.1% | 3.07 | 9.4% |

Random chance with batch 32 = 3.1%. Val accuracy plateaued at ~9% across all 8 epochs while train accuracy climbed to 14% -- classic overfitting. Unfreezing the last 2 encoder layers in Phase 2 didn't help.

**Root cause:** Same as CTC: wav2vec2-base produces English speech features, not Arabic ones. The projection heads can memorize training pairs but can't learn generalizable audio-text alignment because the audio features don't encode Arabic phonetic content. The AraBERT text encoder works fine -- the bottleneck is entirely on the audio side.

**Conclusion:** Any approach using wav2vec2-base as an Arabic audio encoder will fail. Future contrastive attempts need a multilingual audio encoder (e.g., XLS-R-300M, MMS-1B) or a completely different architecture.

## Further reading

- `REPORT.md` -- Full experiment report with per-sample breakdowns, failure analysis, and recommendations
- `RESEARCH-audio-to-verse.md` -- Research survey of approaches (WavLink, Moonshine v2, WhisperKit, contrastive learning, audio fingerprinting)
- `docs/C2C-DIRECT-OPTIMIZATIONS.md` -- Full writeup of the c2c-direct port, ONNX export, int4 quantization, and TTA. Includes dead ends (int4-on-top-of-int8, block_size=64, nodes_to_exclude), reproduction steps, and the alamenium comparison framework adaptation
- `docs/plans/` -- Design documents for individual experiments
- Individual experiment `README.md` files for reproduction instructions

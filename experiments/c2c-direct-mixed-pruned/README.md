# c2c-direct-mixed-pruned

FFN-pruned FastConformer (dim 2048 → 1024) with mixed int4+int8 quantization.

**Status: regression vs c2c-direct-mixed.** This experiment produced a smaller
ONNX (69 MB vs 88 MB) but the pruning + fine-tuning could not recover the
original accuracy (78% SeqAcc vs 98% for the unpruned c2c-direct-mixed).

## What happened

1. **Prune FFN dim 2048 → 1024** (saves 36M params, 31% reduction)
2. **Fine-tune on Modal A100** for 5000 steps with frozen CNN + first 8/17
   transformer layers (final train_loss 0.80, 75 min)
3. **Re-export to ONNX** (316 MB fp32)
4. **Quantize int4+int8 mixed** (69 MB total)

The transcripts show degradation:
- `user_ikhlas_2_3` (Al-Ikhlas 112:2): "اللل الصمد ل ملد و" — typos
  "اللل" instead of "الله", "ملد" instead of "يلد" (extra/missing letters)
- `ref_036001` (Ya-Sin 36:1): "س" — only one letter decoded
- `retasy_019` (3:2): "ح" — only one letter decoded
- `multi_055_001_004` (55:1-4): "الرحمم القرا خلقإنسن" — multiple typos
  ("الرحمم" instead of "الرحمن", "القرا" instead of "القران", "خلقإنسن" instead of "خلق الإنسان")

The pruning destroyed the model's ability to reproduce precise character
sequences. The fine-tuning recovered some of the capability (CTC loss went
from raw 1.16 average to 0.80 final) but not enough.

## Results (54 samples, 15s chunks)

| Experiment | Recall | Precision | SeqAcc | Latency | Size |
|---|---|---|---|---|---|
| c2c-direct-mixed (unpruned) | 98% | 98% | 98% | 0.69s | 88 MB |
| **c2c-direct-mixed-pruned** | **80%** | **78%** | **78%** | **0.86s** | **69 MB** |

## Lessons

1. **FFN dim 1024 is too small** for FastConformer. 31% param reduction is too
   aggressive. The model loses character-level accuracy.
2. **5000 steps of fine-tuning isn't enough** to recover from a deep architectural
   change. The CTC loss converged but the argmax decoding is still garbled.
3. **Pruning the FFN breaks the BPE tokenization**. The model is over-confident
   on wrong tokens and under-confident on the right ones.
4. **The size reduction (22%) doesn't justify the accuracy drop (20pp)**.
   For phone deployment, the unpruned 88 MB is the right choice.

## Files

- `data/fastconformer-pruned-ffn1024/model_state_dict.pt` (316 MB) — the
  fine-tuned state dict, downloaded from Modal volume
  `ffn-pruned-fastconformer`
- `data/onnx_export/fastconformer_full_pruned.onnx` (317 MB) — fp32 ONNX
- `data/onnx_export/fastconformer_full_pruned_q4.onnx` (+ 110 MB data) — int4
- `data/onnx_export/fastconformer_full_pruned_mixed.onnx` (69 MB) — int4+int8
- `scripts/export_pruned_ffn_onnx.py` — re-export with loaded state dict
- `scripts/train_pruned_ffn_modal.py` — Modal training script (75 min, $3-4)
- `scripts/test_ffn_pruning.py` — local test of the structural pruning

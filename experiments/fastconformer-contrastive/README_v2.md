# FastConformer Contrastive v2 - Improved Training

This directory contains an improved training pipeline that addresses the gap between high training accuracy (~95%) and low benchmark accuracy (~12%).

## The Problem

The original training had a critical flaw: it measured **in-batch accuracy** (distinguishing between 32 items in a batch) rather than **global retrieval accuracy** (finding the correct ayah among 6,236 possibilities). These are completely different difficulty levels.

### Key Improvements in v2

1. **Hard Negative Mining**
   - Phonetically similar ayahs are automatically identified and used as negatives
   - Uses n-gram overlap to find ayahs that sound similar but are different
   - Example: "بِسْمِ اللَّهِ" vs similar-sounding phrases from different surahs

2. **Global Retrieval Validation**
   - During training, we now measure true top-1 and top-5 accuracy against ALL ayahs
   - This gives us an honest assessment of model performance
   - Saved as `val_global_top1_acc` and `val_global_top5_acc` in logs

3. **Curriculum Learning**
   - Starts with 30% hard negatives, ramps up to 90% over training
   - Model first learns easy distinctions, then tackles harder cases
   - Controlled by `hard_negative_prob` parameter

4. **Larger Effective Batch Size**
   - Gradient accumulation (default 4 steps) simulates batch size 128
   - More negatives per anchor = better contrastive learning

5. **Proper Train/Val Split**
   - Uses random split instead of reciter-based (which was too restrictive)
   - Ensures validation set mirrors training distribution

## Training

### Prerequisites

Ensure you have pre-extracted features from the original training:

```bash
# If you haven't extracted features yet:
python train.py --extract-features \
    --audio-dir /path/to/everyayah/audio \
    --output-dir ./checkpoints
```

### Train v2 Model

```bash
python train_v2.py --train \
    --feature-dir ./checkpoints/features \
    --output-dir ./checkpoints \
    --epochs 100 \
    --batch-size 32 \
    --negatives 5 \
    --grad-accum 4 \
    --lr 1e-4
```

### Key Parameters

- `--negatives 5`: Number of hard negatives per positive sample
- `--grad-accum 4`: Accumulate gradients over 4 steps (effective batch size 128)
- `--no-curriculum`: Disable curriculum learning (start at 90% hard negatives immediately)

## Expected Results

With the improvements, you should see:

| Metric | Before (v1) | After (v2, expected) |
|--------|-------------|---------------------|
| Train Accuracy | 95% (batch) | ~70-80% (global) |
| Val Global Top-1 | Not measured | 40-60% |
| Val Global Top-5 | Not measured | 70-85% |
| Benchmark SeqAcc | 11.8% | 40-60% |

**Note**: The "training accuracy" will drop because we're now measuring global retrieval, not batch classification. This is expected and actually indicates honest training.

## Monitoring Training

Check the training logs:

```bash
tail -f checkpoints/training_v2_log.json
```

Key metrics to watch:
- `val_global_top1_acc`: Should steadily increase
- `val_global_top5_acc`: Should reach 70%+ by epoch 50
- `train_loss`: Should decrease (may fluctuate with hard negatives)

## How Hard Negatives Work

### Phonetic Similarity

The system computes 3-gram overlap between normalized Arabic texts:

```python
# Example: Surah 1:1 vs a phonetically similar phrase
"بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"
vs
"بِسْمِ اللَّهِ"  # Shortened form appears elsewhere

Similarity score: ~0.6 (high overlap of character 3-grams)
```

### Mining Process

1. Pre-compute phonetic similarity between all ayah pairs
2. For each training sample, find top-20 most similar ayahs
3. During training, sample 5 hard negatives per positive
4. Mix hard negatives with random negatives (curriculum-based ratio)

## Architecture Changes

No changes to the model architecture - only the training procedure:

```python
# Same model, better training
Audio → FastConformer (frozen) → Projection → [256-d] → L2 Normalize
Text  → AraBERT (frozen)      → Projection → [256-d] → L2 Normalize

Loss: InfoNCE with hard negatives
```

## Inference

The `run.py` script automatically detects and loads the v2 model if available:

```python
# Prefers best_model_v2.pt over best_model.pt
CHECKPOINT_PATH = _v2_path if _v2_path.exists() else _v1_path
```

Benchmark as usual:

```bash
python -m benchmark.runner --experiment fastconformer-contrastive
```

## Troubleshooting

### Training Loss Not Decreasing

- Check that phonetic index was created: `ls checkpoints/phonetic_index.pkl`
- Reduce learning rate: `--lr 5e-5`
- Increase warm-up: modify `warmup_steps` in train_v2.py

### Low Global Accuracy

- Ensure hard negatives are being used (check logs for `hard_neg_prob`)
- Increase negatives: `--negatives 10`
- Train longer: `--epochs 150`

### Out of Memory

- Reduce batch size: `--batch-size 16`
- Reduce negatives: `--negatives 3`
- Reduce chunk size in validation (modify `chunk_size` in train_v2.py)

## Future Improvements

1. **Dynamic Hard Negative Mining**: Mine negatives from model's own errors during training
2. **Multi-scale Training**: Train on different audio lengths (1s, 3s, 5s, full)
3. **Data Augmentation**: Add noise, speed perturbation, reverb to simulate RetaSy quality
4. **Ensemble**: Average predictions from multiple checkpoints
5. **Test-time Augmentation**: Multiple forward passes with different crops

## References

- Contrastive Learning: [CLIP](https://arxiv.org/abs/2103.00020)
- Hard Negative Mining: [XBM](https://arxiv.org/abs/1912.06798)
- Curriculum Learning: [CurriculumNet](https://arxiv.org/abs/1804.00296)

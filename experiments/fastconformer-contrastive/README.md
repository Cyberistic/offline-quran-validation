# FastConformer Contrastive Learning

A contrastive learning approach that maps Quranic audio directly to verse embeddings using NVIDIA's Arabic FastConformer encoder.

## Why This Approach?

**Problem with previous contrastive attempts:**
- Used HuBERT (English speech model) → 0% accuracy on Arabic
- Frozen encoder couldn't capture Arabic phonetics

**This approach:**
- Uses FastConformer (Arabic-pretrained) → Already understands Arabic speech
- Freeze encoder, train 256-d projection layers
- Direct audio-to-ayah retrieval (no ASR transcription)

## Architecture

```
Audio (16kHz) → FastConformer encoder (frozen) → mean pool → projection MLP → [256-d] → L2 normalize
                                                                                              |
                                                                                         cosine similarity
                                                                                              |
Arabic verse text → AraBERT encoder (frozen) → CLS token → projection MLP → [256-d] → L2 normalize
```

**Key differences from ASR approach:**
- No transcription errors (no beam search, no token decoding)
- Direct embedding comparison (6,236 candidates)
- Learns ayah-level discrimination, not character-level

## Files

| File | Description |
|------|-------------|
| `model.py` | FastConformerContrastive model with projection heads |
| `dataset.py` | Dataset loaders for audio-text pairs |
| `train.py` | Training script (2-phase: feature extraction + projection training) |
| `run.py` | Inference script with predict() and model_size() exports |
| `README.md` | This file |

## Training

### Step 1: Extract Features (One-time)

Since the FastConformer encoder is frozen, pre-extract features for all training audio:

```bash
cd experiments/fastconformer-contrastive

python train.py \
  --extract-features \
  --audio-dir /path/to/quran/audio \
  --feature-dir ./features \
  --device cuda
```

This creates `.pt` files containing [batch, seq_len, 512] encoder outputs.

### Step 2: Train Projection Heads

```bash
python train.py \
  --train \
  --feature-dir ./features \
  --output-dir ./checkpoints \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --device cuda
```

**Training details:**
- Only projection layers are trainable (~525K params)
- Contrastive loss: InfoNCE (CLIP-style)
- 80/20 train/val split
- Cosine annealing LR schedule

## Inference

```python
from run import predict, model_size

# Predict on audio file
result = predict("/path/to/audio.wav")
print(f"Surah {result['surah']}, Ayah {result['ayah']}, Score: {result['score']:.4f}")

# Get model size
size_bytes = model_size()
print(f"Model size: {size_bytes / (1024*1024):.1f} MB")
```

## Expected Results

Based on the hypothesis:

| Metric | Target | Expected |
|--------|--------|----------|
| Recall | 95%+ | 93-95% (+2-4% over ASR) |
| Sequence Accuracy | 90%+ | 90-93% |
| Latency | <1s | ~0.5s (single forward pass) |
| Model Size | <200MB | ~120MB (FastConformer + projections) |

**Why +2-4% improvement over ASR:**
- Removes beam search errors
- No token-level transcription mistakes
- Direct embedding comparison vs text matching

## Advantages Over ASR

1. **No transcription errors** - Audio goes directly to embedding space
2. **Robust to noise** - Embedding space is learned, not rule-based
3. **Fast inference** - Single forward pass + cosine similarity
4. **Compact** - Only 256-d embeddings per ayah (vs full text)

## Challenges

1. **Requires training** - Unlike pretrained ASR, needs contrastive training
2. **Batch size limits** - Contrastive learning needs large batches (32+) for effective negatives
3. **Data diversity** - Needs multiple reciters to generalize (not just Alafasy)
4. **Cold start** - First run pre-computes all 6,236 ayah embeddings

## Data Requirements

**Minimum:**
- 6,236 ayah audio-text pairs (single reciter)
- ~30 minutes training on A100 GPU

**Optimal:**
- 6,236 ayahs × 10+ reciters = 62K+ pairs
- Hard negative mining (same surah, adjacent verses)
- 50-100 epochs, batch size 64+
- 2-4 hours on A100 GPU

## Comparison

| Approach | Recall | Latency | Size | Training |
|----------|--------|---------|------|----------|
| ASR (FastConformer) | 91% | 0.38s | 115MB | None (pretrained) |
| **This (Contrastive)** | **93-95%?** | **~0.5s** | **~120MB** | **50-100 epochs** |
| CTC Alignment | 83% | 3.2s | 1.2GB | Fine-tuned |

## Next Steps

To maximize performance:

1. **Multi-reciter training** - Use Quran-MD dataset (30+ reciters)
2. **Hard negative mining** - Sample negatives from same surah
3. **Larger batch sizes** - 64-256 for better contrastive learning
4. **Data augmentation** - Speed, pitch, noise
5. **Longer training** - 100+ epochs
6. **Unfreeze encoder** - Phase 2: unfreeze last 2 layers

## References

- NVIDIA FastConformer: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_ar_fastconformer_hybrid_large_pcd
- CLIP: https://arxiv.org/abs/2103.00030
- CLAP: https://arxiv.org/abs/2206.04769

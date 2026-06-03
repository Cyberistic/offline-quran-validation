# c2c-direct-mixed-streaming-v4

Hybrid approach: streaming pipeline (like v1) for live feedback, then validate
the first emission with a full-audio c2c-direct predict (like the offline
c2c-direct-mixed).

## Algorithm

1. Run `c2c-direct.predict()` on the entire audio (0.5-0.7s, 98% accuracy)
2. Run streaming tracker on the audio (0.5s, 78% accuracy) — this provides
   real-time emissions for the web demo's live UI
3. Combine: take offline's first verse (high confidence), then add streaming
   emissions that are sequentially next to the last emitted verse
4. Skip streaming emissions that aren't sequential (they're noise)

## Results (54 samples)

| Mode | Recall | Precision | SeqAcc | Latency |
|---|---|---|---|---|
| Streaming only (v1) | 81% | 86% | 78% | 0.52s |
| **v4 hybrid (streaming + offline)** | **85%** | **96%** | **80%** | **2.06s** |
| v4 with streaming disabled (offline only) | 98% | 97% | 96% | 0.69s |
| Offline c2c-direct-mixed | 98% | 98% | 98% | 0.69s |

The hybrid v4 is 2pp better than v1 at the cost of 4x latency. The streaming
adds latency and the sequential filter rejects most streaming emissions as
noise, so the result is essentially "offline's first verse only" for multi-verse
clips. Not a clear win.

For real-world phone deployment, the right design is:
- Streaming for live UI feedback during recording (shows best-effort guess)
- After the audio ends, run c2c-direct predict on the full audio for the
  final answer (98% accurate)

v4 implements this: `predict()` returns the combined emissions, but the
streaming part can also be consumed in real-time via the pipeline directly.

## Why the gap from 81% to 98%?

The streaming tracker is the bottleneck. It misses multi-verse boundaries
when chunks are too short and over-emits random verses on long clips.
Closing this gap requires either:
- A phoneme-level verifier (r15 model — blocked: HF gated)
- Word-level primary alignment (upstream tried, mixed results)
- More context per chunk (slower latency)

Without those, the streaming plateau is ~78% SeqAcc.

## Configuration

- `STREAMING_V4_CHUNK_SECONDS=15.0` — chunk size for streaming part
- `STREAMING_V4_DISABLE=1` — skip streaming, use offline only (96% SeqAcc, 0.69s)

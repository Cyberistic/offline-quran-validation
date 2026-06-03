"""c2c-direct-mixed-streaming — same FastConformer ONNX as c2c-direct-mixed
in chunked streaming mode, with deferred-emission tracker.

EMPIRICAL RESULTS on 54 samples (c2c-direct-mixed backend):

| Chunks | Recall | Precision | SeqAcc | Latency | Notes |
|--------|--------|-----------|--------|---------|-------|
| 3s     | 67%    | 59%       | 48%    | 0.65s   | Pre-deferred: 44% Seq |
| 5s     | 75%    | 73%       | 67%    | 0.59s   | |
| 10s    | 83%    | 87%       | 76%    | 0.73s   | Pre-deferred: 70% Seq |
| **15s**| **81%**| **86%**   | **78%**| **0.52s** | **Best — pre-deferred: 70%** |
| offline| 98%    | 98%       | 98%    | 0.69s   | Full-audio CTC re-rank |

## Deferred emission (upstream-aligned)

When the accumulator overflows a verse, the matched verse is held in
PENDING state instead of emitted immediately. Emission is confirmed
when:
  - A NEW (different) verse is matched with score >= 0.35, OR
  - `finalize()` is called (end-of-stream), OR
  - pending ages out (3 chunks max).

This prevents the cascading bogus next-verse emissions that hurt long
single-verse clips (upstream EXPERIMENTS.md: "v3 still 57 samples
where recall is perfect but SeqAcc is zero because the tracker emits
many verses after the correct one").

## Trade-offs

Best streaming variant is 15s chunks at 78% SeqAcc / 0.52s — first-match
within ~7-8s of audio, 20pp below offline. The gap to 90%+ SeqAcc
(upstream target) requires either:
  - A true streaming ASR with cache/segment design (WhisperStreaming,
    SimulWhisper) — out of scope.
  - The r15 utterance-level verifier (full-audio phoneme CTC, 96%
    SeqAcc on v3) — blocked on gated model access.
  - Word-level primary alignment gate (upstream TS tracker) — would
    need a significant Python tracker rewrite.
"""

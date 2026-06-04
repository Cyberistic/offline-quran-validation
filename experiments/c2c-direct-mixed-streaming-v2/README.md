"""c2c-direct-mixed-streaming-v2 — per-chunk c2c-direct.predict() calls.

Hypothesis: calling the full c2c-direct algorithm per chunk (including
CTC re-rank) would be more accurate than v1's fuzzy-match-only approach.

EMPIRICAL RESULTS on 54 samples:
  v1 (fuzzy match, 3s):  72% recall / 55% prec / 44% SeqAcc / 0.66s
  v1 (fuzzy match, 10s): 85% recall / 82% prec / 70% SeqAcc / 0.54s
  v2 (CTC re-rank, 7s):  64% recall / 72% prec / 59% SeqAcc / 1.06s
  v2 (CTC re-rank, 15s): 78% recall / 83% prec / 72% SeqAcc / 1.09s

CONCLUSION: v2 is NOT a clear win. v1 at 10s chunks is faster (0.54s vs
1.09s) with similar accuracy (70% vs 72%). The per-chunk CTC re-rank
is more expensive and only marginally more accurate because:
- Short chunks produce sparse transcripts
- Sparse transcripts give sparse candidate sets
- CTC re-rank can only re-rank what fuzzy match already found

Streaming chunked approaches plateau around 70-72% SeqAcc. To break
through to 90%+, we'd need a true streaming ASR (WhisperStreaming,
SimulWhisper, or similar) — out of scope for this experiment.

Best streaming baseline remains: c2c-direct-mixed-streaming @ 10s
chunks: 70% SeqAcc / 0.54s / 88 MB.
"""

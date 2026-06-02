"""c2c-direct-mixed-streaming — same FastConformer ONNX as c2c-direct-mixed
in chunked streaming mode.

Closes the gap to Yazin's v3 streaming baseline (89.3% recall / 58.2%
seq-acc on 256 samples).

EMPIRICAL RESULTS on 54 samples (3s chunks, c2c-direct-mixed backend):
  3s  chunks:  72% recall / 55% prec / 44% SeqAcc / 0.66s
  10s chunks:  85% recall / 82% prec / 70% SeqAcc / 0.54s
  offline:     98% recall / 98% prec / 98% SeqAcc / 0.69s

The streaming version is meaningfully worse than offline. The issue is
fundamental: chunked streaming uses greedy transcript → fuzzy match,
while c2c-direct's offline strength is CTC re-rank (which requires
the full log-prob matrix, not the greedy text).

To close the gap, future work would need:
  - Per-chunk CTC re-rank (each chunk gets its own ctc_loss
    scoring against the candidate verses; matches are accepted only
    when the per-chunk score crosses a threshold).
  - OR: a more confident ASR that doesn't need re-ranking.
  - OR: a verse detection method that operates on log-probs directly
    without going through a transcript (e.g., constrained beam search
    per chunk, with state carried across chunks).

The 10s-chunk variant (70% SeqAcc) is a useful streaming baseline. It
gives a streaming latency advantage: first-match within ~5s instead
of full audio. The 3s-chunk variant (44%) is too noisy to be useful
for our task.
"""

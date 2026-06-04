"""c2c-direct-trie — prefix-constrained CTC beam search + CTC re-ranking.

Tried as a candidate-generation upgrade over c2c-direct's greedy+fuzzy
match. The deepsearch ranked trie-constrained CTC beam search as the
SOTA for closed-vocab ASR, so we expected it to help with retasy_019
(where greedy decode picks an OOV path).

EMPIRICAL RESULT on 54 samples with the int4+int8 ONNX:
  c2c-direct-trie:   59% recall / 58% precision / 57% SeqAcc / 0.74s
  c2c-direct-mixed:  98% recall / 98% precision / 98% SeqAcc / 0.69s
  c2c-direct-mixed-tta: 100% recall / 99% precision / 98% SeqAcc / 0.91s

The trie beam search is **significantly worse** than c2c-direct-mixed
on the int4+int8 ONNX. The fundamental issue: the int4 quantization
changes the model's CTC log-probs enough that the prefix-constrained
search can't follow any coherent verse path. The beam search stays
near the root (mostly blank) and rarely reaches terminal nodes, even
with blank_penalty=-5 and beam_width=500.

WHY IT DOESN'T HELP (analysis):
  - The model's argmax is blank for ~95% of frames in the harder
    samples. Under a hard trie constraint, only the BPE tokens that
    exactly match a verse path are allowed, so the cumulative log-prob
    drops much faster than the all-blank "path of least resistance".
  - The c2c-direct algorithm's greedy transcript + fuzzy match is
    *more robust* to model uncertainty: it finds verse texts that
    approximately match the model's noisy output, then re-ranks.
  - The trie forces commitment to a specific path, which is brittle
    when the model itself is uncertain.

WHEN IT WOULD HELP (untested here, on a different model):
  - A more confident acoustic model with a lower entropy on token
    predictions (e.g., one specifically fine-tuned for recitation).
  - The .nemo model already gets 100% recall; the trie beam search
    would just need to pick the right sub-verse (retasy_019's
    off-by-1 ayah is the issue, but it's a close call between
    3:1+3:2 vs 3:2).

READYMADE LIBRARIES TRIED (all rejected):
  - flashlight-text (C++ CTC decoder via torchaudio.models.decoder.
    ctc_decoder): hardcoded 6-child-per-trie-node limit. Our trie has
    85+ branches at common prefixes (e.g., 113 verses share tokens
    [351, 7, 59, 982, 986] = "بسم الله الرحمن"). The limit is
    silent in the public API and the C++ binding doesn't expose a
    constructor to raise it.
  - pyctcdecode (pure Python prefix beam search): no true prefix
    constraint, only `hotwords` boosting. Search is unconstrained.
  - k2 (k2-fsa): install pulls 500+ MB of CUDA dependencies. The
    full sherpa-onnx build chain is more than we need.
  - sherpa-onnx: has OfflineCtcFstDecoderConfig (FST-based), but
    requires building an OpenFST graph by hand. The lex-format
    interface is harder to use than our flat trie.

LESSONS LEARNED (kept for future attempts):
  - Pure-prefix-constrained CTC beam search is dominated by the
    "all-blank" prior. The model assigns ~99% probability to blank
    at most frames; the cumulative log-prob of a verse path is
    exponentially lower.
  - To make this work, the search needs to score verse paths by
    something other than raw log-prob: length-normalized score,
    per-token marginal probability (forward-backward), or a
    confidence-gated fallback to c2c-direct's algorithm.
  - The cleanest fix: trie beam search only when the model's
    confidence is high (most likely already-correct single verse);
    fall back to c2c-direct's fuzzy match for uncertain samples.
    This is what c2c-direct-mixed-tta already does with TTA.

STATUS: research exploration, kept for the next attempt with
a different model or scoring. The benchmark runner still
recognizes this as an experiment so we can re-test on the
.nemo model later if desired.
"""

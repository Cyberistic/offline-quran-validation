# c2c-direct-phoneme

Phoneme-level FastConformer (from upstream yazinsai/offline-tarteel) +
Levenshtein match against `quran_phonemes.json` (5.3 MB verse reference).

## How it works

1. **Preprocessing**: Extract 80-dim log-mel features (NeMo-style: 25ms win,
   10ms hop, 16kHz, preemphasis 0.97)
2. **ONNX inference**: Run `fastconformer_phoneme_q8.onnx` (125 MB) →
   `[T, 70]` logprobs (68 phonemes + 1 word-boundary + 1 blank)
3. **Greedy decode**: Argmax + CTC collapse → phoneme-words like
   `"bismi allahi arraHmaani arraHiimi"`
4. **Multi-pass Levenshtein match** against 6,236 Quran verses:
   - Pass 1: `ratio()` global ranking + short-query prefix boost
   - Pass 1.5: `fragment_score` boost (disabled by default for speed)
   - Pass 2: multi-verse span matching on top 5 surahs

The upstream uses pyctcdecode beam search + KenLM. We added beam support and
built phoneme unigram/ngram assets locally, but both beam variants regressed
vs greedy on our 54-sample corpus.

## Results (54 samples, full mode)

| Config | SeqAcc | Latency | Notes |
|---|---|---|---|
| **Greedy + spans + no fragment** | **80%** | **0.60s** | best practical setting |
| Beam + custom phoneme n-gram LM | 78% | 0.52s | regresses vs greedy |
| Beam + unigrams | 78% | 0.54s | also regresses |
| Greedy + fragment boost | 83% | 3.97s | +3pp, but 6.6x slower |

For reference, the other c2c-direct variants:
- `c2c-direct-mixed` (BPE, int4+int8): 98% @ 0.69s, 88 MB
- `c2c-direct-mixed-streaming` (BPE, 15s chunks): 78% @ 0.52s, 88 MB

## Why phoneme still loses to c2c-direct-mixed

Best practical setting here is 85% recall / 89% precision / 80% SeqAcc. The
gap to c2c-direct-mixed (98%) comes from:

1. **No CTC re-rank on the phoneme output**. c2c-direct does
   `ctc_loss(candidate_text, log_probs)` for each candidate verse; we do
   Levenshtein text match. CTC re-rank catches ~10-15pp more.
2. **Empty transcripts on hard samples** (retasy_019, retasy_012, etc.).
   The phoneme model gives "ح" or "" for some audios that the BPE model
   gets right.
3. **Bismillah stripping is imperfect**. The model sometimes decodes
   "ama" instead of "alrHmaan" for 55:1 (because the audio starts with
   bismillah, and stripping gives the wrong word).

The 125 MB model is competitive with the 88 MB c2c-direct-mixed on
recall (87% vs 98%) but slower and worse on accuracy. The BPE
approach is the right choice for production.

## Beam search findings

Implemented in this session:

1. `scripts/build_phoneme_kenlm.py`
2. `data/kenlm_phoneme/quran_phoneme_unigrams.txt`
3. `data/kenlm_phoneme/phoneme_ngrams.json`
4. beam decoding in `run.py`

But both beam variants regressed:

- custom n-gram LM: 78% SeqAcc
- unigram beam: 78% SeqAcc

Likely cause: `pyctcdecode` warns that our alphabet has entries of length > 1
(`aa`, `SS`, `<<`, etc.). It is treating phoneme tokens as a non-BPE alphabet,
which does not line up well with its decoding assumptions. The probable fix is
to remap each phoneme token to a single surrogate symbol before beam decoding,
then map back after decode.

## Files

- `data/onnx_export/fastconformer_phoneme_q8.onnx` (131 MB) — from upstream
  yazinsai/offline-tarteel, downloaded via curl
- `data/quran_phonemes_web.json` (5.3 MB) — verse reference with phonemes
  + bismillah stripping metadata
- `data/kenlm_phoneme/quran_phoneme_unigrams.txt` — phoneme-word vocabulary
- `data/kenlm_phoneme/phoneme_ngrams.json` — custom phoneme bigram/trigram LM
- `scripts/build_phoneme_kenlm.py` — local LM asset builder

## Configuration

- `PHONEME_LM_USE_BEAM=1` — use pyctcdecode beam search
- `PHONEME_LM_USE_CUSTOM=1` — use custom phoneme n-gram LM JSON
- `PHONEME_LM_TOP_K=10` — top-K candidates returned
- `PHONEME_LM_TOP_SURAHS=5` — surahs to consider for multi-verse spans
- `PHONEME_LM_MAX_SPAN=2` — max verses per span (1 = single only)
- `PHONEME_LM_SPANS=1` — enable multi-verse span matching
- `PHONEME_LM_FRAGMENT=0` — enable fragment score boost (slow, marginal)
- `PHONEME_LM_KENLM_MODEL=<path>` — KenLM .arpa/.bin file (upstream path)

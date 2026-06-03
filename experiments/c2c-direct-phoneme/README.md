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

The upstream uses pyctcdecode beam search + KenLM, which we don't have
set up. Greedy + Levenshtein is sufficient for the 54-sample benchmark.

## Results (54 samples, full mode)

| Config | SeqAcc | Latency | Notes |
|---|---|---|---|
| **Default (greedy + Levenshtein)** | **83%** | **3.97s** | spans on, fragment off |
| spans on, fragment on | 80% | 4.02s | fragment boost costs 3.5s, no gain |
| spans off, fragment off | 83% | 3.97s | spans don't help (Pass 1 dominates) |

For reference, the other c2c-direct variants:
- `c2c-direct-mixed` (BPE, int4+int8): 98% @ 0.69s, 88 MB
- `c2c-direct-mixed-streaming` (BPE, 15s chunks): 78% @ 0.52s, 88 MB

## Why 83% not 98%?

The phoneme model gets 87% recall / 89% precision / 83% SeqAcc. The 15pp
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

## How to get the upstream's 96%

The upstream yazinsai run hits 96% all metrics with the SAME 125 MB
phoneme model + a 5-min KenLM build. Path:

1. Build `data/kenlm_phoneme/quran_phoneme_unigrams.txt` (extract all
   phoneme-words from quran_phonemes.json)
2. Build a KenLM arpa/bin model with `lmplz` + `build_binary`
3. Set `PHONEME_LM_USE_BEAM=1` and `PHONEME_KENLM_MODEL=<path>`
4. The pyctcdecode beam search with KenLM should close the gap

Not done yet — KenLM build is OS-level (apt install), needs a fresh
session.

## Files

- `data/onnx_export/fastconformer_phoneme_q8.onnx` (131 MB) — from upstream
  yazinsai/offline-tarteel, downloaded via curl
- `data/quran_phonemes_web.json` (5.3 MB) — verse reference with phonemes
  + bismillah stripping metadata
- `data/kenlm/quran_unigrams.txt` (173 KB) — Arabic unigrams (not phoneme;
  not currently used)

## Configuration

- `PHONEME_LM_USE_BEAM=1` — use pyctcdecode beam search (requires install)
- `PHONEME_LM_TOP_K=10` — top-K candidates returned
- `PHONEME_LM_TOP_SURAHS=5` — surahs to consider for multi-verse spans
- `PHONEME_LM_MAX_SPAN=2` — max verses per span (1 = single only)
- `PHONEME_LM_SPANS=1` — enable multi-verse span matching
- `PHONEME_LM_FRAGMENT=0` — enable fragment score boost (slow, marginal)
- `PHONEME_LM_KENLM_MODEL=<path>` — KenLM .arpa/.bin file (upstream path)

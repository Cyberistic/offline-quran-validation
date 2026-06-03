"""Phoneme-level FastConformer (from upstream yazinsai) + Levenshtein match.

Uses fastconformer_phoneme_q8.onnx (125 MB) — a phoneme-level CTC model
trained on the same EveryAyah + RetaSy data as our c2c-direct-mixed. Output
is 70-dim log-probs (68 phonemes + 1 word-boundary + 1 blank).

Pipeline:
1. Extract 80-dim log-mel features (NeMo-compatible: 25ms window, 10ms hop)
2. Run ONNX inference → [T, 70] logprobs
3. Greedy CTC decode → phoneme-words (e.g., "bismi allahi arraHmaani")
4. Multi-pass Levenshtein match against quran_phonemes.json:
   - Pass 1: global ratio() ranking + short-query prefix boost
   - Pass 1.5: fragment score boost for partial matches
   - Pass 2: multi-verse span matching on top surahs

The upstream uses pyctcdecode beam search + KenLM, but for our 54-sample
benchmark greedy + Levenshtein is sufficient and avoids the 5-min KenLM
build. Beam search is opt-in via PHONEME_LM_USE_BEAM=1.

Adapted from yazinsai/offline-tarteel experiments/fastconformer-phoneme-lm/run.py
(Apache-2.0).
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from pyctcdecode.language_model import AbstractLanguageModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ONNX_MODEL_PATH = PROJECT_ROOT / "data" / "onnx_export" / "fastconformer_phoneme_q8.onnx"
QURAN_PHONEMES_PATH = PROJECT_ROOT / "data" / "quran_phonemes_web.json"
UNIGRAMS_PATH = PROJECT_ROOT / "data" / "kenlm_phoneme" / "quran_phoneme_unigrams.txt"
NGRAMS_PATH = PROJECT_ROOT / "data" / "kenlm_phoneme" / "phoneme_ngrams.json"

# Must match the vocabulary used during training
PHONEME_VOCAB = [
    "a", "u", "i", "A", "U", "I", "aa", "uu", "ii", "AA", "UU", "II",
    "<", "b", "t", "^", "j", "H", "x", "d", "*", "r", "z", "s", "$",
    "S", "D", "T", "Z", "E", "g", "f", "q", "k", "l", "m", "n", "h", "w", "y",
    "<<", "bb", "tt", "^^", "jj", "HH", "xx", "dd", "**", "rr", "zz", "ss", "$$",
    "SS", "DD", "TT", "ZZ", "EE", "gg", "ff", "qq", "kk", "ll", "mm", "nn", "hh", "ww", "yy",
    "|",
]
BLANK_ID = len(PHONEME_VOCAB)  # 69

USE_BEAM = os.getenv("PHONEME_LM_USE_BEAM", "0").strip() == "1"
USE_CUSTOM_LM = os.getenv("PHONEME_LM_USE_CUSTOM", "1").strip() == "1"
TOP_K = int(os.getenv("PHONEME_LM_TOP_K", "10"))
TOP_SURAHS = int(os.getenv("PHONEME_LM_TOP_SURAHS", "5"))
MAX_SPAN = int(os.getenv("PHONEME_LM_MAX_SPAN", "2"))
ENABLE_SPANS = os.getenv("PHONEME_LM_SPANS", "1") == "1"
ENABLE_FRAGMENT = os.getenv("PHONEME_LM_FRAGMENT", "0") == "1"
_BSM_PHONEMES_JOINED = "bismi allahi arraHmaani arraHiimi"

_onnx_session = None
_verses = None
_by_surah: dict[int, list[dict]] = {}
_beam_decoder = None


def _read_unigrams(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_labels() -> list[str]:
    labels = []
    for token in PHONEME_VOCAB:
        labels.append(" " if token == "|" else token)
    labels.append("")
    return labels


class PhonemeNgramLM(AbstractLanguageModel):
    """Compact phoneme word-level n-gram LM for pyctcdecode."""

    def __init__(self, ngrams_path: Path, order: int = 3, smoothing: float = 0.1, weight: float = 0.3):
        data = json.loads(ngrams_path.read_text(encoding="utf-8"))
        self._unigrams = data["unigrams"]
        self._bigrams = data["bigrams"]
        self._trigrams = data.get("trigrams", {})
        self._total = data["total_words"]
        self._vocab_size = data["vocab_size"]
        self._order = order
        self._smoothing = smoothing
        self._weight = weight
        self._word_set = set(self._unigrams.keys())
        self._word_prefixes: set[str] = set()
        for w in self._word_set:
            for i in range(1, len(w) + 1):
                self._word_prefixes.add(w[:i])

    @property
    def order(self) -> int:
        return self._order

    def _ngram_logprob(self, context: tuple[str, ...], word: str) -> float:
        import math

        if len(context) >= 2:
            key = f"{context[-2]} {context[-1]}"
            tri = self._trigrams.get(key, {})
            total = sum(tri.values()) if tri else 0
            if total > 0:
                count = tri.get(word, 0)
                prob = (count + self._smoothing) / (total + self._smoothing * self._vocab_size)
                return math.log(prob)

        if len(context) >= 1:
            bi = self._bigrams.get(context[-1], {})
            total = sum(bi.values()) if bi else 0
            if total > 0:
                count = bi.get(word, 0)
                prob = (count + self._smoothing) / (total + self._smoothing * self._vocab_size)
                return math.log(prob)

        count = self._unigrams.get(word, 0)
        prob = (count + self._smoothing) / (self._total + self._smoothing * self._vocab_size)
        return math.log(prob)

    def score(self, prev_state: object, word: str, is_last_word: bool = False):
        context = prev_state if isinstance(prev_state, tuple) else ()
        logp = self._ngram_logprob(context, word) * self._weight
        new_state = (context + (word,))[-(self._order - 1):]
        return logp, new_state

    def score_partial_token(self, partial_token: str) -> float:
        if not partial_token:
            return 0.0
        if partial_token in self._word_prefixes:
            return 0.0
        return -10.0

    def get_start_state(self):
        return ()

    def reset_params(self, **params):
        pass

    def save_to_dir(self, filepath: str):
        pass


def _ensure_loaded():
    global _onnx_session, _verses, _beam_decoder
    if _onnx_session is not None:
        return

    import onnxruntime as ort
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No phoneme ONNX at {ONNX_MODEL_PATH}. "
            f"Download from upstream yazinsai/offline-tarteel."
        )
    print(f"[c2c-direct-phoneme] loading {ONNX_MODEL_PATH.name}...")
    _onnx_session = ort.InferenceSession(
        str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"]
    )

    with open(QURAN_PHONEMES_PATH, encoding="utf-8") as f:
        _verses = json.load(f)
    for v in _verses:
        joined = v.get("phonemes_joined", "")
        v["_phonemes_joined_ns"] = joined.replace(" ", "")
        no_bsm = None
        if (
            v["ayah"] == 1
            and v["surah"] != 1
            and v["surah"] != 9
            and joined.startswith(_BSM_PHONEMES_JOINED)
        ):
            no_bsm = joined[len(_BSM_PHONEMES_JOINED):].strip() or None
        v["_phonemes_joined_no_bsm"] = no_bsm
        v["_phonemes_joined_no_bsm_ns"] = no_bsm.replace(" ", "") if no_bsm else None
        _by_surah.setdefault(v["surah"], []).append(v)
    for verses in _by_surah.values():
        verses.sort(key=lambda x: x["ayah"])
    print(f"[c2c-direct-phoneme] loaded {len(_verses)} verses")

    if USE_BEAM:
        labels = _build_labels()
        unigrams = _read_unigrams(UNIGRAMS_PATH)
        if USE_CUSTOM_LM and NGRAMS_PATH.exists():
            from pyctcdecode.decoder import Alphabet, BeamSearchDecoderCTC

            lm = PhonemeNgramLM(NGRAMS_PATH, order=3, smoothing=0.1, weight=0.3)
            alphabet = Alphabet.build_alphabet(labels)
            _beam_decoder = BeamSearchDecoderCTC(alphabet, language_model=lm)
            print("[c2c-direct-phoneme] using beam search + custom phoneme n-gram LM")
        else:
            from pyctcdecode import build_ctcdecoder

            _beam_decoder = build_ctcdecoder(labels=labels, unigrams=unigrams, alpha=0.7, beta=1.0)
            print("[c2c-direct-phoneme] using beam search + unigrams")


def _extract_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """NeMo-style log-mel features: 80 mels, 25ms win, 10ms hop, preemph 0.97."""
    import librosa

    # Preemphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=512, hop_length=160, win_length=400,
        n_mels=80, fmax=8000, htk=True, norm="slaney",
    )
    mel = np.log(mel + 1e-5)
    mel = (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-10)
    return mel.astype(np.float32)  # [80, T]


def _compute_logprobs(audio: np.ndarray) -> np.ndarray:
    import tempfile
    import soundfile as sf
    # Save to temp wav (librosa needs to read it for some audio formats)
    if audio.ndim == 1:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, 16000)
        from shared.audio import load_audio
        audio = load_audio(tmp.name)
        os.unlink(tmp.name)
    mel = _extract_mel(audio)
    features = mel[np.newaxis]  # [1, 80, T]
    length = np.array([features.shape[-1]], dtype=np.int64)
    input_names = [inp.name for inp in _onnx_session.get_inputs()]
    out = _onnx_session.run(None, {input_names[0]: features, input_names[1]: length})[0]
    return out[0]  # [T, 70]


def _greedy_decode_phonemes(logprobs: np.ndarray) -> str:
    ids = logprobs.argmax(axis=1)
    prev = -1
    tokens = []
    for idx in ids:
        if idx != prev and idx != BLANK_ID:
            if idx < len(PHONEME_VOCAB):
                tokens.append(PHONEME_VOCAB[idx])
        prev = idx
    words = []
    cur: list[str] = []
    for t in tokens:
        if t == "|":
            if cur:
                words.append("".join(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        words.append("".join(cur))
    return " ".join(words)


def semi_global_distance(query: str, ref: str) -> int:
    if not query:
        return 0
    if not ref:
        return len(query)
    m, n = len(query), len(ref)
    prev = list(range(m + 1))
    best = prev[m]
    for j in range(1, n + 1):
        curr = [0] * (m + 1)
        for i in range(1, m + 1):
            cost = 0 if query[i - 1] == ref[j - 1] else 1
            curr[i] = min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost)
        best = min(best, curr[m])
        prev = curr
    return best


def fragment_score(query: str, ref: str) -> float:
    if not query:
        return 1.0
    return max(0.0, 1.0 - semi_global_distance(query, ref) / len(query))


def _short_query_boost(no_space_text: str, verse: dict, use_no_bsm: bool = False) -> float:
    from Levenshtein import ratio
    if use_no_bsm:
        candidate = verse.get("_phonemes_joined_no_bsm_ns", "") or verse.get("_phonemes_joined_ns", "")
    else:
        candidate = verse.get("_phonemes_joined_ns", "")
    if not candidate:
        return 0.0
    prefix_window = min(len(candidate), len(no_space_text) + 6)
    prefix = ratio(no_space_text, candidate[:prefix_window])
    if use_no_bsm:
        joined = verse.get("_phonemes_joined_no_bsm", "") or ""
    else:
        joined = verse.get("phonemes_joined", "")
    first_word = joined.split(" ")[0] if joined else ""
    first_word_score = ratio(no_space_text, first_word) if first_word else 0.0
    return max(prefix, first_word_score)


def _match_phoneme_text(phoneme_text: str, top_k: int = 10) -> list[dict]:
    from Levenshtein import ratio
    if not phoneme_text.strip():
        return []
    no_space_text = phoneme_text.replace(" ", "")
    scored: list[list] = []
    for verse in _verses:
        ref = verse.get("phonemes_joined", "")
        if not ref:
            continue
        raw = ratio(phoneme_text, ref)
        if len(no_space_text) <= 10:
            raw = max(raw, _short_query_boost(no_space_text, verse))
        no_bsm = verse.get("_phonemes_joined_no_bsm")
        if no_bsm:
            raw = max(raw, ratio(phoneme_text, no_bsm))
            if len(no_space_text) <= 10:
                raw = max(raw, _short_query_boost(no_space_text, verse, use_no_bsm=True))
        scored.append([verse, raw, raw])
    scored.sort(key=lambda x: x[2], reverse=True)
    pass2_surahs: list[int] = []
    for entry in scored:
        s = entry[0]["surah"]
        if s not in pass2_surahs:
            pass2_surahs.append(s)
        if len(pass2_surahs) >= TOP_SURAHS:
            break
    if ENABLE_FRAGMENT and len(no_space_text) >= 8:
        for i, (verse, raw, _) in enumerate(scored):
            ref_ns = verse.get("_phonemes_joined_ns", "")
            if not ref_ns:
                continue
            if len(no_space_text) >= len(ref_ns) * 0.8:
                continue
            frag = fragment_score(no_space_text, ref_ns)
            no_bsm_ns = verse.get("_phonemes_joined_no_bsm_ns")
            if no_bsm_ns:
                frag = max(frag, fragment_score(no_space_text, no_bsm_ns))
            if frag > raw:
                scored[i] = [verse, raw + (frag - raw) * 0.7, raw + (frag - raw) * 0.7]
        scored.sort(key=lambda x: x[2], reverse=True)
    span_results: list[dict] = []
    if not ENABLE_SPANS:
        singles = []
        for verse, raw, boosted in scored[:max(top_k, 32)]:
            singles.append({
                "surah": verse["surah"],
                "ayah": verse["ayah"],
                "ayah_end": None,
                "score": round(boosted, 4),
                "phonemes": verse.get("phonemes", ""),
            })
        singles.sort(key=lambda x: x["score"], reverse=True)
        return singles[:top_k]
    for surah_num in pass2_surahs:
        verses = _by_surah.get(surah_num, [])
        for i in range(len(verses)):
            for span in range(2, MAX_SPAN + 1):
                if i + span > len(verses):
                    break
                chunk = verses[i:i + span]
                first_phonemes = chunk[0].get("_phonemes_joined_no_bsm") or chunk[0].get("phonemes_joined", "")
                span_phonemes = first_phonemes + " " + " ".join(
                    v.get("phonemes_joined", "") for v in chunk[1:]
                )
                raw = ratio(phoneme_text, span_phonemes)
                span_results.append({
                    "surah": surah_num,
                    "ayah": chunk[0]["ayah"],
                    "ayah_end": chunk[-1]["ayah"],
                    "score": round(raw, 4),
                    "phonemes": " | ".join(v.get("phonemes", "") for v in chunk),
                })
    singles = []
    for verse, raw, boosted in scored[:max(top_k, 32)]:
        singles.append({
            "surah": verse["surah"],
            "ayah": verse["ayah"],
            "ayah_end": None,
            "score": round(boosted, 4),
            "phonemes": verse.get("phonemes", ""),
        })
    combined = singles + span_results
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]


def predict(audio_path: str) -> dict:
    _ensure_loaded()
    from shared.audio import load_audio
    audio = load_audio(audio_path)
    logprobs = _compute_logprobs(audio)
    if USE_BEAM and _beam_decoder is not None:
        phoneme_text = _beam_decoder.decode(
            logprobs, beam_width=24, beam_prune_logp=-10.0, token_min_logp=-5.0,
        ).strip()
    else:
        phoneme_text = _greedy_decode_phonemes(logprobs)
    top_matches = _match_phoneme_text(phoneme_text, top_k=TOP_K)
    if not top_matches:
        return {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": phoneme_text}
    text_best = top_matches[0]
    return {
        "surah": text_best["surah"],
        "ayah": text_best["ayah"],
        "ayah_end": text_best.get("ayah_end"),
        "score": text_best["score"],
        "transcript": phoneme_text,
    }


def transcribe(audio_path: str) -> str:
    _ensure_loaded()
    from shared.audio import load_audio
    audio = load_audio(audio_path)
    logprobs = _compute_logprobs(audio)
    return _greedy_decode_phonemes(logprobs)


def model_size() -> int:
    return ONNX_MODEL_PATH.stat().st_size if ONNX_MODEL_PATH.exists() else 0


import numpy as np  # noqa: E402  (used throughout)

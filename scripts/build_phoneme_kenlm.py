"""Build phoneme-level LM assets for beam decoding.

Outputs under `data/kenlm_phoneme/`:
- `quran_phoneme_corpus.txt`
- `quran_phoneme_unigrams.txt`
- `phoneme_ngrams.json`

If `lmplz` / `build_binary` are installed, also builds a KenLM ARPA/binary.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.phoneme_codec import encode_phoneme_word

QURAN_PHONEMES = PROJECT_ROOT / "data" / "quran_phonemes_web.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "kenlm_phoneme"


def load_phoneme_sentences() -> list[list[str]]:
    data = json.loads(QURAN_PHONEMES.read_text(encoding="utf-8"))
    sentences: list[list[str]] = []
    for verse in data:
        words = verse.get("phoneme_words", [])
        if words:
            sentences.append(words)
    return sentences


def build_corpus_and_unigrams(sentences: list[list[str]]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_path = OUTPUT_DIR / "quran_phoneme_corpus.txt"
    unigrams_path = OUTPUT_DIR / "quran_phoneme_unigrams.txt"

    vocab: set[str] = set()
    with corpus_path.open("w", encoding="utf-8") as f:
        for words in sentences:
            f.write(" ".join(words) + "\n")
            vocab.update(words)

    with unigrams_path.open("w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(word + "\n")

    print(f"Corpus: {len(sentences)} sentences -> {corpus_path}")
    print(f"Vocabulary: {len(vocab)} unique phoneme-words -> {unigrams_path}")
    return corpus_path, unigrams_path


def build_surrogate_corpus_and_unigrams(sentences: list[list[str]]) -> tuple[Path, Path]:
    corpus_path = OUTPUT_DIR / "quran_phoneme_corpus_surrogate.txt"
    unigrams_path = OUTPUT_DIR / "quran_phoneme_unigrams_surrogate.txt"

    vocab: set[str] = set()
    with corpus_path.open("w", encoding="utf-8") as f:
        for words in sentences:
            encoded_words = [encode_phoneme_word(w) for w in words]
            f.write(" ".join(encoded_words) + "\n")
            vocab.update(encoded_words)

    with unigrams_path.open("w", encoding="utf-8") as f:
        for word in sorted(vocab):
            f.write(word + "\n")

    print(f"Surrogate corpus -> {corpus_path}")
    print(f"Surrogate vocabulary: {len(vocab)} -> {unigrams_path}")
    return corpus_path, unigrams_path


def build_ngram_json(sentences: list[list[str]]) -> Path:
    unigram_counts: Counter[str] = Counter()
    bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)
    trigram_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for words in sentences:
        for i, w in enumerate(words):
            unigram_counts[w] += 1
            if i > 0:
                bigram_counts[words[i - 1]][w] += 1
            if i > 1:
                key = f"{words[i - 2]} {words[i - 1]}"
                trigram_counts[key][w] += 1

    ngrams = {
        "unigrams": dict(unigram_counts),
        "bigrams": {k: dict(v) for k, v in bigram_counts.items()},
        "trigrams": {k: dict(v) for k, v in trigram_counts.items()},
        "total_words": sum(unigram_counts.values()),
        "vocab_size": len(unigram_counts),
    }

    out_path = OUTPUT_DIR / "phoneme_ngrams.json"
    out_path.write_text(json.dumps(ngrams, ensure_ascii=False), encoding="utf-8")
    print(
        f"N-grams -> {out_path} "
        f"({out_path.stat().st_size / 1024:.0f} KB, {len(unigram_counts)} vocab)"
    )
    return out_path


def build_surrogate_ngram_json(sentences: list[list[str]]) -> Path:
    unigram_counts: Counter[str] = Counter()
    bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)
    trigram_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for words in sentences:
        enc_words = [encode_phoneme_word(w) for w in words]
        for i, w in enumerate(enc_words):
            unigram_counts[w] += 1
            if i > 0:
                bigram_counts[enc_words[i - 1]][w] += 1
            if i > 1:
                key = f"{enc_words[i - 2]} {enc_words[i - 1]}"
                trigram_counts[key][w] += 1

    ngrams = {
        "unigrams": dict(unigram_counts),
        "bigrams": {k: dict(v) for k, v in bigram_counts.items()},
        "trigrams": {k: dict(v) for k, v in trigram_counts.items()},
        "total_words": sum(unigram_counts.values()),
        "vocab_size": len(unigram_counts),
    }

    out_path = OUTPUT_DIR / "phoneme_ngrams_surrogate.json"
    out_path.write_text(json.dumps(ngrams, ensure_ascii=False), encoding="utf-8")
    print(
        f"Surrogate n-grams -> {out_path} "
        f"({out_path.stat().st_size / 1024:.0f} KB, {len(unigram_counts)} vocab)"
    )
    return out_path


def maybe_build_kenlm(corpus_path: Path) -> None:
    lmplz = shutil.which("lmplz")
    if not lmplz:
        print("lmplz not found - skipping ARPA/binary build")
        return

    order = 5
    arpa_path = OUTPUT_DIR / f"quran_phoneme_{order}gram.arpa"
    print(f"Building {order}-gram ARPA with lmplz...")
    subprocess.run(
        [
            lmplz,
            "-o",
            str(order),
            "--discount_fallback",
            "--text",
            str(corpus_path),
            "--arpa",
            str(arpa_path),
        ],
        check=True,
    )
    print(f"ARPA model: {arpa_path}")

    build_binary = shutil.which("build_binary")
    if build_binary:
        binary_path = OUTPUT_DIR / f"quran_phoneme_{order}gram.binary"
        subprocess.run([build_binary, str(arpa_path), str(binary_path)], check=True)
        print(f"Binary model: {binary_path}")


def main() -> None:
    sentences = load_phoneme_sentences()
    corpus_path, _ = build_corpus_and_unigrams(sentences)
    build_surrogate_corpus_and_unigrams(sentences)
    build_ngram_json(sentences)
    build_surrogate_ngram_json(sentences)
    maybe_build_kenlm(corpus_path)


if __name__ == "__main__":
    main()

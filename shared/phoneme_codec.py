"""Utilities for Quran phoneme token encoding.

`pyctcdecode` expects a character-like alphabet. Our phoneme vocabulary contains
multi-character tokens like `aa`, `SS`, and `<<`, so we map each phoneme token
to a single private-use Unicode codepoint when using beam search / LM.
"""
from __future__ import annotations

PHONEME_VOCAB = [
    "a", "u", "i", "A", "U", "I", "aa", "uu", "ii", "AA", "UU", "II",
    "<", "b", "t", "^", "j", "H", "x", "d", "*", "r", "z", "s", "$",
    "S", "D", "T", "Z", "E", "g", "f", "q", "k", "l", "m", "n", "h", "w", "y",
    "<<", "bb", "tt", "^^", "jj", "HH", "xx", "dd", "**", "rr", "zz", "ss", "$$",
    "SS", "DD", "TT", "ZZ", "EE", "gg", "ff", "qq", "kk", "ll", "mm", "nn", "hh", "ww", "yy",
    "|",
]

BLANK_ID = len(PHONEME_VOCAB)
_SURROGATE_BASE = 0xE000

TOKEN_TO_SYMBOL = {
    token: chr(_SURROGATE_BASE + idx)
    for idx, token in enumerate(PHONEME_VOCAB)
    if token != "|"
}
SYMBOL_TO_TOKEN = {v: k for k, v in TOKEN_TO_SYMBOL.items()}

_WORD_TOKENS = [t for t in PHONEME_VOCAB if t != "|"]
_WORD_TOKENS_SORTED = sorted(_WORD_TOKENS, key=len, reverse=True)


def labels_for_ctc() -> list[str]:
    labels = []
    for token in PHONEME_VOCAB:
        labels.append(" " if token == "|" else TOKEN_TO_SYMBOL[token])
    labels.append("")
    return labels


def tokenize_phoneme_word(word: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(word):
        match = None
        for tok in _WORD_TOKENS_SORTED:
            if word.startswith(tok, i):
                match = tok
                break
        if match is None:
            raise ValueError(f"Cannot tokenize phoneme word at position {i}: {word!r}")
        tokens.append(match)
        i += len(match)
    return tokens


def encode_phoneme_word(word: str) -> str:
    return "".join(TOKEN_TO_SYMBOL[t] for t in tokenize_phoneme_word(word))


def encode_phoneme_text(text: str) -> str:
    if not text.strip():
        return ""
    return " ".join(encode_phoneme_word(word) for word in text.split())


def decode_surrogate_word(word: str) -> str:
    return "".join(SYMBOL_TO_TOKEN[ch] for ch in word if ch in SYMBOL_TO_TOKEN)


def decode_surrogate_text(text: str) -> str:
    if not text.strip():
        return ""
    return " ".join(decode_surrogate_word(word) for word in text.split())

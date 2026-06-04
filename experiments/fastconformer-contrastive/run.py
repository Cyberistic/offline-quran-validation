"""FastConformer Contrastive inference script.

Inference pipeline:
  Audio → FastConformer encoder → projection → [256-d] → cosine similarity → nearest ayah

This removes ASR transcription entirely - direct audio-to-ayah retrieval.

Required exports for benchmark runner:
  - predict(audio_path: str) -> dict
  - model_size() -> int
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from shared.audio import load_audio
from shared.quran_db import QuranDB
from model import FastConformerContrastive, get_tokenizer

# Configuration
# Support both v1 and v2 models - prefer v2 if available
_v2_path = Path(__file__).parent / "checkpoints" / "best_model_v2.pt"
_v1_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
CHECKPOINT_PATH = _v2_path if _v2_path.exists() else _v1_path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_EMBEDDINGS_PATH = CACHE_DIR / "ayah_embeddings.pt"
NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inference behavior (Step 3: sliding window + voting)
WINDOW_ENABLE = os.getenv("FC_WINDOW_ENABLE", "1") == "1"
WINDOW_SECONDS = float(os.getenv("FC_WINDOW_SECONDS", "3.0"))
HOP_SECONDS = float(os.getenv("FC_HOP_SECONDS", "1.5"))
WINDOW_TOPK = int(os.getenv("FC_WINDOW_TOPK", "3"))
VITERBI_ENABLE = os.getenv("FC_VITERBI_ENABLE", "1") == "1"
TRANSITION_PENALTY = float(os.getenv("FC_TRANSITION_PENALTY", "0.75"))
JUMP_PENALTY = float(os.getenv("FC_JUMP_PENALTY", "1.5"))

# Global singletons (lazy-loaded)
_quran_db = None
_nemo_model = None
_contrastive_model = None
_tokenizer = None
_ayah_embeddings = None  # Pre-computed embeddings for all ayahs
_ayah_metadata = None  # List of (surah, ayah) for each embedding


def _checkpoint_fingerprint() -> str:
    """Return a stable fingerprint for the active checkpoint."""
    if not CHECKPOINT_PATH.exists():
        return "no-checkpoint"
    h = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _install_kaldialign_fallback():
    """Install kaldialign fallback for NeMo."""
    import types

    def align(ref, hyp, eps="<eps>"):
        ref = list(ref)
        hyp = list(hyp)
        n, m = len(ref), len(hyp)

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        bt = [[None] * (m + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            dp[i][0] = i
            bt[i][0] = "D"
        for j in range(1, m + 1):
            dp[0][j] = j
            bt[0][j] = "I"

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                sub = dp[i - 1][j - 1] + cost
                ins = dp[i - 1][j] + 1
                delete = dp[i][j - 1] + 1
                best = min(sub, ins, delete)
                dp[i][j] = best
                if best == sub:
                    bt[i][j] = "S"
                elif best == ins:
                    bt[i][j] = "I"
                else:
                    bt[i][j] = "D"

        out = []
        i, j = n, m
        while i > 0 or j > 0:
            move = bt[i][j]
            if move == "S":
                out.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif move == "I":
                out.append((ref[i - 1], eps))
                i -= 1
            else:
                out.append((eps, hyp[j - 1]))
                j -= 1

        out.reverse()
        return out

    mod = types.ModuleType("kaldialign")
    mod.align = align
    sys.modules["kaldialign"] = mod


def _ensure_loaded():
    """Lazy-load all models and pre-compute ayah embeddings."""
    global _quran_db, _nemo_model, _contrastive_model, _tokenizer
    global _ayah_embeddings, _ayah_metadata

    if _contrastive_model is not None:
        return

    print("Loading FastConformer contrastive model...")

    # Install kaldialign fallback
    _install_kaldialign_fallback()

    # Suppress NeMo warnings
    os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    # Load QuranDB
    _quran_db = QuranDB()

    # Load tokenizer
    _tokenizer = get_tokenizer()

    # Load FastConformer audio encoder (frozen)
    try:
        from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
        from nemo.utils import logging as nemo_logging

        nemo_logging.set_verbosity(nemo_logging.ERROR)
    except ImportError as exc:
        raise ImportError(
            "NeMo ASR dependencies required. "
            "Install with: pip install 'nemo_toolkit[asr]'"
        ) from exc

    print(f"Loading FastConformer encoder from {NVIDIA_MODEL_ID}...")
    _nemo_model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        NVIDIA_MODEL_ID,
        map_location=DEVICE,
    )
    _nemo_model.eval()
    _nemo_model.to(DEVICE)

    # Freeze encoder
    for param in _nemo_model.parameters():
        param.requires_grad = False

    # Load contrastive model
    print("Loading contrastive projection model...")
    _contrastive_model = FastConformerContrastive(embed_dim=256)

    if CHECKPOINT_PATH.exists():
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        _contrastive_model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print(f"Warning: No checkpoint found at {CHECKPOINT_PATH}")
        print("Using untrained projection model (random embeddings)")

    _contrastive_model.to(DEVICE)
    _contrastive_model.eval()

    # Load or compute ayah embeddings
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    current_fp = _checkpoint_fingerprint()
    use_cache = False
    if CACHE_EMBEDDINGS_PATH.exists():
        print(f"Loading cached ayah embeddings from {CACHE_EMBEDDINGS_PATH}...")
        cache = torch.load(CACHE_EMBEDDINGS_PATH, map_location="cpu")
        cache_fp = cache.get("checkpoint_fingerprint")
        if cache_fp == current_fp and "embeddings" in cache and "metadata" in cache:
            _ayah_embeddings = cache["embeddings"]
            _ayah_metadata = cache["metadata"]
            use_cache = True
        else:
            print("Cached embeddings are stale for current checkpoint; recomputing...")

    if not use_cache:
        print("Pre-computing ayah embeddings...")
        _ayah_embeddings, _ayah_metadata = _compute_ayah_embeddings()
        torch.save(
            {
                "embeddings": _ayah_embeddings,
                "metadata": _ayah_metadata,
                "checkpoint_fingerprint": current_fp,
            },
            CACHE_EMBEDDINGS_PATH,
        )
        print(f"Saved ayah embeddings to {CACHE_EMBEDDINGS_PATH}")
    print(f"Loaded {len(_ayah_metadata)} ayah embeddings")


def _compute_ayah_embeddings():
    """Pre-compute embeddings for all 6,236 ayahs."""
    embeddings = []
    metadata = []

    with torch.no_grad():
        for verse in tqdm(_quran_db.verses, desc="Encoding ayah embeddings"):
            surah = verse["surah"]
            ayah = verse["ayah"]
            text = verse["text_clean"]

            # Tokenize
            encoding = _tokenizer(
                text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Encode text
            input_ids = encoding["input_ids"].to(DEVICE)
            attention_mask = encoding["attention_mask"].to(DEVICE)

            embedding = _contrastive_model.encode_text(input_ids, attention_mask)
            embeddings.append(embedding.cpu())
            metadata.append((surah, ayah))

    # Stack all embeddings [num_ayahs, 256]
    all_embeddings = torch.cat(embeddings, dim=0)
    return all_embeddings, metadata


def _extract_audio_features_from_audio(audio: np.ndarray) -> torch.Tensor:
    """Extract FastConformer features from mono float32 waveform."""
    # Truncate to 20 seconds max
    max_samples = 20 * 16000
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(DEVICE)

    # Extract features
    with torch.no_grad():
        audio_length = torch.tensor([audio_tensor.shape[1]], device=DEVICE)
        processed_signal, processed_signal_length = _nemo_model.preprocessor(
            input_signal=audio_tensor,
            length=audio_length,
        )
        features, _ = _nemo_model.encoder(
            audio_signal=processed_signal,
            length=processed_signal_length,
        )
        if (
            features.dim() == 3
            and features.shape[-1] != 512
            and features.shape[1] == 512
        ):
            features = features.transpose(1, 2)

    return features


def _extract_audio_features(audio_path: str) -> torch.Tensor:
    """Extract FastConformer features from audio file."""
    audio = load_audio(audio_path, sr=16000)
    return _extract_audio_features_from_audio(audio)


def _retrieve_ayah(audio_features: torch.Tensor, top_k: int = 1) -> List[Dict]:
    """
    Retrieve top-k matching ayahs using cosine similarity.

    Returns:
        List of dicts with surah, ayah, score
    """
    with torch.no_grad():
        # Encode audio
        audio_embed = _contrastive_model.encode_audio(audio_features)
        audio_embed = audio_embed.cpu()

        # Compute cosine similarity with all ayahs
        similarities = F.cosine_similarity(audio_embed, _ayah_embeddings, dim=1)

        # Get top-k
        top_scores, top_indices = similarities.topk(top_k)

        results = []
        for score, idx in zip(top_scores, top_indices):
            surah, ayah = _ayah_metadata[idx]
            results.append(
                {
                    "surah": surah,
                    "ayah": ayah,
                    "score": float(score),
                }
            )

        return results


def debug_topk(audio_path: str, top_k: int = 5) -> List[Dict]:
    """Return top-k retrieved ayahs for debugging."""
    _ensure_loaded()
    try:
        audio_features = _extract_audio_features(audio_path)
        return _retrieve_ayah(audio_features, top_k=top_k)
    except Exception as e:
        print(f"Error debugging {audio_path}: {e}")
        return []


def _predict_with_window_voting(audio: np.ndarray) -> Dict:
    """Run sliding-window retrieval and vote across window candidates."""
    window_samples = max(1, int(WINDOW_SECONDS * 16000))
    hop_samples = max(1, int(HOP_SECONDS * 16000))

    if len(audio) <= window_samples:
        feats = _extract_audio_features_from_audio(audio)
        top = _retrieve_ayah(feats, top_k=1)
        return top[0] if top else {"surah": 0, "ayah": 0, "score": 0.0}

    votes = {}
    per_window = []
    prev_top1 = None
    for start in range(0, max(1, len(audio) - window_samples + 1), hop_samples):
        chunk = audio[start : start + window_samples]
        if len(chunk) < window_samples:
            break

        feats = _extract_audio_features_from_audio(chunk)
        cands = _retrieve_ayah(feats, top_k=WINDOW_TOPK)
        if not cands:
            continue

        per_window.append(cands)

        for rank, cand in enumerate(cands):
            key = (cand["surah"], cand["ayah"])
            weight = (WINDOW_TOPK - rank) + cand["score"]
            bonus = 0.0
            if prev_top1 is not None:
                ps, pa = prev_top1
                if cand["surah"] == ps and abs(cand["ayah"] - pa) <= 2:
                    bonus = 0.5
            votes[key] = votes.get(key, 0.0) + weight + bonus

        prev_top1 = (cands[0]["surah"], cands[0]["ayah"])

    if not votes:
        return {"surah": 0, "ayah": 0, "score": 0.0}

    if VITERBI_ENABLE and len(per_window) >= 2:
        best = _viterbi_pick(per_window)
        if best is not None:
            return best

    (surah, ayah), score = max(votes.items(), key=lambda kv: kv[1])
    return {"surah": surah, "ayah": ayah, "score": float(score)}


def _transition_cost(prev: Dict, curr: Dict) -> float:
    """Sequence-aware transition penalty between consecutive window picks."""
    if prev["surah"] != curr["surah"]:
        return JUMP_PENALTY
    gap = abs(int(curr["ayah"]) - int(prev["ayah"]))
    if gap <= 1:
        return 0.0
    if gap <= 3:
        return TRANSITION_PENALTY
    return JUMP_PENALTY


def _viterbi_pick(per_window: List[List[Dict]]) -> Optional[Dict]:
    """Pick a continuity-respecting best candidate path across windows."""
    if not per_window:
        return None

    dp = []
    back = []

    first_scores = [cand["score"] for cand in per_window[0]]
    dp.append(first_scores)
    back.append([-1] * len(per_window[0]))

    for t in range(1, len(per_window)):
        curr_scores = []
        curr_back = []
        for j, curr in enumerate(per_window[t]):
            best_score = -1e9
            best_i = 0
            for i, prev in enumerate(per_window[t - 1]):
                score = dp[t - 1][i] + curr["score"] - _transition_cost(prev, curr)
                if score > best_score:
                    best_score = score
                    best_i = i
            curr_scores.append(best_score)
            curr_back.append(best_i)
        dp.append(curr_scores)
        back.append(curr_back)

    # Backtrack best path
    last_idx = int(np.argmax(dp[-1]))
    path = [last_idx]
    for t in range(len(per_window) - 1, 0, -1):
        last_idx = back[t][last_idx]
        path.append(last_idx)
    path.reverse()

    # Return mode across path for stability
    counts = {}
    score_sum = {}
    for t, idx in enumerate(path):
        c = per_window[t][idx]
        key = (c["surah"], c["ayah"])
        counts[key] = counts.get(key, 0) + 1
        score_sum[key] = score_sum.get(key, 0.0) + c["score"]

    best_key = max(counts.keys(), key=lambda k: (counts[k], score_sum[k]))
    return {
        "surah": int(best_key[0]),
        "ayah": int(best_key[1]),
        "score": float(score_sum[best_key] / counts[best_key]),
    }


def predict(audio_path: str) -> dict:
    """
    Run inference on an audio file.

    Returns:
        {
            "surah": int,
            "ayah": int,
            "ayah_end": int|None,
            "score": float,
            "transcript": str
        }
    """
    _ensure_loaded()

    try:
        audio = load_audio(audio_path, sr=16000)

        if WINDOW_ENABLE:
            best = _predict_with_window_voting(audio)
        else:
            audio_features = _extract_audio_features_from_audio(audio)
            results = _retrieve_ayah(audio_features, top_k=1)
            best = results[0] if results else {"surah": 0, "ayah": 0, "score": 0.0}

        if best["surah"] == 0:
            return {
                "surah": 0,
                "ayah": 0,
                "ayah_end": None,
                "score": 0.0,
                "transcript": "",
            }

        # Get verse text for transcript field
        verse = _quran_db.get_verse(best["surah"], best["ayah"])
        text = verse["text_uthmani"] if verse else ""

        return {
            "surah": best["surah"],
            "ayah": best["ayah"],
            "ayah_end": None,
            "score": best["score"],
            "transcript": text[:100],  # First 100 chars as "transcript"
        }

    except Exception as e:
        print(f"Error predicting {audio_path}: {e}")
        return {
            "surah": 0,
            "ayah": 0,
            "ayah_end": None,
            "score": 0.0,
            "transcript": "",
        }


def transcribe(audio_path: str) -> str:
    """
    Transcribe audio (required for benchmark runner).
    Returns the matched ayah text.
    """
    result = predict(audio_path)
    return result.get("transcript", "")


def model_size() -> int:
    """
    Return model size in bytes.

    Includes:
    - FastConformer encoder (frozen)
    - Projection layers (trainable)
    """
    _ensure_loaded()

    # Calculate size of all loaded models
    total_params = 0

    # Audio encoder (FastConformer)
    if _nemo_model is not None:
        total_params += sum(p.numel() for p in _nemo_model.parameters())

    # Contrastive model (projections)
    if _contrastive_model is not None:
        total_params += sum(p.numel() for p in _contrastive_model.parameters())

    # Text encoder (AraBERT - part of contrastive model)
    # Already included in _contrastive_model

    # Convert to bytes (float32 = 4 bytes per param)
    size_bytes = total_params * 4

    # Add embedding cache size (6236 ayahs * 256 dim * 4 bytes)
    if _ayah_embeddings is not None:
        size_bytes += _ayah_embeddings.numel() * 4

    return size_bytes


if __name__ == "__main__":
    # Test inference
    print("Testing FastConformer contrastive inference...")

    # Load models
    _ensure_loaded()

    print(f"Model size: {model_size() / (1024 * 1024):.1f} MB")

    # Test on a sample if available
    test_corpus = PROJECT_ROOT / "benchmark" / "test_corpus"
    if test_corpus.exists():
        wav_files = list(test_corpus.glob("*.wav"))
        if wav_files:
            test_file = wav_files[0]
            print(f"\nTesting on: {test_file.name}")
            result = predict(str(test_file))
            print(f"Prediction: Surah {result['surah']}, Ayah {result['ayah']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['transcript'][:50]}...")

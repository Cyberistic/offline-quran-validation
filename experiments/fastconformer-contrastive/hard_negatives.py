"""Hard negative mining for contrastive learning.

Generates hard negatives by:
1. Embedding all ayahs and finding nearest neighbors (phonetically similar)
2. Mining from training errors (model's own mistakes)
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Set
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_phonetic_similarity(text1: str, text2: str) -> float:
    """Compute phonetic similarity between two Arabic texts.

    Uses a simplified approach: normalized character overlap.
    """

    # Normalize: remove diacritics, normalize alef variants
    def normalize(t: str) -> str:
        t = t.replace("َ", "").replace("ُ", "").replace("ِ", "")
        t = t.replace("ّ", "").replace("ْ", "").replace("ً", "")
        t = t.replace("ٌ", "").replace("ٍ", "").replace("ٰ", "")
        t = t.replace("ٓ", "")
        # Normalize alef variants
        t = t.replace("آ", "ا").replace("أ", "ا").replace("إ", "ا").replace("ٱ", "ا")
        # Normalize taa marbuta
        t = t.replace("ة", "ه")
        return t.strip()

    n1, n2 = normalize(text1), normalize(text2)

    if not n1 or not n2:
        return 0.0

    # Character n-gram overlap
    def get_ngrams(text: str, n: int = 3) -> Set[str]:
        return set(text[i : i + n] for i in range(len(text) - n + 1))

    ngrams1 = get_ngrams(n1, n=3)
    ngrams2 = get_ngrams(n2, n=3)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def build_phonetic_index(
    quran_db, top_k: int = 20
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Build index of phonetically similar ayahs.

    Returns dict mapping (surah, ayah) -> list of similar (surah, ayah)
    """
    print("Building phonetic similarity index...")

    # Get all verses
    verses = [(v["surah"], v["ayah"], v["text_clean"]) for v in quran_db.verses]

    # Compute similarity matrix (this is O(n^2) but only done once)
    index = {}

    for i, (s1, a1, t1) in enumerate(tqdm(verses, desc="Computing similarities")):
        similarities = []
        for j, (s2, a2, t2) in enumerate(verses):
            if i == j:
                continue
            sim = compute_phonetic_similarity(t1, t2)
            if sim > 0.3:  # Only keep reasonably similar ones
                similarities.append((sim, (s2, a2)))

        # Sort by similarity and take top-k
        similarities.sort(reverse=True)
        index[(s1, a1)] = [ayah for _, ayah in similarities[:top_k]]

    return index


class HardNegativeMiner:
    """Mine hard negatives for contrastive learning."""

    def __init__(self, quran_db, cache_path: str = None):
        self.quran_db = quran_db
        self.cache_path = Path(cache_path) if cache_path else None
        self.phonetic_index = None
        self.mining_history = []  # Track which negatives were hardest

        if self.cache_path and self.cache_path.exists():
            print(f"Loading phonetic index from {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                self.phonetic_index = pickle.load(f)
        else:
            self.phonetic_index = build_phonetic_index(quran_db)
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "wb") as f:
                    pickle.dump(self.phonetic_index, f)
                print(f"Saved phonetic index to {self.cache_path}")

    def get_hard_negatives(
        self,
        surah: int,
        ayah: int,
        n: int = 5,
        include_phonetic: bool = True,
        include_random: bool = True,
    ) -> List[Tuple[int, int]]:
        """Get hard negatives for a given ayah.

        Args:
            surah: Target surah number
            ayah: Target ayah number
            n: Number of negatives to return
            include_phonetic: Include phonetically similar ayahs
            include_random: Include random negatives

        Returns:
            List of (surah, ayah) tuples
        """
        negatives = []
        key = (surah, ayah)

        # Get phonetically similar negatives
        if include_phonetic and key in self.phonetic_index:
            phonetic_negs = self.phonetic_index[key][:n]
            negatives.extend(phonetic_negs)

        # Fill remaining with random negatives
        if include_random and len(negatives) < n:
            needed = n - len(negatives)
            all_ayahs = [(v["surah"], v["ayah"]) for v in self.quran_db.verses]
            all_ayahs = [a for a in all_ayahs if a != key and a not in negatives]
            if len(all_ayahs) >= needed:
                random_negs = np.random.choice(len(all_ayahs), needed, replace=False)
                negatives.extend([all_ayahs[i] for i in random_negs])

        return negatives[:n]

    def mine_from_model_errors(
        self,
        model,
        audio_features: torch.Tensor,
        targets: List[Tuple[int, int]],
        device: str = "cuda",
        n_hard: int = 5,
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Mine hard negatives from model's own errors.

        This is done during training to find ayahs that the model confuses.
        """
        hard_negs = {}

        with torch.no_grad():
            # Encode audio
            audio_embeds = model.encode_audio(audio_features)

            # For each sample, find what the model incorrectly thinks is similar
            for i, target_key in enumerate(targets):
                audio_emb = audio_embeds[i : i + 1]  # [1, embed_dim]

                # Compute similarity with all text embeddings (this is expensive,
                # so we do it on a subset)
                # In practice, we'd maintain a cache of text embeddings
                pass  # Implementation would go here

        return hard_negs


def create_hard_negative_dataset(
    feature_dir: str,
    output_dir: str,
    quran_db,
    negatives_per_sample: int = 5,
):
    """Create a dataset with hard negatives pre-computed.

    This creates triplets: (anchor_audio, positive_text, negative_texts)
    """
    from pathlib import Path
    import torch

    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    miner = HardNegativeMiner(quran_db, cache_path=output_dir / "phonetic_index.pkl")

    # Load all feature files
    feature_files = list(feature_dir.rglob("*.pt"))

    dataset = []
    for feat_file in tqdm(feature_files, desc="Creating hard negative dataset"):
        try:
            parts = feat_file.stem.split("_")
            if len(parts) < 2:
                continue

            surah = int(parts[0])
            ayah = int(parts[1])

            # Get hard negatives
            hard_negs = miner.get_hard_negatives(surah, ayah, n=negatives_per_sample)

            dataset.append(
                {
                    "feature_path": str(feat_file),
                    "surah": surah,
                    "ayah": ayah,
                    "hard_negatives": hard_negs,
                }
            )
        except (ValueError, IndexError):
            continue

    # Save dataset
    output_path = output_dir / "hard_negative_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Saved to {output_path}")

    return dataset


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from shared.quran_db import QuranDB

    print("Testing hard negative mining...")

    db = QuranDB()
    miner = HardNegativeMiner(db)

    # Test on a few ayahs
    test_cases = [(1, 1), (2, 255), (112, 1)]

    for surah, ayah in test_cases:
        verse = db.get_verse(surah, ayah)
        print(f"\nSurah {surah}:{ayah}: {verse['text_clean'][:50]}...")

        hard_negs = miner.get_hard_negatives(surah, ayah, n=5)
        print("Hard negatives:")
        for ns, na in hard_negs:
            nv = db.get_verse(ns, na)
            sim = compute_phonetic_similarity(verse["text_clean"], nv["text_clean"])
            print(f"  {ns}:{na} (sim={sim:.3f}): {nv['text_clean'][:40]}...")

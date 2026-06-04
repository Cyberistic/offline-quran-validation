"""Dataset for FastConformer contrastive learning.

Loads Quran audio recordings and pairs them with their corresponding
text verses for supervised contrastive training.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB


class QuranContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning: audio + text pairs.
    
    Each sample is a (surah, ayah) pair with:
    - Audio waveform (from local files or pre-extracted features)
    - Text (Arabic verse from quran.json)
    """
    
    def __init__(
        self,
        audio_dir: str = None,
        quran_json_path: str = None,
        reciters: List[str] = None,
        max_ayah_length: int = 512,
        max_audio_length: int = 20,  # seconds
        augment: bool = True,
        feature_cache_dir: str = None,
    ):
        """
        Args:
            audio_dir: Directory containing audio files (e.g., EveryAyah)
            quran_json_path: Path to quran.json
            reciters: List of reciter names to include (None = all)
            max_ayah_length: Max tokens for text
            max_audio_length: Max audio length in seconds
            augment: Whether to apply audio augmentation
            feature_cache_dir: Directory to cache pre-extracted features
        """
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.db = QuranDB() if quran_json_path is None else QuranDB(quran_json_path)
        self.reciters = reciters
        self.max_ayah_length = max_ayah_length
        self.max_audio_length = max_audio_length
        self.augment = augment
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
        
        # Build sample list: (surah, ayah, audio_path, text)
        self.samples = self._build_samples()
        
        print(f"Loaded {len(self.samples)} ayah samples for contrastive training")
    
    def _build_samples(self) -> List[Dict]:
        """Build list of training samples with audio paths and text."""
        samples = []
        
        if self.audio_dir and self.audio_dir.exists():
            # Scan audio directory for ayah files
            for audio_file in self.audio_dir.rglob("*.mp3"):
                # Parse filename: surah_ayah.mp3 or similar
                try:
                    parts = audio_file.stem.split("_")
                    if len(parts) >= 2:
                        surah = int(parts[0])
                        ayah = int(parts[1])
                        
                        # Get text from QuranDB
                        verse = self.db.get_verse(surah, ayah)
                        if verse:
                            text = normalize_arabic(verse["text_clean"])
                            samples.append({
                                "surah": surah,
                                "ayah": ayah,
                                "audio_path": str(audio_file),
                                "text": text,
                            })
                except (ValueError, IndexError):
                    continue
        
        # If no audio found, create samples with text only (for feature extraction mode)
        if not samples:
            print("Warning: No audio files found. Using text-only samples.")
            for verse in self.db.verses:
                samples.append({
                    "surah": verse["surah"],
                    "ayah": verse["ayah"],
                    "audio_path": None,
                    "text": normalize_arabic(verse["text_clean"]),
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dict with:
            - audio: [samples] waveform or pre-extracted features
            - input_ids: [max_len] tokenized text
            - attention_mask: [max_len] attention mask
            - surah: int
            - ayah: int
        """
        sample = self.samples[idx]
        
        # Load audio if available
        audio = None
        if sample["audio_path"] and Path(sample["audio_path"]).exists():
            try:
                audio = load_audio(sample["audio_path"], sr=16000)
                
                # Truncate if too long
                max_samples = self.max_audio_length * 16000
                if len(audio) > max_samples:
                    # Random crop for training
                    start = random.randint(0, len(audio) - max_samples)
                    audio = audio[start:start + max_samples]
                
                # Augmentation
                if self.augment and random.random() > 0.5:
                    audio = self._augment_audio(audio)
                
            except Exception as e:
                print(f"Error loading audio {sample['audio_path']}: {e}")
                audio = None
        
        # Tokenize text
        text = sample["text"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_ayah_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "audio": torch.FloatTensor(audio) if audio is not None else None,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "surah": sample["surah"],
            "ayah": sample["ayah"],
            "text": text,
        }
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply simple audio augmentation."""
        # Speed perturbation (±10%)
        if random.random() > 0.5:
            speed_factor = random.uniform(0.9, 1.1)
            # Simple linear interpolation for speed change
            indices = np.round(np.arange(0, len(audio), speed_factor)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]
        
        # Add noise (SNR ~30dB)
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.01, audio.shape)
            audio = audio + noise
        
        return audio


class FeatureCachedDataset(Dataset):
    """
    Dataset that uses pre-extracted FastConformer features instead of raw audio.
    Much faster for training since encoder is frozen.
    """
    
    def __init__(
        self,
        feature_dir: str,
        quran_json_path: str = None,
        feature_limit: int = 0,
    ):
        """
        Args:
            feature_dir: Directory with pre-extracted .pt feature files
            quran_json_path: Path to quran.json
        """
        self.feature_dir = Path(feature_dir)
        self.db = QuranDB() if quran_json_path is None else QuranDB(quran_json_path)
        self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
        self.feature_limit = feature_limit
        self.ayah_token_cache = self._build_ayah_token_cache()
        
        # Load all feature files
        self.samples = self._load_features()
        print(f"Loaded {len(self.samples)} pre-extracted feature samples")
    
    def _load_features(self) -> List[Dict]:
        """Load pre-extracted features and pair with text."""
        samples = []
        
        feature_files = []
        for p in self.feature_dir.rglob("*.pt"):
            if p.name.startswith("._"):
                continue
            feature_files.append(p)
            if self.feature_limit and self.feature_limit > 0 and len(feature_files) >= self.feature_limit:
                break

        for feature_file in tqdm(feature_files, desc="Indexing feature files"):
            try:
                # Parse filename: surah_ayah_reciter.pt
                parts = feature_file.stem.split("_")
                if len(parts) >= 2:
                    surah = int(parts[0])
                    ayah = int(parts[1])
                    
                    cached = self.ayah_token_cache.get((surah, ayah))
                    if cached is None:
                        continue

                    samples.append({
                        "feature_path": str(feature_file),
                        "surah": surah,
                        "ayah": ayah,
                        "reciter": "_".join(parts[2:]) if len(parts) > 2 else None,
                        "input_ids": cached["input_ids"],
                        "attention_mask": cached["attention_mask"],
                    })
            except (ValueError, IndexError):
                continue
        
        return samples

    def _build_ayah_token_cache(self) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
        """Tokenize each ayah once and reuse for all reciters."""
        texts = []
        keys = []
        for verse in self.db.verses:
            keys.append((verse["surah"], verse["ayah"]))
            texts.append(normalize_arabic(verse["text_clean"]))

        encoding = self.tokenizer(
            texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        cache = {}
        for i, key in enumerate(keys):
            cache[key] = {
                "input_ids": encoding["input_ids"][i],
                "attention_mask": encoding["attention_mask"][i],
            }
        return cache
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load pre-extracted features and text."""
        sample = self.samples[idx]
        
        # Load features
        features = torch.load(sample["feature_path"], map_location="cpu")
        
        return {
            "audio_features": features,
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "surah": sample["surah"],
            "ayah": sample["ayah"],
            "reciter": sample.get("reciter"),
        }


def collate_contrastive_batch(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.
    
    Handles None audio (for samples without audio) by filtering them out.
    """
    # Filter out samples without audio
    valid_batch = [b for b in batch if b.get("audio") is not None or b.get("audio_features") is not None]
    
    if not valid_batch:
        return None
    
    # Stack audio features
    if "audio_features" in valid_batch[0]:
        # Pre-extracted features: pad to max length
        feature_list = []
        for b in valid_batch:
            feat = b["audio_features"]
            # Normalize to [seq_len, feat_dim]
            if feat.dim() == 2 and feat.shape[0] == 512 and feat.shape[1] != 512:
                feat = feat.transpose(0, 1)
            feature_list.append(feat)

        max_len = max(f.shape[0] for f in feature_list)
        feat_dim = feature_list[0].shape[1]
        audio_features = torch.zeros(len(feature_list), max_len, feat_dim)
        for i, feat in enumerate(feature_list):
            audio_features[i, :feat.shape[0]] = feat
    else:
        # Raw audio - will need feature extraction
        # For now, skip since we need pre-extracted features
        audio_features = None
    
    # Stack text
    input_ids = torch.stack([b["input_ids"] for b in valid_batch])
    attention_mask = torch.stack([b["attention_mask"] for b in valid_batch])
    
    return {
        "audio_features": audio_features,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "surahs": [b["surah"] for b in valid_batch],
        "ayahs": [b["ayah"] for b in valid_batch],
        "reciters": [b.get("reciter") for b in valid_batch],
    }


if __name__ == "__main__":
    # Test dataset
    print("Testing QuranContrastiveDataset...")
    
    dataset = QuranContrastiveDataset()
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Audio shape: {sample['audio'].shape if sample['audio'] is not None else None}")
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Text: {sample['text'][:50]}...")

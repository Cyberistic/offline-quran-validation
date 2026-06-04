"""Training script for FastConformer contrastive learning.

Two-phase training:
1. Pre-extract features from frozen FastConformer encoder (one-time)
2. Train projection heads with supervised contrastive loss

Usage:
    python train.py --extract-features --audio-dir /path/to/audio --output-dir ./checkpoints
    python train.py --train --feature-dir ./features --output-dir ./checkpoints --epochs 50
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from multiprocessing import cpu_count

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.audio import load_audio
from shared.quran_db import QuranDB
from model import FastConformerContrastive, contrastive_loss, get_tokenizer


def _build_ayah_text_bank(db, tokenizer, max_length=512, batch_size=256):
    """Build one text embedding target per ayah for global retrieval eval."""
    ayah_keys = []
    texts = []
    for verse in db.verses:
        ayah_keys.append((verse["surah"], verse["ayah"]))
        texts.append(verse["text_clean"])
    input_ids_parts = []
    attention_parts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing ayah text bank"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids_parts.append(encoded["input_ids"])
        attention_parts.append(encoded["attention_mask"])

    return ayah_keys, torch.cat(input_ids_parts, dim=0), torch.cat(attention_parts, dim=0)


def _build_ayah_text_bank_from_db(db, max_length=512):
    """Build full ayah text bank once from QuranDB using AraBERT tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    ayah_keys = []
    texts = []
    for verse in db.verses:
        ayah_keys.append((verse["surah"], verse["ayah"]))
        texts.append(verse["text_clean"])

    input_ids_parts = []
    attention_parts = []
    batch_size = 256
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing ayah text bank"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids_parts.append(encoded["input_ids"])
        attention_parts.append(encoded["attention_mask"])

    return ayah_keys, torch.cat(input_ids_parts, dim=0), torch.cat(attention_parts, dim=0)


def _split_by_reciter(dataset, val_fraction=0.2):
    """Split by reciter to reduce leakage between train and val."""
    by_reciter = {}
    for idx, sample in enumerate(dataset.samples):
        reciter = sample.get("reciter") or "unknown"
        by_reciter.setdefault(reciter, []).append(idx)

    rng = np.random.default_rng(42)
    reciters = sorted(by_reciter.keys())
    rng.shuffle(reciters)
    val_count = max(1, int(len(reciters) * val_fraction))
    val_reciters = set(reciters[:val_count])

    train_idx, val_idx = [], []
    for reciter, indices in by_reciter.items():
        if reciter in val_reciters:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    if not train_idx or not val_idx:
        split = int(0.8 * len(dataset))
        train_idx = list(range(split))
        val_idx = list(range(split, len(dataset)))

    return train_idx, val_idx


def extract_features(
    audio_dir: str,
    output_dir: str,
    batch_size: int = 8,
    device: str = "cuda",
):
    """
    Pre-extract features from frozen FastConformer encoder.
    This is a one-time cost - after this, training is much faster.
    """
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from nemo.utils import logging as nemo_logging
    nemo_logging.set_verbosity(nemo_logging.ERROR)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    print(f"Loading FastConformer model on {device}...")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
        map_location=device,
    )
    encoder = model.encoder
    encoder.eval()
    encoder.to(device)
    
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Find all audio files + metadata
    audio_root = Path(audio_dir)
    manifest_path = audio_root / "manifest.json"
    items = []
    if manifest_path.exists():
        print(f"Using manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        for sample in manifest.get("samples", []):
            # Skip multi-ayah samples for now
            if sample.get("ayah_end"):
                continue
            file_name = sample.get("file")
            if not file_name:
                continue
            audio_path = audio_root / file_name
            if not audio_path.is_file() or audio_path.stat().st_size == 0:
                continue
            items.append({
                "path": audio_path,
                "surah": sample.get("surah"),
                "ayah": sample.get("ayah"),
                "reciter": sample.get("reciter", "unknown"),
            })
    else:
        audio_files = list(audio_root.rglob("*.mp3")) + list(audio_root.rglob("*.wav"))
        for audio_file in audio_files:
            parts = audio_file.stem.split("_")
            if len(parts) < 2:
                continue
            try:
                surah = int(parts[0])
                ayah = int(parts[1])
            except ValueError:
                continue
            items.append({"path": audio_file, "surah": surah, "ayah": ayah})
    print(f"Found {len(items)} audio items")
    
    # Extract features in batches
    features_extracted = 0
    bad_log_path = output_path / "bad_audio.txt"
    bad_log = open(bad_log_path, "a")
    for i in tqdm(range(0, len(items), batch_size), desc="Extracting features"):
        batch_items = items[i:i + batch_size]
        
        # Load audio
        audio_batch = []
        valid_files = []
        for item in batch_items:
            audio_file = item["path"]
            reciter = str(item.get("reciter", "unknown"))
            reciter_safe = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in reciter
            )
            feature_file = output_path / f"{int(item['surah']):03d}_{int(item['ayah']):03d}_{reciter_safe}.pt"
            if feature_file.exists():
                continue
            try:
                audio = load_audio(str(audio_file), sr=16000)
                # Truncate to 20 seconds max
                max_samples = 20 * 16000
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
                audio_batch.append(torch.FloatTensor(audio))
                valid_files.append(item)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
                bad_log.write(str(audio_file) + "\n")
                bad_log.flush()
                continue
        
        if not audio_batch:
            continue
        
        # Pad batch
        max_len = max(len(a) for a in audio_batch)
        padded = torch.zeros(len(audio_batch), max_len)
        for j, audio in enumerate(audio_batch):
            padded[j, :len(audio)] = audio
        
        # Extract features
        with torch.no_grad():
            audio_signal = padded.to(device)
            audio_length = torch.tensor(
                [len(a) for a in audio_batch], device=device
            )
            processed_signal, processed_signal_length = model.preprocessor(
                input_signal=audio_signal,
                length=audio_length,
            )
            features, _ = encoder(
                audio_signal=processed_signal,
                length=processed_signal_length,
            )
        
        # Save features
        for j, item in enumerate(valid_files):
            # Parse surah/ayah from filename
            try:
                surah = int(item["surah"])
                ayah = int(item["ayah"])
                
                reciter = str(item.get("reciter", "unknown"))
                reciter_safe = "".join(
                    c if c.isalnum() or c in ("-", "_") else "_" for c in reciter
                )
                feature_file = output_path / f"{surah:03d}_{ayah:03d}_{reciter_safe}.pt"
                torch.save(features[j].cpu(), feature_file)
                features_extracted += 1
            except (ValueError, IndexError) as e:
                print(f"Error parsing metadata for {item['path']}: {e}")
    
    bad_log.close()
    print(f"Extracted features for {features_extracted} ayahs")
    print(f"Bad audio log: {bad_log_path}")
    return features_extracted


def train_projection_heads(
    feature_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    quran_json_path: str = None,
    resume: bool = False,
    feature_limit: int = 0,
    val_text_chunk_size: int = 8,
):
    """Train projection heads with contrastive loss."""
    
    from dataset import FeatureCachedDataset, collate_contrastive_batch
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_path / "training_log.json"
    logs = []
    last_checkpoint = output_path / "last_checkpoint.pt"
    
    # Load dataset
    print(f"Loading dataset from {feature_dir}...")
    dataset = FeatureCachedDataset(feature_dir, quran_json_path, feature_limit=feature_limit)
    
    if len(dataset) == 0:
        print("Error: No training samples found!")
        return
    
    # Split train/val by reciter (reduces leakage)
    train_indices, val_indices = _split_by_reciter(dataset, val_fraction=0.2)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_contrastive_batch,
        drop_last=True,  # Drop incomplete batches for contrastive learning
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_contrastive_batch,
        num_workers=0,
    )

    print("Building ayah text bank for global retrieval validation...")
    ayah_keys, ayah_input_ids, ayah_attention_mask = _build_ayah_text_bank_from_db(
        dataset.db,
        max_length=512,
    )
    print(f"Built text bank with {len(ayah_keys)} ayahs")
    ayah_key_to_idx = {k: i for i, k in enumerate(ayah_keys)}
    
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    # Initialize model
    print("Initializing contrastive model...")
    model = FastConformerContrastive(embed_dim=256)
    model.to(device)
    
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    # Optimizer - only trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.1
    )
    
    # Resume (if requested)
    start_epoch = 0
    if resume and last_checkpoint.exists():
        print(f"Resuming from {last_checkpoint}")
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        logs = checkpoint.get("logs", [])

    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            if batch is None:
                continue
            
            audio_features = batch["audio_features"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            logits_a, logits_t = model(audio_features, input_ids, attention_mask)
            loss = contrastive_loss(logits_a, logits_t)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            if device == "cuda":
                torch.cuda.empty_cache()

            # Precompute text bank once per epoch (GPU-only)
            all_text_embeds = []
            chunk_size = max(4, int(val_text_chunk_size))
            for start in range(0, ayah_input_ids.size(0), chunk_size):
                end = min(start + chunk_size, ayah_input_ids.size(0))
                text_emb = model.encode_text(
                    ayah_input_ids[start:end].to(device),
                    ayah_attention_mask[start:end].to(device),
                )
                all_text_embeds.append(text_emb.cpu())
            text_bank = torch.cat(all_text_embeds, dim=0).to(device)

            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                if batch is None:
                    continue
                
                audio_features = batch["audio_features"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                logits_a, logits_t = model(audio_features, input_ids, attention_mask)
                loss = contrastive_loss(logits_a, logits_t)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Global retrieval eval: query against all ayahs
                audio_emb = model.encode_audio(audio_features)

                sims = audio_emb @ text_bank.T
                top1 = sims.argmax(dim=1)
                top5 = sims.topk(k=min(5, sims.size(1)), dim=1).indices

                targets = []
                for s, a in zip(batch["surahs"], batch["ayahs"]):
                    targets.append(ayah_key_to_idx[(int(s), int(a))])
                target_idx = torch.tensor(targets, device=device)

                correct_top1 += (top1 == target_idx).sum().item()
                correct_top5 += sum(
                    (top5[i] == target_idx[i]).any().item() for i in range(top5.size(0))
                )
                total += top5.size(0)
        
        avg_val_loss = val_loss / max(val_batches, 1)
        top1_acc = correct_top1 / max(total, 1)
        top5_acc = correct_top5 / max(total, 1)
        
        # Logging
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_global_top1_acc": top1_acc,
            "val_global_top5_acc": top5_acc,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
              f"Global Top-1={top1_acc:.2%}, Global Top-5={top5_acc:.2%}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_global_top1_acc": top1_acc,
                "val_global_top5_acc": top5_acc,
            }
            torch.save(checkpoint, output_path / "best_model.pt")
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")
        
        # Save last checkpoint for resume
        last_ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "logs": logs,
        }
        torch.save(last_ckpt, last_checkpoint)
        
        # Step scheduler
        scheduler.step()
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Logs saved to {log_file}")
    print(f"Best model saved to {output_path / 'best_model.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train FastConformer contrastive model")
    parser.add_argument("--extract-features", action="store_true", help="Extract features from audio")
    parser.add_argument("--train", action="store_true", help="Train projection heads")
    parser.add_argument("--audio-dir", type=str, help="Directory containing audio files")
    parser.add_argument("--feature-dir", type=str, help="Directory with pre-extracted features")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--quran-json", type=str, help="Path to quran.json")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--feature-limit", type=int, default=0, help="Limit number of feature files for quick runs")
    parser.add_argument("--val-text-chunk-size", type=int, default=64, help="Ayah text encoding chunk size during validation")
    
    args = parser.parse_args()
    
    if args.extract_features:
        if not args.audio_dir:
            print("Error: --audio-dir required for feature extraction")
            return
        extract_features(
            audio_dir=args.audio_dir,
            output_dir=args.feature_dir or f"{args.output_dir}/features",
            batch_size=8,
            device=args.device,
        )
    
    if args.train:
        if not args.feature_dir:
            print("Error: --feature-dir required for training")
            return
        train_projection_heads(
            feature_dir=args.feature_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            quran_json_path=args.quran_json,
            resume=args.resume,
            feature_limit=args.feature_limit,
            val_text_chunk_size=args.val_text_chunk_size,
        )
    
    if not args.extract_features and not args.train:
        print("Error: Specify --extract-features or --train")
        parser.print_help()


if __name__ == "__main__":
    main()

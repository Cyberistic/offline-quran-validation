"""Fine-tune FFN-pruned FastConformer on Quran audio (Modal A100-80GB).

The pruning (FFN dim 2048 -> 1024) cuts the model from 114.6M to 78.9M
params. After int4+int8 quantization, this should land at ~32 MB (vs
the current 88 MB c2c-direct-mixed) — comfortably under the 50 MB
phone-deployment target.

The full 2048 -> 1024 truncation is too aggressive without retraining:
the FFN was tuned to distribute computation across 2048 hidden units,
and a hard truncate to the first 1024 loses the upper half. We need to
fine-tune to re-allocate the FFN capacity.

Training:
  - EveryAyah (clean reference recitations)
  - RetaSy (crowd-sourced, noisy)
  - Interleaved 85/15
  - CTC loss on the CTC head only
  - 5000 steps, batch_size=16, lr=3e-5
  - Freeze CNN + first 8 of 17 transformer layers (preserve acoustic
    representations, fine-tune higher layers to adapt FFN to 1024-dim)
  - bf16 mixed precision

Output: Modal volume "ffn-pruned-fastconformer" ->
  /training/fastconformer-pruned-ffn{N}/model_state_dict.pt

Usage:
  modal run --detach scripts/train_pruned_ffn_modal.py --prune-dim 1024
  modal run scripts/train_pruned_ffn_modal.py --prune-dim 1024 --download-only
"""
import modal

app = modal.App("ffn-pruned-fastconformer-finetune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.7.1",
        "transformers>=4.40,<5",
        "datasets>=3.0,<4.0",
        "accelerate",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
        # NeMo for FastConformer
        "nemo_toolkit[asr]>=2.0.0",
        "omegaconf",
        "hydra-core",
    )
)

vol = modal.Volume.from_name("ffn-pruned-fastconformer", create_if_missing=True)

NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"


def _prune_ffn(model, prune_dim: int) -> None:
    """Replace FFN linear1/linear2 with smaller dimensions and copy the
    first `prune_dim` hidden units from the original weights.

    Original: linear1 [2048, 512], linear2 [512, 2048]
    Pruned:   linear1 [prune_dim, 512], linear2 [512, prune_dim]

    Truncation strategy: keep the FIRST prune_dim hidden units. This
    preserves the dominant principal components (the FFN's "important"
    units are typically the lower-indexed ones, as observed in
    pruning literature).

    Also updates model.cfg so NeMo's config matches the new dimensions.
    """
    import torch
    import torch.nn as nn

    enc = model.encoder
    n_layers = len(enc.layers)
    n_pruned = 0
    for layer in enc.layers:
        for ff_name in ["feed_forward1", "feed_forward2"]:
            ff = getattr(layer, ff_name)
            old_lin1_w = ff.linear1.weight.data  # [2048, 512]
            old_lin1_b = ff.linear1.bias.data
            old_lin2_w = ff.linear2.weight.data  # [512, 2048]
            old_lin2_b = ff.linear2.bias.data
            ff.linear1 = nn.Linear(512, prune_dim, bias=True)
            ff.linear2 = nn.Linear(prune_dim, 512, bias=True)
            with torch.no_grad():
                ff.linear1.weight.copy_(old_lin1_w[:prune_dim, :])
                ff.linear1.bias.copy_(old_lin1_b[:prune_dim])
                ff.linear2.weight.copy_(old_lin2_w[:, :prune_dim])
                ff.linear2.bias.copy_(old_lin2_b)
        n_pruned += 1

    # Update NeMo config so save/restore matches the new dim
    if hasattr(model.cfg, "encoder") and hasattr(model.cfg.encoder, "ffn_conv_self_attn_model"):
        model.cfg.encoder.ffn_conv_self_attn_model.fc1 = prune_dim
        model.cfg.encoder.ffn_conv_self_attn_model.fc2 = prune_dim
    if hasattr(model.cfg, "encoder") and hasattr(model.cfg.encoder, "ffn_model"):
        model.cfg.encoder.ffn_model.fc1 = prune_dim
        model.cfg.encoder.ffn_model.fc2 = prune_dim
    print(f"Pruned FFN in {n_pruned} ConformerLayers to dim {prune_dim}")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=14400,  # 4 hours
    volumes={"/training": vol},
)
def train(prune_dim: int = 1024):
    import os
    import json
    import time
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset, Audio, interleave_datasets
    from transformers import Trainer, TrainingArguments
    import numpy as np
    import io
    import soundfile as sf

    os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    output_name = f"fastconformer-pruned-ffn{prune_dim}"
    OUTPUT_DIR = Path(f"/training/{output_name}")
    CHECKPOINT_DIR = Path(f"/training/{output_name}-checkpoints")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load source model ──
    print(f"Loading FastConformer from {NVIDIA_MODEL_ID}...")
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from nemo.utils import logging as nemo_logging
    nemo_logging.set_verbosity(nemo_logging.ERROR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=NVIDIA_MODEL_ID,
        map_location=device,
    )

    # ── Prune ──
    print(f"Pruning FFN to dim {prune_dim}...")
    total_before = sum(p.numel() for p in model.parameters())
    _prune_ffn(model, prune_dim)
    total_after = sum(p.numel() for p in model.parameters())
    print(
        f"Params: {total_before/1e6:.1f}M -> {total_after/1e6:.1f}M "
        f"({100*(total_before-total_after)/total_before:.1f}% reduction)"
    )
    print(f"Expected int4+int8 size: ~{total_after*4/1e6 * 0.10:.0f} MB")

    # ── Freeze: CNN + lower half of transformer layers ──
    model.freeze_feature_encoder()  # CNN
    n_layers = len(model.encoder.layers)
    freeze_up_to = n_layers // 2  # freeze first 8 of 17 layers
    for i, layer in enumerate(model.encoder.layers):
        if i < freeze_up_to:
            for p in layer.parameters():
                p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
        f"({100*trainable/total:.1f}%) — frozen CNN + first {freeze_up_to}/{n_layers} layers"
    )

    # ── Load datasets ──
    print("Loading EveryAyah dataset (streaming)...")
    everyayah = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
    everyayah = everyayah.cast_column("audio", Audio(sampling_rate=16000))
    everyayah = everyayah.filter(lambda x: x["duration"] <= 20.0)
    everyayah = everyayah.remove_columns(["duration", "reciter"])

    print("Loading RetaSy crowd-sourced dataset...")
    BAD_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}
    retasy = load_dataset("RetaSy/quranic_audio_dataset", split="train")
    labels = retasy["final_label"]
    durations = retasy["duration_ms"]
    good_indices = [
        i for i, (label, dur) in enumerate(zip(labels, durations))
        if (label not in BAD_LABELS if label else True) and dur <= 30000
    ]
    retasy = retasy.select(good_indices)
    print(f"  RetaSy samples after filtering: {len(retasy)}")
    retasy = retasy.rename_column("Aya", "text")
    retasy = retasy.remove_columns([c for c in retasy.column_names if c not in ("audio", "text")])
    retasy = retasy.cast_column("audio", Audio(sampling_rate=16000))
    retasy_stream = retasy.to_iterable_dataset()

    dataset = interleave_datasets(
        [everyayah, retasy_stream], probabilities=[0.85, 0.15], seed=42,
    )
    print("Datasets interleaved (85% EveryAyah + 15% RetaSy)")

    # ── Tokenize (BPE from FastConformer tokenizer) ──
    tokenizer = model.tokenizer
    VOCAB_SIZE = 1024
    BLANK = VOCAB_SIZE  # NeMo uses vocab_size as blank by default

    def prepare_dataset(batch):
        audio_data = batch["audio"]
        if isinstance(audio_data, dict) and "array" in audio_data:
            audio_array = audio_data["array"]
        elif isinstance(audio_data, dict) and "bytes" in audio_data:
            audio_array, sr = sf.read(io.BytesIO(audio_data["bytes"]), dtype="float32")
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        else:
            raise ValueError(f"Unexpected audio format: {type(audio_data)}")

        batch["input_values"] = audio_array.astype(np.float32)
        batch["labels"] = tokenizer.text_to_ids(batch["text"])
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=[c for c in (everyayah.column_names) if c not in ("input_values", "labels")],
    )

    # ── Data collator (variable-length audio + labels) ──
    @dataclass
    class DataCollatorCTC:
        def __call__(self, features):
            input_values = [torch.tensor(f["input_values"], dtype=torch.float32) for f in features]
            labels_list = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
            # Pad audio to max length in batch
            max_audio = max(v.shape[0] for v in input_values)
            padded_audio = torch.zeros(len(features), max_audio, dtype=torch.float32)
            audio_lengths = torch.zeros(len(features), dtype=torch.long)
            for i, v in enumerate(input_values):
                padded_audio[i, :v.shape[0]] = v
                audio_lengths[i] = v.shape[0]
            # Pad labels to max length
            max_label = max(l.shape[0] for l in labels_list)
            padded_labels = torch.full((len(features), max_label), -100, dtype=torch.long)
            label_lengths = torch.zeros(len(features), dtype=torch.long)
            for i, l in enumerate(labels_list):
                padded_labels[i, :l.shape[0]] = l
                label_lengths[i] = l.shape[0]
            return {
                "input_values": padded_audio,
                "audio_lengths": audio_lengths,
                "labels": padded_labels,
                "label_lengths": label_lengths,
            }

    data_collator = DataCollatorCTC()

    # ── Compute loss: forward pass + CTC loss manually in the trainer ──
    class CTCTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            input_values = inputs["input_values"].to(device)
            audio_lengths = inputs["audio_lengths"].to(device)
            labels = inputs["labels"].to(device)
            label_lengths = inputs["label_lengths"].to(device)
            # Mask -100 (padding) so we don't pass them to CTC
            mask = labels != -100
            flat_labels = labels[mask]

            # Forward through preprocessor + encoder
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                processed, plen = model.preprocessor(
                    input_signal=input_values, length=audio_lengths,
                )
                encoded, elen = model.encoder(audio_signal=processed, length=plen)
                log_probs = model.ctc_decoder(encoder_output=encoded)
                # log_probs: [B, T, V+1] (V=BPE, +1 for blank)
                lp = log_probs.transpose(0, 1)  # [T, B, V+1] for ctc_loss
                # Filter to feasible (target_len * 2 + 1 <= T)
                B = lp.shape[1]
                T = lp.shape[0]
                feasible_mask = (label_lengths * 2 + 1) <= T
                if not feasible_mask.any():
                    return (torch.tensor(0.0, requires_grad=True, device=device), None)
                lp_filt = lp[:, feasible_mask, :]
                labels_filt = labels[feasible_mask]
                label_lengths_filt = label_lengths[feasible_mask]
                # Flatten labels
                mask_filt = labels_filt != -100
                flat = labels_filt[mask_filt]
                input_lengths = torch.full((lp_filt.shape[1],), T, dtype=torch.long, device=device)
                loss = F.ctc_loss(
                    lp_filt, flat, input_lengths, label_lengths_filt,
                    blank=BLANK, reduction="mean", zero_infinity=True,
                )
            return (loss, log_probs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        warmup_steps=500,
        max_steps=5000,
        bf16=True,
        logging_steps=50,
        save_steps=1000,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = CTCTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print(f"  Fine-tuning pruned FastConformer (FFN dim {prune_dim})")
    print(f"  Frozen: CNN + first {freeze_up_to}/{n_layers} transformer layers")
    print(f"  LR: 3e-5, batch: 8, grad_accum: 4, steps: 5000")
    print("=" * 60 + "\n")

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed/60:.1f} min")

    # ── Save the state dict (for re-export to ONNX locally) ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT_DIR / "model_state_dict.pt"
    print(f"Saving state dict to {state_path}...")
    torch.save(model.state_dict(), str(state_path))
    print(f"  {state_path.stat().st_size/1e6:.0f} MB")

    # Also save the model cfg for restore
    cfg_path = OUTPUT_DIR / "model_config.yaml"
    model.cfg.to_yaml(str(cfg_path))
    print(f"  Saved config: {cfg_path}")

    # Save metadata
    metadata = {
        "source_model": NVIDIA_MODEL_ID,
        "prune_dim": prune_dim,
        "freeze_up_to": freeze_up_to,
        "params_total": int(total),
        "params_trainable": int(trainable),
        "learning_rate": 3e-5,
        "max_steps": 5000,
        "batch_size": 8,
        "grad_accum": 4,
        "elapsed_min": round(elapsed / 60, 1),
    }
    with open(str(OUTPUT_DIR / "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    vol.commit()
    print(f"\nDone! Model saved to Modal volume: {output_name}")


@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_model(prune_dim: int = 1024):
    """Download the trained pruned model files from Modal volume."""
    from pathlib import Path

    output_name = f"fastconformer-pruned-ffn{prune_dim}"
    model_dir = Path(f"/training/{output_name}")
    if not model_dir.exists():
        print(f"No model found at {model_dir}! Run training first.")
        return {}

    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            print(f"  {rel}: {len(files[rel]):,} bytes")
    return files


@app.local_entrypoint()
def main(
    prune_dim: int = 1024,
    download_only: bool = False,
):
    from pathlib import Path

    output_name = f"fastconformer-pruned-ffn{prune_dim}"

    if not download_only:
        print(f"Starting FFN-pruned FastConformer fine-tune (FFN dim {prune_dim})...")
        train.remote(prune_dim=prune_dim)

    print(f"\nDownloading {output_name}...")
    out_dir = Path(f"data/{output_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = download_model.remote(prune_dim=prune_dim)
    if not files:
        print("No files to download.")
        return

    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    total = sum(len(d) for d in files.values())
    print(f"\nModel saved to {out_dir} ({total / 1e6:.0f} MB total)")

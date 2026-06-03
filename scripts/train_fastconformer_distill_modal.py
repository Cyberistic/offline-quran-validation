"""Distill NVIDIA Arabic FastConformer into a smaller 8-layer student.

Teacher: full `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
Student: same width/tokenizer, but encoder layers pruned 17 -> 8 (evenly spaced)

Why this design:
- keeps tokenizer / hidden size / decoder identical, so weights can be copied
- estimated mixed int4+int8 size ~44 MB (from 88 MB teacher)
- much simpler and less risky than width-pruning

Training objective:
- alpha * CTC(student, labels)
- (1-alpha) * KL(student_ctc || teacher_ctc)

Artifacts are written to Modal volume `fastconformer-distill-quran` under:
  /training/<output_name>/
    manifests/
    student_state_dict.pt
    student_config.json
    training_metadata.json

This script intentionally does not auto-export ONNX; export/quantization comes
after we confirm the distilled student learns.
"""
from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import modal

app = modal.App("fastconformer-distill-quran")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.2",
        "nemo_toolkit[asr]>=2.7.0",
        "datasets>=3.0,<4.0",
        "lightning>=2.4,<3.0",
        "omegaconf>=2.3,<3.0",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("fastconformer-distill-quran", create_if_missing=True)

BASE_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
PRIMARY_DATASET_ID = "rabah2026/Quran-Ayah-Corpus"
RETASY_DATASET_ID = "RetaSy/quranic_audio_dataset"
BAD_RETASY_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}


def _evenly_spaced_indices(total_layers: int, keep_layers: int) -> list[int]:
    if keep_layers >= total_layers:
        return list(range(total_layers))
    if keep_layers <= 1:
        return [total_layers - 1]
    raw = [round(i * (total_layers - 1) / (keep_layers - 1)) for i in range(keep_layers)]
    fixed = []
    prev = -1
    for idx in raw:
        idx = max(idx, prev + 1)
        fixed.append(min(idx, total_layers - 1))
        prev = fixed[-1]
    return fixed


def _install_kaldialign_fallback() -> None:
    import sys
    import types
    try:
        import kaldialign  # noqa: F401
        return
    except Exception:
        pass

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
                ins = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                best = min(sub, ins, delete)
                dp[i][j] = best
                bt[i][j] = "S" if best == sub else ("I" if best == ins else "D")
        out = []
        i, j = n, m
        while i > 0 or j > 0:
            move = bt[i][j]
            if move == "S":
                out.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif move == "I":
                out.append((eps, hyp[j - 1]))
                j -= 1
            else:
                out.append((ref[i - 1], eps))
                i -= 1
        return list(reversed(out))

    module = types.ModuleType("kaldialign")
    module.align = align
    sys.modules["kaldialign"] = module


@dataclass
class BuildStats:
    written: int = 0
    skipped_duration: int = 0
    skipped_empty: int = 0
    bytes_written: int = 0


@app.function(
    image=image,
    timeout=14400,
    volumes={"/training": vol},
)
def prepare_data(
    output_name: str = "fastconformer-distill-8l-v1",
    max_train_samples: int = 120_000,
    max_val_samples: int = 8_000,
    include_retasy: bool = True,
    max_retasy_samples: int = 6_000,
    min_duration: float = 0.6,
    max_duration: float = 30.0,
    force_rebuild: bool = False,
):
    import io
    import soundfile as sf
    from datasets import Audio, load_dataset

    base = Path(f"/training/{output_name}")
    manifests_dir = base / "manifests"
    audio_root = base / "audio"
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    metadata_path = manifests_dir / "data_metadata.json"

    if train_manifest.exists() and val_manifest.exists() and not force_rebuild:
        print("Manifests already exist, skipping rebuild.")
        return

    manifests_dir.mkdir(parents=True, exist_ok=True)
    (audio_root / "train").mkdir(parents=True, exist_ok=True)
    (audio_root / "val").mkdir(parents=True, exist_ok=True)
    (audio_root / "retasy").mkdir(parents=True, exist_ok=True)

    def _safe_text(text: str) -> str:
        text = (text or "").strip()
        return " ".join(text.split())

    def write_audio(audio_obj, out_path: Path) -> float:
        if isinstance(audio_obj, dict) and audio_obj.get("bytes") is not None:
            raw = audio_obj["bytes"]
            out_path.write_bytes(raw)
            try:
                with sf.SoundFile(io.BytesIO(raw)) as f:
                    return float(len(f) / float(f.samplerate))
            except Exception:
                return -1.0
        if isinstance(audio_obj, dict) and audio_obj.get("array") is not None:
            arr = audio_obj["array"]
            sr = int(audio_obj.get("sampling_rate", 16000))
            sf.write(out_path, arr, sr)
            return float(len(arr) / max(sr, 1))
        raise ValueError(f"Unsupported audio object format: {type(audio_obj)}")

    def build_split(split: str, limit: int, subdir: str):
        stats = BuildStats()
        ds = load_dataset(PRIMARY_DATASET_ID, split=split, streaming=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))
        manifest_path = train_manifest if split == "train" else val_manifest
        with manifest_path.open("w", encoding="utf-8") as mf:
            for idx, sample in enumerate(ds):
                if stats.written >= limit:
                    break
                text = _safe_text(sample.get("text", ""))
                if not text:
                    stats.skipped_empty += 1
                    continue
                duration = float(sample.get("duration", -1.0))
                if duration > 0 and (duration < min_duration or duration > max_duration):
                    stats.skipped_duration += 1
                    continue
                out_file = audio_root / subdir / f"{split}_{idx:09d}.wav"
                measured = write_audio(sample["audio"], out_file)
                effective = duration if duration > 0 else measured
                if effective <= 0 or effective < min_duration or effective > max_duration:
                    stats.skipped_duration += 1
                    out_file.unlink(missing_ok=True)
                    continue
                stats.written += 1
                stats.bytes_written += out_file.stat().st_size
                mf.write(json.dumps({
                    "audio_filepath": str(out_file),
                    "duration": round(float(effective), 4),
                    "text": text,
                }, ensure_ascii=False) + "\n")
        return stats

    print(f"Building manifests from {PRIMARY_DATASET_ID}...")
    train_stats = build_split("train", max_train_samples, "train")
    val_stats = build_split("validation", max_val_samples, "val")

    retasy_stats = BuildStats()
    if include_retasy and max_retasy_samples > 0:
        from datasets import Audio, load_dataset
        retasy = load_dataset(RETASY_DATASET_ID, split="train")
        retasy = retasy.cast_column("audio", Audio(sampling_rate=16000, decode=False))
        with train_manifest.open("a", encoding="utf-8") as mf:
            for idx, sample in enumerate(retasy):
                if retasy_stats.written >= max_retasy_samples:
                    break
                label = sample.get("final_label")
                if label in BAD_RETASY_LABELS:
                    continue
                text = _safe_text(sample.get("Aya", ""))
                if not text:
                    continue
                dur_ms = float(sample.get("duration_ms", -1.0))
                if dur_ms <= 0:
                    continue
                duration = dur_ms / 1000.0
                if duration < min_duration or duration > max_duration:
                    continue
                out_file = audio_root / "retasy" / f"retasy_{idx:09d}.wav"
                write_audio(sample["audio"], out_file)
                retasy_stats.written += 1
                mf.write(json.dumps({
                    "audio_filepath": str(out_file),
                    "duration": round(duration, 4),
                    "text": text,
                }, ensure_ascii=False) + "\n")

    metadata = {
        "train_written": train_stats.written + retasy_stats.written,
        "val_written": val_stats.written,
        "retasy_written": retasy_stats.written,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    vol.commit()
    print(f"Prepared data: train={metadata['train_written']}, val={metadata['val_written']}")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=14400,
    volumes={"/training": vol},
    memory=32768,
)
def train(
    output_name: str = "fastconformer-distill-8l-v1",
    student_layers: int = 8,
    alpha: float = 0.5,
    temperature: float = 2.0,
    max_steps: int = 6000,
    train_batch_size: int = 8,
    grad_accum: int = 4,
    learning_rate: float = 3e-5,
    freeze_preprocessor: bool = True,
):
    import time
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from omegaconf import open_dict
    import lightning.pytorch as pl

    _install_kaldialign_fallback()
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(f"/training/{output_name}")
    manifests_dir = base / "manifests"
    train_manifest = manifests_dir / "train_manifest.jsonl"
    val_manifest = manifests_dir / "val_manifest.jsonl"
    out_dir = base / "student"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError("Missing manifests. Run prepare_data first.")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading teacher: {BASE_MODEL_ID}")
    teacher = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        map_location=device,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("Building 8-layer student from teacher...")
    student = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        map_location=device,
    )
    keep_indices = _evenly_spaced_indices(len(student.encoder.layers), int(student_layers))
    student.encoder.layers = torch.nn.ModuleList([student.encoder.layers[i] for i in keep_indices])
    with open_dict(student.cfg):
        student.cfg.encoder.n_layers = int(student_layers)
        student.cfg.train_ds.manifest_filepath = str(train_manifest)
        student.cfg.train_ds.is_tarred = False
        student.cfg.train_ds.tarred_audio_filepaths = None
        student.cfg.train_ds.batch_size = int(train_batch_size)
        student.cfg.train_ds.shuffle = True
        student.cfg.train_ds.num_workers = 8
        student.cfg.train_ds.pin_memory = True
        student.cfg.train_ds.max_duration = 30.0
        student.cfg.train_ds.min_duration = 0.6
        student.cfg.validation_ds.manifest_filepath = str(val_manifest)
        student.cfg.validation_ds.batch_size = 8
        student.cfg.validation_ds.shuffle = False
        student.cfg.validation_ds.num_workers = 8
        student.cfg.validation_ds.pin_memory = True
        student.cfg.tokenizer.dir = "/tmp"
    if freeze_preprocessor and hasattr(student, "preprocessor"):
        for p in student.preprocessor.parameters():
            p.requires_grad = False

    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student keeps layers {keep_indices}")
    print(f"Student params: {total_params/1e6:.1f}M (estimated mixed ~{88*total_params/sum(p.numel() for p in teacher.parameters()):.1f} MB)")

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=int(max_steps),
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        default_root_dir=str(base / "checkpoints"),
        accumulate_grad_batches=int(grad_accum),
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    student.set_trainer(trainer)
    student.setup_training_data(student.cfg.train_ds)

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=float(learning_rate),
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(max_steps), eta_min=1e-6)

    print("\n" + "=" * 72)
    print("FastConformer distillation")
    print(f"student_layers: {student_layers}")
    print(f"alpha:          {alpha}")
    print(f"temperature:    {temperature}")
    print(f"batch:          {train_batch_size} x {grad_accum}")
    print(f"max_steps:      {max_steps}")
    print("=" * 72 + "\n")

    teacher.to(device)
    student.to(device)
    student.train()
    optimizer.zero_grad()
    step = 0
    micro = 0
    start = time.time()

    while step < int(max_steps):
        for batch in student._train_dl:
            if step >= int(max_steps):
                break
            try:
                signal, signal_len, transcript, transcript_len = batch
                signal = signal.to(device)
                signal_len = signal_len.to(device)
                transcript = transcript.to(device)
                transcript_len = transcript_len.to(device)

                with torch.no_grad():
                    t_processed, t_len = teacher.preprocessor(input_signal=signal, length=signal_len)
                    t_encoded, t_len = teacher.encoder(audio_signal=t_processed, length=t_len)
                    teacher_logits = teacher.ctc_decoder(encoder_output=t_encoded)

                s_processed, s_len = student.preprocessor(input_signal=signal, length=signal_len)
                s_encoded, s_len = student.encoder(audio_signal=s_processed, length=s_len)
                student_logits = student.ctc_decoder(encoder_output=s_encoded)

                T_student = student_logits.size(1)
                T_teacher = teacher_logits.size(1)
                if T_teacher != T_student:
                    teacher_aligned = F.interpolate(
                        teacher_logits.permute(0, 2, 1),
                        size=T_student,
                        mode="linear",
                        align_corners=False,
                    ).permute(0, 2, 1)
                else:
                    teacher_aligned = teacher_logits

                log_probs = F.log_softmax(student_logits, dim=-1).permute(1, 0, 2)
                ctc = F.ctc_loss(
                    log_probs,
                    transcript,
                    s_len,
                    transcript_len,
                    blank=student.decoder.blank_idx if hasattr(student.decoder, 'blank_idx') else student.joint.num_classes_with_blank - 1,
                    reduction="mean",
                    zero_infinity=True,
                )

                student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
                teacher_probs = F.softmax(teacher_aligned / temperature, dim=-1)
                kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
                loss = (alpha * ctc + (1 - alpha) * kl) / int(grad_accum)
                loss.backward()
                micro += 1

                if micro % int(grad_accum) == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    if step % 50 == 0:
                        print(f"step {step}/{max_steps}: loss={loss.item()*grad_accum:.4f} ctc={ctc.item():.4f} kl={kl.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                    if step % 1000 == 0:
                        torch.save(student.state_dict(), str(out_dir / f"student_step_{step}.pt"))
                        vol.commit()
            except Exception as e:
                print(f"Skipping batch: {e}")
                optimizer.zero_grad()
                micro = 0
                continue

    elapsed = time.time() - start
    state_path = out_dir / "student_state_dict.pt"
    torch.save(student.state_dict(), str(state_path))
    (out_dir / "student_config.json").write_text(
        json.dumps({"student_layers": student_layers, "keep_indices": keep_indices}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "training_metadata.json").write_text(
        json.dumps({
            "base_model": BASE_MODEL_ID,
            "student_layers": student_layers,
            "keep_indices": keep_indices,
            "alpha": alpha,
            "temperature": temperature,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "elapsed_min": round(elapsed / 60, 1),
            "params_total": total_params,
        }, indent=2),
        encoding="utf-8",
    )
    vol.commit()
    print(f"Done! Saved distilled student to {out_dir}")


@app.function(image=image, volumes={"/training": vol})
def download_model(output_name: str = "fastconformer-distill-8l-v1"):
    model_dir = Path(f"/training/{output_name}/student")
    if not model_dir.exists():
        print(f"No distilled model found at {model_dir}")
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
    output_name: str = "fastconformer-distill-8l-v1",
    student_layers: int = 8,
    max_steps: int = 6000,
    alpha: float = 0.5,
    temperature: float = 2.0,
    download_only: bool = False,
):
    from pathlib import Path

    if not download_only:
        print("Preparing manifests/audio on Modal volume...")
        prepare_data.remote(output_name=output_name)
        print("\nStarting distillation job...")
        train.remote(
            output_name=output_name,
            student_layers=student_layers,
            max_steps=max_steps,
            alpha=alpha,
            temperature=temperature,
        )

    print(f"\nDownloading model artifacts for {output_name}...")
    out_dir = Path("data") / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = download_model.remote(output_name=output_name)
    if not files:
        print("No files to download.")
        return
    total = 0
    for rel, data in files.items():
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        total += len(data)
        print(f"  saved {rel} ({len(data):,} bytes)")
    print(f"\nDownloaded to {out_dir} ({total / 1e6:.1f} MB)")

#!/usr/bin/env python3
"""Download Quran-MD ayah audio and build manifest.

Dataset: Buraaq/quran-md-ayahs (187K samples, 30 reciters)

Outputs:
  - <output_dir>/audio/*.wav
  - <output_dir>/manifest.json

Example:
  .venv/bin/python scripts/download_quran_md.py --output-dir data/quran-md
  .venv/bin/python scripts/download_quran_md.py --output-dir data/quran-md --limit 1000
"""

import argparse
import io
import json
from datetime import datetime
from pathlib import Path
import shutil

import soundfile as sf
from datasets import Audio, load_dataset, load_dataset_builder
from tqdm import tqdm


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Quran-MD dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/quran-md",
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max samples to download (0 = all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Buraaq/quran-md-ayahs (streaming)...")
    ds = load_dataset("Buraaq/quran-md-ayahs", split=args.split, streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    total = None
    try:
        builder = load_dataset_builder("Buraaq/quran-md-ayahs")
        split_info = builder.info.splits.get(args.split)
        if split_info is not None:
            total = split_info.num_examples
    except Exception:
        pass

    manifest_path = output_dir / "manifest.json"
    existing_entries = []
    existing_files = set()
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            existing_entries = data.get("samples", [])
            existing_files = {e.get("file") for e in existing_entries if e.get("file")}
        except json.JSONDecodeError:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = manifest_path.with_suffix(f".json.bak.{timestamp}")
            shutil.move(manifest_path, backup_path)
            print(f"Warning: invalid manifest. Backed up to {backup_path}")

    count = len(existing_entries)
    target_limit = args.limit if args.limit and args.limit > 0 else None
    if target_limit is not None and count >= target_limit:
        print(f"Manifest already has {count} samples (limit={target_limit}).")
        return

    pbar = tqdm(ds, desc="Downloading", total=total)
    for sample in pbar:
        if target_limit is not None and count >= target_limit:
            break

        try:
            surah = int(sample.get("surah"))
            ayah = int(sample.get("ayah"))
        except Exception:
            continue

        reciter = sample.get("reciter", "unknown")
        reciter_safe = _safe_name(str(reciter))
        audio = sample.get("audio")
        if not audio:
            continue

        rel_path = f"audio/{surah:03d}_{ayah:03d}_{reciter_safe}.wav"
        if rel_path in existing_files:
            continue

        audio_path = output_dir / rel_path
        if audio_path.exists():
            existing_files.add(rel_path)
            continue

        try:
            if "array" in audio and "sampling_rate" in audio:
                sf.write(audio_path, audio["array"], audio["sampling_rate"])
            else:
                raw_bytes = audio.get("bytes")
                if not raw_bytes:
                    continue
                data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                sf.write(audio_path, data, sr)
        except Exception:
            continue

        entry = {
            "id": f"quran_md_{count}",
            "file": rel_path,
            "surah": surah,
            "ayah": ayah,
            "ayah_end": None,
            "source": "quran-md",
            "reciter": reciter,
        }

        existing_entries.append(entry)
        existing_files.add(rel_path)
        count += 1

    with open(manifest_path, "w") as f:
        json.dump({"samples": existing_entries}, f, ensure_ascii=True)

    print(f"Saved {count} samples to {manifest_path}")


if __name__ == "__main__":
    main()

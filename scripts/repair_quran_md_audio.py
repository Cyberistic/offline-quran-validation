#!/usr/bin/env python3
"""Repair corrupted Quran-MD audio files by re-downloading them.

Reads a list of bad file paths and re-downloads matching items from
Buraaq/quran-md-ayahs, overwriting the corrupted files.

Example:
  .venv/bin/python scripts/repair_quran_md_audio.py \
    --root /Volumes/brrrrrr/quran \
    --bad-list /Volumes/brrrrrr/quran/bad_audio.txt
"""

import argparse
import json
import re
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


_PATH_RE = re.compile(r".*/audio/ayahs/(?P<reciter>[^/]+)/(?P<stem>\d{6})\.mp3$")


def _parse_bad_list(bad_list_path: Path) -> dict[tuple[str, int, int], str]:
    """Return mapping (reciter, surah, ayah) -> relative path."""
    targets = {}
    for line in bad_list_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _PATH_RE.match(line)
        if not match:
            continue
        reciter = match.group("reciter")
        stem = match.group("stem")
        surah = int(stem[:3])
        ayah = int(stem[3:])
        rel_path = f"audio/ayahs/{reciter}/{stem}.mp3"
        targets[(reciter, surah, ayah)] = rel_path
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Quran-MD audio")
    parser.add_argument("--root", type=str, required=True, help="Dataset root")
    parser.add_argument("--bad-list", type=str, required=True, help="Bad file list")
    args = parser.parse_args()

    root = Path(args.root)
    bad_list_path = Path(args.bad_list)
    targets = _parse_bad_list(bad_list_path)
    if not targets:
        print("No valid targets found in bad list.")
        return

    print(f"Repairing {len(targets)} files from Quran-MD...")

    ds = load_dataset("Buraaq/quran-md-ayahs", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    repaired = 0
    for sample in tqdm(ds, desc="Repairing"):
        reciter = sample.get("reciter")
        surah = sample.get("surah")
        ayah = sample.get("ayah")
        key = (str(reciter), int(surah), int(ayah))
        rel_path = targets.get(key)
        if not rel_path:
            continue

        audio = sample.get("audio")
        if not audio or "array" not in audio:
            continue

        out_path = root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio["array"], audio["sampling_rate"])
        repaired += 1

        if repaired >= len(targets):
            break

    print(f"Repaired {repaired} files")


if __name__ == "__main__":
    main()

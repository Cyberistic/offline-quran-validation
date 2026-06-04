#!/usr/bin/env python3
"""Build manifest.json from extracted Quran-MD archive.

Expected layout:
  <root>/audio/ayahs/<reciter>/<SSSAAA>.mp3

Outputs:
  <root>/manifest.json
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Quran-MD manifest")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder containing audio/ayahs",
    )
    args = parser.parse_args()

    root = Path(args.root)
    ayahs_dir = root / "audio" / "ayahs"
    if not ayahs_dir.exists():
        raise SystemExit(f"Missing {ayahs_dir}")

    manifest_path = root / "manifest.json"
    count = 0

    with open(manifest_path, "w") as f:
        f.write('{"samples":[\n')
        first = True

        for reciter_dir in sorted(ayahs_dir.iterdir()):
            if not reciter_dir.is_dir():
                continue
            reciter = reciter_dir.name
            for audio_file in sorted(reciter_dir.glob("*.mp3")):
                stem = audio_file.stem
                if len(stem) != 6 or not stem.isdigit():
                    continue
                surah = int(stem[:3])
                ayah = int(stem[3:])

                rel_path = str(audio_file.relative_to(root))
                entry = {
                    "id": f"quran_md_{count}",
                    "file": rel_path,
                    "surah": surah,
                    "ayah": ayah,
                    "ayah_end": None,
                    "source": "quran-md",
                    "reciter": reciter,
                }

                if not first:
                    f.write(",\n")
                json.dump(entry, f, ensure_ascii=True)
                first = False
                count += 1

        f.write("\n]}")

    print(f"Saved {count} samples to {manifest_path}")


if __name__ == "__main__":
    main()

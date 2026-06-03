#!/usr/bin/env python3
"""Debug top-k predictions for short benchmark samples."""

import json
from pathlib import Path
import importlib.util


ROOT = Path(__file__).parent.parent
MANIFEST = ROOT / "benchmark" / "test_corpus" / "manifest.json"
CORPUS = ROOT / "benchmark" / "test_corpus"
RUN_PY = ROOT / "experiments" / "fastconformer-contrastive" / "run.py"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("fc_contrastive_run", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)

    samples = [s for s in manifest["samples"] if s.get("category") == "short"]
    mod = load_module(RUN_PY)

    for s in samples:
        audio_path = CORPUS / s["file"]
        topk = mod.debug_topk(str(audio_path), top_k=5)
        print(f"\n{s['id']} expected={s['surah']}:{s['ayah']} file={s['file']}")
        for i, cand in enumerate(topk, start=1):
            print(
                f"  {i}. {cand['surah']}:{cand['ayah']} score={cand['score']:.4f}"
            )


if __name__ == "__main__":
    main()

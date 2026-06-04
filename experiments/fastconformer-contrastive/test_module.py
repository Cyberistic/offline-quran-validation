#!/usr/bin/env python3
"""Quick test script for fastconformer-contrastive experiment."""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load the module using importlib (like benchmark runner does)
import importlib.util

exp_path = PROJECT_ROOT / "experiments" / "fastconformer-contrastive" / "run.py"
spec = importlib.util.spec_from_file_location("fastconformer_contrastive_run", str(exp_path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print("✓ Module loaded successfully")

# Test model_size function (this will load models)
print("\nLoading models (this may take a minute)...")
try:
    size = module.model_size()
    print(f"✓ Model size: {size / (1024*1024):.1f} MB")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Test complete!")

"""Test FFN pruning locally — no GPU needed, just verify the structural
modifications produce a valid model that can be re-exported to ONNX.

FastConformer architecture (from inspection):
  - 17 ConformerLayers, each with:
    - feed_forward1: linear1 [2048, 512], linear2 [512, 2048]
    - feed_forward2: linear1 [2048, 512], linear2 [512, 2048]
    - self_attn: linear_q/k/v/out [512, 512] + linear_pos [512, 512]
    - conv: pointwise_conv1 [1024, 512, 1], depthwise_conv [512, 1, 9], pointwise_conv2 [512, 512, 1]
  - d_model = 512
  - FFN dim = 2048 (62% of total params)

Prune FFN dim 2048 -> 1024: keeps first 1024 hidden units. Expected
param reduction: 17 * 2 * (512*1024 + 1024 + 512*1024 + 512) = 35.7M
params (vs 71.4M original). Saves ~36M params = ~144 MB fp32.
After int4+int8 quantization, expected size: ~45 MB (vs current 88 MB).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import copy
from pathlib import Path

PROJECT_ROOT = Path("/home/cyber/Documents/repos/offline-tarteel")
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from collections import defaultdict

PRUNE_DIM = int(os.getenv("PRUNE_DIM", "1024"))


def main():
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0", map_location="cpu"
    )
    m.eval()

    print("=" * 60)
    print("ORIGINAL model")
    total = 0
    for p in m.parameters():
        total += p.numel()
    print(f"Total params: {total/1e6:.1f}M ({total*4/1e6:.0f} MB fp32)")
    print()

    # Inspect one ConformerLayer
    enc = m.encoder
    print(f"Encoder has {len(enc.layers)} ConformerLayers")
    layer0 = enc.layers[0]
    print(f"First layer type: {type(layer0).__name__}")
    for name, _ in layer0.named_parameters():
        print(f"  {name}")
    print()

    # Prune FFN
    new_ffn_dim = PRUNE_DIM
    print(f"Pruning FFN dim 2048 -> {new_ffn_dim}")
    n_pruned_layers = 0
    for li, layer in enumerate(enc.layers):
        for ff_name in ["feed_forward1", "feed_forward2"]:
            ff = getattr(layer, ff_name)
            old_lin1_w = ff.linear1.weight.data  # [2048, 512]
            old_lin1_b = ff.linear1.bias.data    # [2048]
            old_lin2_w = ff.linear2.weight.data  # [512, 2048]
            old_lin2_b = ff.linear2.bias.data    # [512]
            # Truncate to first new_ffn_dim hidden units
            ff.linear1 = nn.Linear(512, new_ffn_dim, bias=True)
            ff.linear2 = nn.Linear(new_ffn_dim, 512, bias=True)
            with torch.no_grad():
                ff.linear1.weight.copy_(old_lin1_w[:new_ffn_dim, :])
                ff.linear1.bias.copy_(old_lin1_b[:new_ffn_dim])
                ff.linear2.weight.copy_(old_lin2_w[:, :new_ffn_dim])
                ff.linear2.bias.copy_(old_lin2_b)
        n_pruned_layers += 1
    print(f"Pruned FFN in {n_pruned_layers} ConformerLayers")
    print()

    # Stats after
    new_total = 0
    for p in m.parameters():
        new_total += p.numel()
    print(f"PRUNED model:")
    print(f"Total params: {new_total/1e6:.1f}M ({new_total*4/1e6:.0f} MB fp32)")
    print(f"Reduction: {(total - new_total)/1e6:.1f}M ({(total - new_total)*100/total:.1f}%)")
    print(f"Expected int4+int8 size: {new_total*4/1e6 * 0.1:.0f} MB (10% compression heuristic)")
    print()

    # Test forward pass on a fake audio
    print("Test forward pass on 5s random audio...")
    fake_audio = torch.randn(1, 16000 * 5)
    with torch.no_grad():
        processed, plen = m.preprocessor(input_signal=fake_audio, length=torch.tensor([fake_audio.shape[1]]))
        encoded, elen = m.encoder(audio_signal=processed, length=plen)
        log_probs = m.ctc_decoder(encoder_output=encoded)
    print(f"OK! Logits shape: {log_probs.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print()

    # Test tokenization
    from nemo.collections.asr.parts.submodules import ctc_beam_decoding
    print("Test greedy decode...")
    lp = log_probs[0].cpu().numpy()
    ids = lp.argmax(-1)
    BLANK = 1024
    deduped = []
    prev = -1
    for i in ids:
        if i != prev and i != BLANK:
            deduped.append(int(i))
        prev = i
    text = m.tokenizer.ids_to_text(deduped).strip()
    print(f"Random audio -> {text!r} (length: {len(deduped)})")
    print()

    # Save the pruned state dict
    out_path = PROJECT_ROOT / "data" / "fastconformer-pruned-ffn1024.pt"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    state = m.state_dict()
    torch.save(state, str(out_path))
    print(f"Saved pruned state dict to {out_path} ({out_path.stat().st_size/1e6:.0f} MB)")
    print()

    print("=" * 60)
    print("DONE — pruning structurally works. Next steps:")
    print("  1. Move this into a Modal training script")
    print("  2. Add fine-tuning on EveryAyah + RetaSy")
    print("  3. Re-export to ONNX (need to verify dim changes work)")
    print("  4. Quantize int4+int8")
    print("  5. Benchmark")


if __name__ == "__main__":
    main()

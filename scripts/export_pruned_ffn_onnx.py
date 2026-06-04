"""Export the FFN-pruned FastConformer (FFN dim 1024) to ONNX.

Same approach as export_fastconformer_onnx.py but:
- Prune FFN to dim 1024 first
- Load the trained state dict from data/fastconformer-pruned-ffn1024/

Output: data/onnx_export/fastconformer_full_pruned.onnx (~316 MB fp32)
Then quantize with scripts/quantize_mixed.py.
"""
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cudnn
torch.backends.cudnn.flags(enabled=False)

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
PRUNE_DIM = 1024
PRUNED_STATE_DICT = PROJECT_ROOT / "data" / "fastconformer-pruned-ffn1024" / "model_state_dict.pt"
OUT_DIR = PROJECT_ROOT / "data" / "onnx_export"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fastconformer_full_pruned.onnx"

EXAMPLE_SEC = 4.0
SAMPLE_RATE = 16000


def _real_stft(self: FilterbankFeatures, x: torch.Tensor) -> torch.Tensor:
    n_fft = self.n_fft
    hop = self.hop_length
    win_len = self.win_length
    center = not self.exact_pad
    window = self.window.to(dtype=torch.float, device=x.device)

    if self.stft_pad_amount is not None:
        pad = self.stft_pad_amount
        x = F.pad(x.unsqueeze(1), (pad, pad), "constant", 0.0).squeeze(1)
    elif center:
        x = F.pad(x, (n_fft // 2, n_fft // 2), "constant", 0.0)

    B, T_audio = x.shape
    T_frames = 1 + (T_audio - n_fft) // hop
    offsets = torch.arange(n_fft, device=x.device, dtype=torch.int64)
    starts = torch.arange(0, T_frames * hop, hop, device=x.device, dtype=torch.int64)
    idx_2d = starts.unsqueeze(1) + offsets.unsqueeze(0)
    idx_1d = (idx_2d.unsqueeze(0) + torch.arange(B, device=x.device, dtype=torch.int64).unsqueeze(1).unsqueeze(2) * T_audio).reshape(-1)
    x_flat = x.reshape(-1)
    x = torch.index_select(x_flat, 0, idx_1d).reshape(B, T_frames, n_fft)
    if win_len < n_fft:
        pad_left = (n_fft - win_len) // 2
        pad_right = n_fft - win_len - pad_left
        window = F.pad(window, (pad_left, pad_right), "constant", 0.0)
    x = x * window

    k = torch.arange(n_fft // 2 + 1, device=x.device, dtype=torch.float32)
    n = torch.arange(n_fft, device=x.device, dtype=torch.float32)
    angles = 2 * 3.14159265 * k.unsqueeze(1) * n.unsqueeze(0) / n_fft
    W_re = torch.cos(angles)
    W_im = torch.sin(angles)
    X_real = torch.matmul(x, W_re.t())
    X_imag = torch.matmul(x, W_im.t())
    mag = torch.sqrt(X_real * X_real + X_imag * X_imag + 1e-9)
    mag = mag.transpose(-1, -2)  # [B, n_fft//2+1, T_frames]
    return mag


def _featurizer_forward(self: FilterbankFeatures, x: torch.Tensor, seq_len: torch.Tensor, linear_spec: bool = False):
    """Mirror of the original FilterbankFeatures.forward, but with our real STFT."""
    seq_len = self.get_seq_len(seq_len)
    if self.exact_pad:
        x = _real_stft(self, x)
    else:
        x = self.stft(x) if hasattr(self.stft, '__call__') else _real_stft(self, x)
        # If patching didn't take, force real STFT
        if x.is_complex():
            x = _real_stft(self, x.to(torch.float) if torch.is_tensor(x) else x)

    if self.mag_power != 1.0:
        x = x.pow(self.mag_power)

    if linear_spec:
        return x, seq_len

    with torch.amp.autocast(x.device.type, enabled=False):
        x = torch.matmul(self.fb.to(x.dtype), x)

    if self.log:
        if self.log_zero_guard_type == "add":
            x = torch.log(x + self.log_zero_guard_value_fn(x))
        elif self.log_zero_guard_type == "clamp":
            x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
        else:
            raise ValueError("log_zero_guard_type was not understood")

    if self.frame_splicing > 1:
        from nemo.collections.asr.parts.preprocessing.features import splice_frames
        x = splice_frames(x, self.frame_splicing)

    if self.normalize:
        from nemo.collections.asr.parts.preprocessing.features import normalize_batch
        x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

    max_len = x.size(-1)
    mask = torch.arange(max_len, device=x.device)
    mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
    x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
    del mask
    pad_to = self.pad_to
    if pad_to == "max":
        x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
    elif pad_to > 0:
        pad_amt = x.size(-1) % pad_to
        if pad_amt != 0:
            x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
    return x, seq_len


def patch_featurizer(feat: FilterbankFeatures) -> None:
    feat.stft = types.MethodType(_real_stft, feat)
    feat.forward = types.MethodType(_featurizer_forward, feat)


def prune_ffn(model: nn.Module, new_dim: int) -> None:
    enc = model.encoder
    n_pruned = 0
    for li, layer in enumerate(enc.layers):
        for ff_name in ["feed_forward1", "feed_forward2"]:
            ff = getattr(layer, ff_name)
            old_lin1_w = ff.linear1.weight.data
            old_lin1_b = ff.linear1.bias.data
            old_lin2_w = ff.linear2.weight.data
            old_lin2_b = ff.linear2.bias.data
            ff.linear1 = nn.Linear(512, new_dim, bias=True)
            ff.linear2 = nn.Linear(new_dim, 512, bias=True)
            with torch.no_grad():
                ff.linear1.weight.copy_(old_lin1_w[:new_dim, :])
                ff.linear1.bias.copy_(old_lin1_b[:new_dim])
                ff.linear2.weight.copy_(old_lin2_w[:, :new_dim])
                ff.linear2.bias.copy_(old_lin2_b)
        n_pruned += 1
    print(f"Pruned FFN in {n_pruned} ConformerLayers to dim {new_dim}")


class FullCTCPipeline(torch.nn.Module):
    def __init__(self, nemo_model, patched: bool = True):
        super().__init__()
        if not patched:
            patch_featurizer(nemo_model.preprocessor.featurizer)
        self.preprocessor = nemo_model.preprocessor
        self.encoder = nemo_model.encoder
        self.ctc_decoder = nemo_model.ctc_decoder
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        processed, plen = self.preprocessor(input_signal=audio_signal, length=length)
        encoded, _ = self.encoder(audio_signal=processed, length=plen)
        log_probs = self.ctc_decoder(encoder_output=encoded)
        return log_probs


def main():
    print(f"Loading {NVIDIA_MODEL_ID}...")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(NVIDIA_MODEL_ID, map_location="cpu")
    model.eval()

    print(f"Pruning FFN to dim {PRUNE_DIM}...")
    prune_ffn(model, PRUNE_DIM)

    print(f"Loading trained state dict from {PRUNED_STATE_DICT}...")
    state = torch.load(str(PRUNED_STATE_DICT), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: missing keys: {len(missing)}")
        for k in missing[:3]:
            print(f"    {k}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {len(unexpected)}")
        for k in unexpected[:3]:
            print(f"    {k}")
    print(f"  Loaded state dict")

    total = sum(p.numel() for p in model.parameters())
    print(f"Pruned model: {total/1e6:.1f}M params ({total*4/1e6:.0f} MB fp32)")

    fake_audio = torch.randn(1, int(EXAMPLE_SEC * SAMPLE_RATE))
    with torch.no_grad():
        processed, plen = model.preprocessor(input_signal=fake_audio, length=torch.tensor([fake_audio.shape[1]]))
        encoded, elen = model.encoder(audio_signal=processed, length=plen)
        log_probs = model.ctc_decoder(encoder_output=encoded)
    print(f"Forward pass OK! log_probs shape: {log_probs.shape}")

    patch_featurizer(model.preprocessor.featurizer)
    wrapper = FullCTCPipeline(model, patched=True).eval()

    audio = torch.zeros(1, int(SAMPLE_RATE * EXAMPLE_SEC), dtype=torch.float32)
    length = torch.tensor([audio.shape[1]], dtype=torch.int64)

    print(f"Tracing pipeline on {audio.shape} ...")
    with torch.backends.cudnn.flags(enabled=False), torch.no_grad():
        ref = wrapper(audio, length)
    print(f"  output shape: {tuple(ref.shape)}")

    print(f"Exporting to {OUT_PATH} ...")
    with torch.backends.cudnn.flags(enabled=False), torch.no_grad():
        torch.onnx.export(
            wrapper,
            (audio, length),
            str(OUT_PATH),
            input_names=["audio_signal", "length"],
            output_names=["log_probs"],
            dynamic_axes={
                "audio_signal": {0: "batch", 1: "time"},
                "length": {0: "batch"},
                "log_probs": {0: "batch", 1: "frames"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  done: {OUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

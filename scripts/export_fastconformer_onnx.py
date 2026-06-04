"""Export the FastConformer preprocessor+encoder pipeline to ONNX.

The whole pipeline (preprocessor + encoder + CTC head) is traced as a
single ONNX graph. Pre-quantization ~456 MB, uint8 quantized ~127 MB
on disk — the same size budget as the upstream yazinsai v0.1.0 release
artifact (132 MB).

The NeMo preprocessor's `torch.stft(return_complex=True)` is not
exportable as-is because ONNX doesn't have a complex STFT op. We
monkey-patch the featurizer with a real-valued STFT implementation
(magnitude only — the downstream code only uses |X| anyway) and
skip the `view_as_real + sqrt(sum)` block. The 1024-token CTC head
is small and is wrapped into the same graph, so runtime callers
get [B, T, 1025] log-probs directly.

Usage:
    .venv/bin/python scripts/export_fastconformer_onnx.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

NVIDIA_MODEL_ID = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
OUT_DIR = PROJECT_ROOT / "data" / "onnx_export"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLE_SEC = 4.0
SAMPLE_RATE = 16000


def _real_stft(self: FilterbankFeatures, x: torch.Tensor) -> torch.Tensor:
    """Real-valued STFT: returns magnitude [B, N, T].

    Drop-in for `torch.stft(..., return_complex=True)` followed by
    `view_as_real` + `sqrt(sum)`. Equivalent math, real-only ops so
    the ONNX exporter is happy. Uses `as_strided` instead of `unfold`
    (also ONNX-supported) and explicit pad values to avoid symbolic
    None constants.
    """
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

    # Sliding window via 1D index_select (ONNX-friendly; no materialised
    # expand of [B, T_frames, T_audio]). Flatten the (B, T_audio) input
    # to 1D, compute 1D source indices for each (b, t, k) output, and
    # gather. T_frames matches torch.stft's formula:
    # 1 + (T_padded - n_fft) / hop (we frame at n_fft intervals; the
    # FFT zero-pads the rest).
    B, T_audio = x.shape
    T_frames = 1 + (T_audio - n_fft) // hop
    offsets = torch.arange(n_fft, device=x.device, dtype=torch.int64)
    starts = torch.arange(0, T_frames * hop, hop, device=x.device, dtype=torch.int64)
    # 2D index per-batch, then convert to 1D
    idx_2d = starts.unsqueeze(1) + offsets.unsqueeze(0)  # [T_frames, n_fft]
    # 1D index into the flattened (B*T_audio,) tensor
    idx_1d = (idx_2d.unsqueeze(0) + torch.arange(B, device=x.device, dtype=torch.int64).unsqueeze(1).unsqueeze(2) * T_audio).reshape(-1)
    x_flat = x.reshape(-1)
    x = torch.index_select(x_flat, 0, idx_1d).reshape(B, T_frames, n_fft)
    # Pad window on BOTH sides to n_fft (matches torch.stft convention)
    if win_len < n_fft:
        pad_left = (n_fft - win_len) // 2
        pad_right = n_fft - win_len - pad_left
        window = F.pad(window, (pad_left, pad_right), "constant", 0.0)
    x = x * window

    # Manual rfft via matrix multiplication: |X|^2 = (W_re @ x)^2 + (W_im @ x)^2
    # W_re[k,n] = cos(2*pi*k*n/n_fft), W_im[k,n] = sin(2*pi*k*n/n_fft)
    # k = 0..n_fft//2 (inclusive). Precompute as buffers.
    k = torch.arange(n_fft // 2 + 1, device=x.device, dtype=torch.float32)
    n = torch.arange(n_fft, device=x.device, dtype=torch.float32)
    angles = 2 * 3.14159265 * k.unsqueeze(1) * n.unsqueeze(0) / n_fft
    W_re = torch.cos(angles)  # [n_fft//2+1, n_fft]
    W_im = torch.sin(angles)
    # X_real = x @ W_re.T : [B, T_frames, n_fft] @ [n_fft, n_fft//2+1] = [B, T_frames, n_fft//2+1]
    # But to keep the freq axis first (matches torch.stft), transpose.
    X_real = torch.matmul(x, W_re.t())
    X_imag = torch.matmul(x, W_im.t())
    # |X|^2 — but we want magnitude, then take sqrt
    mag = torch.sqrt(X_real.pow(2) + X_imag.pow(2) + 0.0)
    mag = mag.transpose(-1, -2)
    return mag  # [B, n_fft//2+1, T_frames]


def _featurizer_forward(
    self: FilterbankFeatures, x, seq_len, linear_spec: bool = False
):
    """Same as FilterbankFeatures.forward but skips the
    view_as_real + sqrt(sum) block since _real_stft already returns magnitude.
    """
    seq_len_time = seq_len
    seq_len_unfixed = self.get_seq_len(seq_len)
    seq_len = torch.where(
        seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed
    )

    if self.preemph is not None:
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(
            0
        ) < seq_len_time.unsqueeze(1)
        x = torch.cat(
            (x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1
        )
        x = x.masked_fill(~timemask, 0.0)

    with torch.amp.autocast(x.device.type, enabled=False):
        x = _real_stft(self, x)

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
    x = x.masked_fill(
        mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value
    )
    del mask
    pad_to = self.pad_to
    if pad_to == "max":
        x = nn.functional.pad(
            x, (0, self.max_length - x.size(-1)), value=self.pad_value
        )
    elif pad_to > 0:
        pad_amt = x.size(-1) % pad_to
        if pad_amt != 0:
            x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
    return x, seq_len


def patch_featurizer(feat: FilterbankFeatures) -> None:
    """Monkey-patch the featurizer to use a real-valued STFT path."""
    feat.stft = types.MethodType(_real_stft, feat)
    feat.forward = types.MethodType(_featurizer_forward, feat)


import types  # noqa: E402  (used by patch_featurizer)


class FullCTCPipeline(torch.nn.Module):
    """audio (16 kHz float32) -> [B, T, 1025] log-probs."""

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


def trace_and_export(model, out_path: Path) -> None:
    wrapper = FullCTCPipeline(model, patched=True).cuda().eval()
    audio = torch.zeros(
        1, int(SAMPLE_RATE * EXAMPLE_SEC), dtype=torch.float32, device="cuda"
    )
    length = torch.tensor([audio.shape[1]], dtype=torch.int64, device="cuda")

    print(f"Tracing pipeline on {audio.shape} ...")
    with torch.backends.cudnn.flags(enabled=False), torch.no_grad():
        ref = wrapper(audio, length)
    print(f"  output shape: {tuple(ref.shape)}")
    # Run a second time with a different length so cudnn benchmark can settle
    audio2 = torch.zeros(1, 32000, dtype=torch.float32, device="cuda")
    length2 = torch.tensor([32000], dtype=torch.int64, device="cuda")
    with torch.backends.cudnn.flags(enabled=False), torch.no_grad():
        _ = wrapper(audio2, length2)

    print(f"Exporting to {out_path} ...")
    with torch.backends.cudnn.flags(enabled=False), torch.no_grad():
        torch.onnx.export(
            wrapper,
            (audio, length),
            str(out_path),
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
    print(f"  done: {out_path.stat().st_size / 1e6:.1f} MB")


def quantize_uint8(in_path: Path, out_path: Path) -> None:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"Quantizing {in_path.name} -> {out_path.name} (int8 weights) ...")
    # reduce_range=True + symmetric weights keeps activations in a tighter
    # range that doesn't overflow for the manual-DFT + log-mel pipeline
    # (without it the per-channel scales blow up and the argmax goes blank).
    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        extra_options={"WeightSymmetric": True, "MatMulConstBOnly": True},
    )
    print(f"  done: {out_path.stat().st_size / 1e6:.1f} MB")


def verify(onnx_path: Path, model) -> None:
    import onnxruntime as ort

    print(f"Verifying {onnx_path.name} against PyTorch reference ...")

    nemo_ref = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=NVIDIA_MODEL_ID, map_location="cuda"
    )
    nemo_ref.eval()
    patch_featurizer(nemo_ref.preprocessor.featurizer)
    pt_wrapper = FullCTCPipeline(nemo_ref, patched=True).cuda().eval()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    from shared.audio import load_audio

    audio = load_audio(str(PROJECT_ROOT / "benchmark/test_corpus/001001.mp3"))
    audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
    length_t = torch.tensor([audio_t.shape[1]], dtype=torch.int64).cuda()

    with torch.backends.cudnn.flags(enabled=False), torch.no_grad():
        ref = pt_wrapper(audio_t, length_t).cpu().numpy()
    out = sess.run(
        None,
        {"audio_signal": audio_t.cpu().numpy(), "length": length_t.cpu().numpy()},
    )[0]

    T = min(ref.shape[1], out.shape[1])
    diff = np.abs(ref[0, :T] - out[0, :T])
    print(f"  shapes: ref={ref.shape}  onnx={out.shape}")
    print(f"  max abs diff: {diff.max():.5f}  mean abs diff: {diff.mean():.5f}")

    deduped = []
    prev = -1
    for i in out[0].argmax(-1):
        i = int(i)
        if i != prev and i != 1024:
            deduped.append(i)
        prev = i
    text = model.tokenizer.ids_to_text(deduped)
    print(f"  ONNX greedy text: {text}")


def main() -> None:
    print("Loading FastConformer from HuggingFace cache ...")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(
        model_name=NVIDIA_MODEL_ID,
        map_location="cuda",
    )
    model.eval()
    patch_featurizer(model.preprocessor.featurizer)

    fp32_path = OUT_DIR / "fastconformer_full.onnx"
    if not fp32_path.exists():
        trace_and_export(model, fp32_path)
    else:
        print(f"  fp32 already exists ({fp32_path.stat().st_size / 1e6:.1f} MB)")

    q8_path = OUT_DIR / "fastconformer_full_q8.onnx"
    if not q8_path.exists():
        quantize_uint8(fp32_path, q8_path)
    else:
        print(f"  q8 already exists ({q8_path.stat().st_size / 1e6:.1f} MB)")

    verify(q8_path, model)


if __name__ == "__main__":
    main()

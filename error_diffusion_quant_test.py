"""
Error-diffusion ternary quantization -- honest test.

The graph-diffusion family (hopfield_dequant_test.py, v1/v2/v3) is ruled
out: sign-pattern correlation between rows doesn't predict where the
quantization error actually is, so nudging toward neighbor-averaged values
only adds noise. All three variants lost to plain per-row ternary, even
after a 192-config sweep.

This is a structurally different idea, not a patch on that one: classical
error-diffusion / dithering (Floyd-Steinberg et al.), used for decades in
image quantization. Instead of quantizing each weight independently
(memoryless, what plain ternary does), process a row sequentially and
carry the rounding error forward into the next element's decision:

  effective = W[i,j] + carried_error
  q         = ternary(effective)
  carried_error = effective - dequant(q)     # residual pushed onto next element

This can't perfectly reconstruct any single element better than plain
rounding -- but it should preserve LOCAL SUMS along a row much more
accurately, because error that would otherwise just vanish gets pushed
onto neighbors instead of discarded. Since a row's contribution to a
matmul is (essentially) a sum over its elements times activations, better
local-sum fidelity is the real hypothesis being tested here -- not
smaller per-element error necessarily, but smaller PARTIAL-SUM error,
which is what actually matters for a forward pass.

Two metrics reported: standard elementwise MSE (comparable to the
Hopfield tests), and windowed partial-sum error (the thing this method
is actually designed to help).

Usage:
  python error_diffusion_quant_test.py
"""
import numpy as np
import gguf

MODEL_PATH = r"C:\Users\gbran\llama_demo\qwen3b.gguf"
TARGET_MIN_ROWS = 512
TRIT_THRESH = 0.7
SUM_WINDOW = 32  # window size for the partial-sum fidelity check


def load_real_weight_tensor(path):
    reader = gguf.GGUFReader(path)
    best = None
    for t in reader.tensors:
        if len(t.shape) != 2:
            continue
        rows, cols = int(t.shape[-1]), int(t.shape[-2])
        if min(rows, cols) < TARGET_MIN_ROWS:
            continue
        if "ffn_down" in t.name:
            best = t
            break
        if best is None:
            best = t
    arr = gguf.dequantize(best.data, best.tensor_type).astype(np.float32)
    arr = arr.reshape(best.shape[::-1])
    print(f"Using real tensor: {best.name}  shape={arr.shape}  "
          f"original_quant={best.tensor_type.name} (dequantized to f32 as ground-truth proxy)")
    return arr


def plain_ternary_quantize(W, thresh=TRIT_THRESH):
    scale = thresh * np.abs(W).mean(axis=1, keepdims=True)
    T = np.where(W > scale, 1.0, np.where(W < -scale, -1.0, 0.0))
    row_scale = np.abs(W).mean(axis=1, keepdims=True)
    return T * row_scale


def error_diffusion_ternary_quantize(W, thresh=TRIT_THRESH, leak=1.0, clamp_mult=None):
    """Row-sequential, error-carrying ternary quantization. Same per-row
    scale convention as the plain baseline, so the comparison is fair --
    only the DECISION PROCESS differs, not the storage format or bit
    budget (still 1 scale float + 2-bit code per element).

    leak: classical dithering fix for carry overshoot -- only a fraction
    of the residual is carried forward each step (leak=1.0 = original,
    no leak, exactly the version already tested).
    clamp_mult: if set, clip the carry to +/- clamp_mult * row_scale each
    step, an alternative/additional way to stop runaway drift."""
    rows, cols = W.shape
    scale = thresh * np.abs(W).mean(axis=1, keepdims=True)  # (rows,1)
    row_scale = np.abs(W).mean(axis=1, keepdims=True)       # (rows,1)
    out = np.zeros_like(W)
    carry = np.zeros(rows, dtype=np.float32)
    scale_flat = scale[:, 0]
    row_scale_flat = row_scale[:, 0]
    clamp_val = (clamp_mult * row_scale_flat) if clamp_mult is not None else None
    for j in range(cols):
        col = W[:, j] + carry
        q = np.where(col > scale_flat, 1.0, np.where(col < -scale_flat, -1.0, 0.0))
        dequant = q * row_scale_flat
        residual = col - dequant
        carry = leak * residual
        if clamp_val is not None:
            carry = np.clip(carry, -clamp_val, clamp_val)
        out[:, j] = dequant
    return out


def windowed_partial_sum_error(recon, W_true, window=SUM_WINDOW):
    """RMS error of windowed row-sums -- the thing error-diffusion is
    actually designed to help, as opposed to raw elementwise error."""
    rows, cols = W_true.shape
    n_windows = cols // window
    true_sums = W_true[:, :n_windows * window].reshape(rows, n_windows, window).sum(axis=2)
    recon_sums = recon[:, :n_windows * window].reshape(rows, n_windows, window).sum(axis=2)
    return float(np.sqrt(np.mean((true_sums - recon_sums) ** 2)))


def evaluate(name, recon, W_true):
    mse = float(np.mean((recon - W_true) ** 2))
    mae = float(np.mean(np.abs(recon - W_true)))
    cos = float(np.sum(recon * W_true) / (np.linalg.norm(recon) * np.linalg.norm(W_true) + 1e-12))
    psum_rmse = windowed_partial_sum_error(recon, W_true)
    print(f"  {name:<28} MSE={mse:.6e}  MAE={mae:.6e}  cos_sim={cos:.6f}  "
          f"windowed_partial_sum_RMSE={psum_rmse:.6e}")
    return mse, psum_rmse


if __name__ == "__main__":
    print("=" * 70)
    print("  Error-diffusion ternary quantization -- honest test")
    print("=" * 70)

    W = load_real_weight_tensor(MODEL_PATH)
    print(f"  True weight stats: mean|W|={np.abs(W).mean():.5f} std={W.std():.5f}\n")

    baseline_recon = plain_ternary_quantize(W)

    print("Reconstruction quality:")
    base_mse, base_psum = evaluate("baseline (plain per-row ternary)", baseline_recon, W)
    ed_mse, ed_psum = evaluate("error-diffusion, leak=1.0 (original, no leak)",
                                error_diffusion_ternary_quantize(W, leak=1.0), W)

    print("\nLeak/clamp sweep (all still testing the same real tensor):")
    configs = [
        {"leak": 0.9, "clamp_mult": None},
        {"leak": 0.7, "clamp_mult": None},
        {"leak": 0.5, "clamp_mult": None},
        {"leak": 0.3, "clamp_mult": None},
        {"leak": 1.0, "clamp_mult": 1.0},
        {"leak": 1.0, "clamp_mult": 2.0},
        {"leak": 0.7, "clamp_mult": 1.5},
        {"leak": 0.5, "clamp_mult": 1.0},
    ]
    results = []
    for cfg in configs:
        recon = error_diffusion_ternary_quantize(W, **cfg)
        mse, psum = evaluate(f"leak={cfg['leak']} clamp={cfg['clamp_mult']}", recon, W)
        results.append((cfg, mse, psum))

    print("\nSummary vs baseline:")
    for cfg, mse, psum in results:
        mse_delta = (mse - base_mse) / base_mse * 100
        psum_delta = (psum - base_psum) / base_psum * 100
        both_better = mse_delta < 0 and psum_delta < 0
        flag = "  <-- BOTH better than baseline" if both_better else ""
        print(f"  leak={cfg['leak']:<4} clamp={str(cfg['clamp_mult']):<5}  "
              f"MSE {'WORSE' if mse_delta>0 else 'better'} {abs(mse_delta):5.1f}%   "
              f"psum {'WORSE' if psum_delta>0 else 'better'} {abs(psum_delta):5.1f}%{flag}")

    print("\n" + "=" * 70)
    print("  Honest result above -- not engineered for either side to win.")
    print("=" * 70)

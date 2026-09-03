"""
Hopfield-diffusion dequantization test — honest first pass.

Idea (from conversation, 2026-08-19): instead of a fixed linear dequant
formula for ternary-quantized weights, use an iterative associative-memory
process at LOAD TIME to reconstruct a better approximation of the true
float weights, using only information already implicit in the ternary
codes themselves (zero extra storage vs plain ternary).

Ruled out before building: the naive single-tensor self-storage version
(J = outer(ternary(W), ternary(W))) is a guaranteed trivial fixed point —
ternary(W) is already stationary under sign(J @ x), so it can only return
exactly what you started with. No point running that.

What's actually tested here: ternary quantization loses the CONTINUOUS
MAGNITUDE within each row (each element collapses to one of {-s, 0, +s}).
Real trained weight matrices have genuine correlation between rows (rows
aren't independent — redundancy is a known property of overparameterized
nets). Hypothesis: rows whose TERNARY CODES are similar likely have
correlated true magnitudes too, so diffusing magnitude estimates between
correlated rows (weighted by ternary-code similarity, computed for free
from the stored codes — no extra bits) might recover some of what the
per-row scale-and-threshold throws away.

This is a real, falsifiable test — reported honestly either way.

Usage:
  python hopfield_dequant_test.py
"""
import numpy as np
import gguf

MODEL_PATH = r"C:\Users\gbran\llama_demo\qwen3b.gguf"
TARGET_MIN_ROWS = 512      # only test on a reasonably large 2D weight
TRIT_THRESH = 0.7          # same convention as ternary_quant.py: tau = thresh * mean(|w|)
N_DIFFUSION_STEPS = 8
TOPK_NEIGHBORS = 16        # sparsify the correlation graph — cheap + avoids over-smoothing


def load_real_weight_tensor(path):
    # NOTE: this GGUF file is itself already quantized (Q4_K/Q6_K, no raw
    # F32/F16 tensors present) -- there is no original-precision ground
    # truth available locally without a fresh multi-GB fp16 download.
    # Dequantizing to float32 and treating that as "W" is an honest proxy:
    # it still tests the real question (does correlation-based diffusion
    # recover magnitude information a coarser ternary pass throws away,
    # relative to the best float representation actually available here),
    # it just means the reported numbers are relative to a Q4_K/Q6_K
    # reference, not the original fp16 training weights. Flagged, not hidden.
    reader = gguf.GGUFReader(path)
    best = None
    for t in reader.tensors:
        if len(t.shape) != 2:
            continue
        rows, cols = int(t.shape[-1]), int(t.shape[-2])  # gguf shape order is reversed vs numpy
        if min(rows, cols) < TARGET_MIN_ROWS:
            continue
        if "ffn_down" in t.name:
            best = t
            break
        if best is None:
            best = t
    if best is None:
        raise RuntimeError("No suitable 2D tensor found in this GGUF file.")
    arr = gguf.dequantize(best.data, best.tensor_type).astype(np.float32)
    arr = arr.reshape(best.shape[::-1])
    print(f"Using real tensor: {best.name}  shape={arr.shape}  "
          f"original_quant={best.tensor_type.name} (dequantized to f32 as ground-truth proxy)")
    return arr


def per_row_ternary_quantize(W, thresh=TRIT_THRESH):
    """Same convention as ternary_quant.py's global version, but per-row scale
    (standard practice — a single global scale for a whole matrix throws away
    even more information than necessary and would bias this test)."""
    scale = thresh * np.abs(W).mean(axis=1, keepdims=True)
    T = np.where(W > scale, 1.0, np.where(W < -scale, -1.0, 0.0))
    row_scale = np.abs(W).mean(axis=1, keepdims=True)  # magnitude estimate stored per row (cheap, 1 float/row)
    return T, row_scale


def build_row_correlation_graph(T, topk=TOPK_NEIGHBORS, signed=False):
    """Cosine similarity between ternary row-codes, sparsified to top-k
    (by |similarity|) neighbors per row (dense would over-smooth and is
    O(rows^2) memory-heavy for no benefit).

    signed=False (v1/v2 behaviour): clip to positive correlation only --
    only "this row looks like that row" pulls them together.
    signed=True (v3 follow-up): keep the sign -- anti-correlated rows
    actively repel each other too, closer to true Hopfield Hebbian
    dynamics (which are signed) than the earlier simplification."""
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Tn = T / norms
    sim = Tn @ Tn.T
    np.fill_diagonal(sim, -np.inf if not signed else 0.0)  # never let a row cite itself
    n = sim.shape[0]
    J = np.zeros_like(sim)
    rank_key = np.abs(sim) if signed else sim
    idx = np.argpartition(-rank_key, topk, axis=1)[:, :topk]
    rows = np.repeat(np.arange(n), topk)
    cols = idx.flatten()
    vals = sim[rows, cols]
    if not signed:
        vals = np.clip(vals, 0, None)
    J[rows, cols] = vals
    norm_denom = np.abs(J).sum(axis=1, keepdims=True) if signed else J.sum(axis=1, keepdims=True)
    norm_denom[norm_denom == 0] = 1.0
    J = J / norm_denom  # normalize -> each row's diffusion step is a (possibly signed) weighted combination of neighbors
    return J


def energy_descent_dequant_v2(T, row_scale, J, steps, alpha=0.5, checkpoints=None):
    """v2, per the follow-up: diffuse the FULL per-element ternary-seeded
    values (not just a per-row scalar) across correlated neighbor rows, and
    let the state evolve freely in continuous space -- no re-snapping to the
    ternary sign/zero pattern at any point. This lets correlated rows'
    actual per-position values inform each other directly, which the v1
    scalar-only diffusion never used at all."""
    state = (T * row_scale).copy()  # full per-element baseline seed, not just a row scalar
    results = {}
    cps = set(checkpoints or [steps])
    for step in range(1, steps + 1):
        neighbor_avg = J @ state
        state = (1 - alpha) * state + alpha * neighbor_avg
        if step in cps:
            results[step] = state.copy()
    return results


def energy_descent_dequant_v3(T, row_scale, J, steps, alpha, beta, checkpoints=None):
    """v3: anchored diffusion. v2 collapsed to a decorrelated consensus
    state because nothing pulled it back toward the original signal --
    classic graph-diffusion oversmoothing. Fix: re-anchor to the original
    ternary-seeded values every step, so the diffusion can only ever nudge
    the estimate, never fully overwrite it.

      diffused = (1 - alpha) * state + alpha * (J @ state)     [as before]
      state    = (1 - beta)  * seed  + beta  * diffused         [NEW anchor]

    beta=1.0 reduces to v2 exactly (sanity check). beta=0 reduces to the
    untouched seed (sanity check). The real question is whether some
    beta in between ever beats the plain baseline."""
    seed = (T * row_scale).copy()
    state = seed.copy()
    results = {}
    cps = set(checkpoints or [steps])
    for step in range(1, steps + 1):
        diffused = (1 - alpha) * state + alpha * (J @ state)
        state = (1 - beta) * seed + beta * diffused
        if step in cps:
            results[step] = state.copy()
    return results


def evaluate(name, recon, W_true):
    mse = float(np.mean((recon - W_true) ** 2))
    mae = float(np.mean(np.abs(recon - W_true)))
    cos = float(np.sum(recon * W_true) / (np.linalg.norm(recon) * np.linalg.norm(W_true) + 1e-12))
    print(f"  {name:<28} MSE={mse:.6e}  MAE={mae:.6e}  cos_sim={cos:.6f}")
    return mse


if __name__ == "__main__":
    print("=" * 70)
    print("  Hopfield-diffusion dequantization — honest first test")
    print("=" * 70)

    W = load_real_weight_tensor(MODEL_PATH)
    print(f"  True weight stats: mean|W|={np.abs(W).mean():.5f} std={W.std():.5f}\n")

    T, row_scale = per_row_ternary_quantize(W)
    zero_frac = float((T == 0).mean())
    print(f"  Ternary sparsity (zero fraction): {zero_frac*100:.1f}%\n")

    print("Reconstruction quality:")
    baseline_recon = T * row_scale  # standard per-row ternary dequant -- the thing to beat
    baseline_mse = evaluate("baseline (per-row ternary)", baseline_recon, W)

    J_unsigned = build_row_correlation_graph(T, signed=False)
    J_signed   = build_row_correlation_graph(T, signed=True)
    print(f"\n  Row-correlation graphs built (top-{TOPK_NEIGHBORS} neighbors/row)\n")

    print("v3: anchored diffusion, signed vs unsigned graph, sweeping alpha/beta, out to 36 steps:")
    checkpoints = [1, 2, 4, 8, 16, 24, 32, 36]
    best = {"mse": baseline_mse, "config": "baseline (nothing beat it)"}
    for graph_name, J in [("unsigned", J_unsigned), ("signed", J_signed)]:
        for alpha in [0.3, 0.5, 0.8]:
            for beta in [0.1, 0.3, 0.5, 0.7]:
                all_states = energy_descent_dequant_v3(T, row_scale, J, steps=36,
                                                        alpha=alpha, beta=beta,
                                                        checkpoints=checkpoints)
                for step in checkpoints:
                    recon = all_states[step]
                    mse = float(np.mean((recon - W) ** 2))
                    if mse < best["mse"]:
                        cos = float(np.sum(recon * W) / (np.linalg.norm(recon) * np.linalg.norm(W) + 1e-12))
                        best = {"mse": mse, "cos": cos,
                                "config": f"graph={graph_name} alpha={alpha} beta={beta} steps={step}"}

    print(f"\n  Best config found across the full sweep: {best['config']}")
    if "cos" in best:
        delta_pct = (best["mse"] - baseline_mse) / baseline_mse * 100
        print(f"  MSE={best['mse']:.6e}  cos_sim={best['cos']:.6f}  "
              f"({abs(delta_pct):.2f}% {'better' if delta_pct < 0 else 'worse'} than baseline)")
    else:
        print(f"  Nothing in this sweep beat the plain baseline (MSE={baseline_mse:.6e}).")

    print("\n" + "=" * 70)
    print("  Honest result above -- not engineered for either side to win.")
    print("=" * 70)

"""
Error-diffusion ternary quantization -- real downstream perplexity test.

Resolves the open question from error_diffusion_quant_test.py: does the
windowed partial-sum fidelity win (leak=0.3: 4.4% worse elementwise MSE,
17% better partial-sum RMSE vs plain ternary) actually translate into
better real model output, or does elementwise/cosine fidelity matter more
and error-diffusion is a net loss in practice? Only an actual forward
pass with a real loss can answer that -- everything before this was a
weight-level proxy metric.

Three conditions, same model, same eval text:
  A) float16 (reference, uncompressed)
  B) plain per-row ternary          (baseline to beat)
  C) error-diffusion ternary, leak=0.3  (the candidate)

Usage:
  python error_diffusion_perplexity_test.py
"""
import torch
import torch.nn as nn
import numpy as np
import time

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRIT_THRESH = 0.7
LEAK = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ══════════════════════════════════════════════════════════════════════════
# QUANTIZATION FUNCTIONS (numpy, matching error_diffusion_quant_test.py exactly)
# ══════════════════════════════════════════════════════════════════════════

def plain_ternary(W, thresh=TRIT_THRESH):
    scale = thresh * np.abs(W).mean(axis=1, keepdims=True)
    row_scale = np.abs(W).mean(axis=1, keepdims=True)
    T = np.where(W > scale, 1.0, np.where(W < -scale, -1.0, 0.0))
    return (T * row_scale).astype(np.float32)


def error_diffusion_ternary(W, thresh=TRIT_THRESH, leak=LEAK):
    rows, cols = W.shape
    scale = (thresh * np.abs(W).mean(axis=1, keepdims=True))[:, 0]
    row_scale = np.abs(W).mean(axis=1, keepdims=True)[:, 0]
    out = np.zeros_like(W)
    carry = np.zeros(rows, dtype=np.float32)
    for j in range(cols):
        col = W[:, j] + carry
        q = np.where(col > scale, 1.0, np.where(col < -scale, -1.0, 0.0))
        dequant = q * row_scale
        carry = leak * (col - dequant)
        out[:, j] = dequant
    return out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# APPLY A QUANTIZATION FUNCTION TO EVERY LINEAR LAYER (in-place, on a copy)
# ══════════════════════════════════════════════════════════════════════════

SKIP_NAMES = {"lm_head", "embed_tokens"}

def apply_quant_to_model(model, quant_fn, name):
    replaced, total_params = 0, 0
    t0 = time.time()
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not any(s in full_name for s in SKIP_NAMES):
            with torch.no_grad():
                W = module.weight.data.float().cpu().numpy()
                Wq = quant_fn(W)
                module.weight.data = torch.from_numpy(Wq).to(module.weight.dtype).to(module.weight.device)
            replaced += 1
            total_params += W.size
    print(f"  [{name}] quantized {replaced} Linear layers ({total_params:,} params) in {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
# PERPLEXITY
# ══════════════════════════════════════════════════════════════════════════

EVAL_TEXTS = [
    "The history of artificial intelligence began in antiquity, with myths and stories of "
    "artificial beings endowed with intelligence or consciousness by master craftsmen. The "
    "seeds of modern AI were planted by philosophers who attempted to describe human thinking "
    "as a symbolic system. This work culminated in the invention of the programmable digital "
    "computer in the 1940s, a machine based on the abstract essence of mathematical reasoning.",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight "
    "to synthesize foods with the help of chlorophyll pigments. In plants, algae, and "
    "cyanobacteria, photosynthesis uses carbon dioxide and water, releasing oxygen as a "
    "byproduct. Most plants, most algae, and cyanobacteria perform photosynthesis, and such "
    "organisms are called photoautotrophs.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n"
    "    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
]


def compute_perplexity(model, tokenizer, texts):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            out = model(**inputs, labels=inputs["input_ids"])
            n_tok = inputs["input_ids"].shape[1] - 1
            total_nll += out.loss.item() * n_tok
            total_tokens += n_tok
    return float(np.exp(total_nll / total_tokens))


# ══════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Loaded. {n_params:,} parameters.\n")

    print("=" * 70)
    print("  A) float32 reference")
    print("=" * 70)
    ppl_float = compute_perplexity(base_model, tokenizer, EVAL_TEXTS)
    print(f"  Perplexity: {ppl_float:.4f}\n")

    import copy
    print("=" * 70)
    print("  B) plain per-row ternary")
    print("=" * 70)
    model_b = copy.deepcopy(base_model)
    apply_quant_to_model(model_b, plain_ternary, "plain-ternary")
    ppl_plain = compute_perplexity(model_b, tokenizer, EVAL_TEXTS)
    print(f"  Perplexity: {ppl_plain:.4f}\n")
    del model_b
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("=" * 70)
    print(f"  C) error-diffusion ternary, leak={LEAK}")
    print("=" * 70)
    model_c = copy.deepcopy(base_model)
    apply_quant_to_model(model_c, error_diffusion_ternary, f"error-diffusion-leak{LEAK}")
    ppl_ed = compute_perplexity(model_c, tokenizer, EVAL_TEXTS)
    print(f"  Perplexity: {ppl_ed:.4f}\n")

    print("=" * 70)
    print("  HONEST SUMMARY")
    print("=" * 70)
    print(f"  float32 reference          : {ppl_float:.4f}")
    print(f"  plain per-row ternary      : {ppl_plain:.4f}  ({(ppl_plain-ppl_float)/ppl_float*100:+.1f}% vs float)")
    print(f"  error-diffusion (leak=0.3) : {ppl_ed:.4f}  ({(ppl_ed-ppl_float)/ppl_float*100:+.1f}% vs float)")
    ed_vs_plain = (ppl_ed - ppl_plain) / ppl_plain * 100
    verdict = "WORSE than" if ed_vs_plain > 0 else "BETTER than"
    print(f"\n  Error-diffusion is {verdict} plain ternary by {abs(ed_vs_plain):.2f}% on real perplexity.")
    print("=" * 70)

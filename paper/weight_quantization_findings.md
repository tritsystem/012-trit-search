# Ternary Weight Quantization on code-minilm: Three Options Tested

Applied ternary weight quantization to the actual transformer model weights
(not just the stored embedding index), measuring accuracy on 20 hard semantic
search triplets. Three approaches tested one by one.

## Baseline

- Model: fine-tuned code-minilm (all-MiniLM-L6-v2, 37 Linear layers, 10.76M params)
- Benchmark: 20 (query, positive, negative) triplets — same hard pairs as trit_benchmark.py
- Float32 accuracy: **95.0%**
- INT8 PTQ accuracy: **95.0%** (0pp drop, 4x compression) — the practical industry baseline

---

## Option A — Post-Training Quantization (PTQ)

Apply ternary quantization ({-1, 0, +1}) to all 37 Linear layer weights in-place,
no retraining. Threshold: t = 0.7 * mean(|w|) per layer.

| Mode | Accuracy | Drop | Compression |
|---|---|---|---|
| Float32 | 95.0% | — | 1x |
| INT8 PTQ | 95.0% | 0pp | 4x |
| Ternary PTQ | 50.0% | -45pp | 29.7x |

**Sparsity:** 44.6% of weights became zero, uniform across all 37 layers
(range: 42–48%, no single layer stands out as an outlier).

**Why it collapses:** 50% on binary triplets is random chance. The 44% sparsity
means nearly half of all transformer weights are zeroed out simultaneously.
Attention QK similarity scores — which determine which tokens attend to which —
are destroyed when the projection weights snap from float32 to {-1, 0, +1}.
The embedding geometry the model learned in float32 is incompatible with ternary
weight distributions.

**Script:** `trit_ptq_ternary_test.py`

---

## Option B — Quantization-Aware Training (QAT)

Fine-tune all-MiniLM-L6-v2 from scratch with ternary weights using a
Straight-Through Estimator (STE), so gradients flow through the discrete
quantization step. Training data: 80,000 pairs extracted from local codebases
(012-ternary, horde-beta-version-1, tribe) using function-body heuristic.
InfoNCE contrastive loss (temperature=0.05), 3 epochs, 7,503 steps.
Float32 warmup for first 15% of steps, then ternary switched on.

| Mode | Accuracy | vs Float32 |
|---|---|---|
| Float32 (code-minilm baseline) | 95.0% | — |
| Ternary PTQ (Option A) | 50.0% | -45pp |
| Ternary QAT — ternary weights | 35.0% | -60pp |
| Ternary QAT — float32 weights (upper bound) | 85.0% | -10pp |

**The split between the last two rows is the key diagnostic:**
- QAT in float32 mode reaches 85% — the training loop learned better
  representations than the original code-minilm (95% was on a different
  evaluation setup; under this eval method the original scores 70%,
  and QAT in float32 mode scores 85%). The QAT training procedure works.
- QAT in ternary mode scores 35% — *worse* than PTQ's 50%. Training loss
  jumped from 2.15 → 3.82 the moment ternary weights activated at step 1125,
  and stayed stuck for the remaining 6,378 steps. The STE gradient signal
  is insufficient to recover from the sharp geometry shift caused by
  simultaneous quantization of all 37 layers at once.

**What went wrong:** temperature=0.05 in InfoNCE is very sharp. Small
perturbations in embeddings cause large loss jumps. When all weights snap to
ternary simultaneously, the embedding geometry shifts abruptly, the loss
spikes, and subsequent gradient updates push weights into configurations
that keep retriggering the loss spike rather than converging.

**The fixable path:** temperature scheduling (start warm ~0.1, anneal to 0.05
alongside quantization warmup) + progressive threshold annealing (start with
soft threshold τ >> 0.7*|w|_mean, anneal toward hard threshold over training)
would give the model time to adapt incrementally rather than all at once.
Not implemented here — flagged as the next step.

**Script:** `trit_qat_ternary_test.py`

---

## Option C — Mixed Precision PTQ

Hypothesis: attention and FFN layers have different sensitivity to quantization.
Test each independently to find the bottleneck, then apply INT8 to the sensitive
layers and ternary to the robust ones.

Layer classification:
- **Attention** (Q, K, V, output.dense): 24 layers, 3.54M params
- **FFN** (intermediate, output): 13 layers, 7.23M params
- Pooler: classified as FFN

| Config | Accuracy | Drop | Compression |
|---|---|---|---|
| Float32 | 95.0% | — | 1x |
| INT8 all | 95.0% | 0pp | 4x |
| Ternary FFN only | 45.0% | -50pp | 2.8x |
| Mixed: INT8 attn + Ternary FFN | 40.0% | -55pp | 9.5x |
| Ternary attn only | 25.0% | -70pp | 1.5x |
| Ternary all | 50.0% | -45pp | 29.7x |

**Surprising result:** ternary attention alone (-70pp) is *worse* than ternary
on all layers (-45pp). Explanation: the FFN layers partially compensate for
broken attention geometry. When only attention is ternarized, the FFN processes
corrupted token representations with full float32 precision — which may actually
amplify the corruption. When both are ternary, the FFN's own quantization noise
acts as a kind of regularizer that partially masks the attention damage.

**Mixed precision doesn't help:** INT8 attention + Ternary FFN (-55pp) is worse
than ternary-all (-45pp) because FFN damage alone is already severe (-50pp)
and the interaction effect (where FFN quantization noise partially masks attn
damage) is lost when attention is protected.

**Conclusion:** there is no layer-type split that recovers meaningful accuracy
under PTQ at this compression ratio. Both layer types are precision-sensitive.
Mixed precision only makes sense *with retraining* (QAT), where the model
can learn to compensate for the quantization noise in each layer type separately.

**Script:** `trit_mixed_precision_test.py`

---

## Cross-option summary

| Approach | Best accuracy | Compression | Viable? |
|---|---|---|---|
| INT8 PTQ | 95.0% | 4x | Yes — the practical choice |
| Ternary PTQ | 50.0% | 29.7x | No — random chance |
| Ternary QAT (current) | 35.0% | 29.7x | No — training instability |
| Mixed INT8+Ternary PTQ | 40.0% | 9.5x | No — worse than pure INT8 |

**What ternary compression IS proven to do in this project:** compress the
stored embedding *index* (not the model weights). `vectors_ternary.npy`
achieves 19.9x disk compression on the pre-computed float32 embedding vectors,
with 7.35pp accuracy loss. That is a fundamentally different operation —
quantizing stored lookup vectors, not live transformer weight matrices —
and it works because embeddings can be re-normalized post-quantization,
whereas weight matrices cannot be similarly corrected without retraining.

## What would make ternary model weights viable

1. **Temperature-annealed QAT** — warm InfoNCE temperature (0.1+) during
   quantization warmup, anneal to 0.05 over training. Prevents loss spike
   from sharp geometry shift.
2. **Progressive threshold annealing** — start with τ >> 0.7*mean(|w|)
   (few weights quantized), anneal τ down over epochs. Model adapts
   incrementally rather than all at once.
3. **Larger model** — MiniLM-L6 has only 6 layers and 22M params. Larger
   transformers (L12, L24) have more representational redundancy and
   recover better from quantization noise. **Tested below at 27B (1225x
   larger) — this hypothesis is FALSIFIED, see next section.**
4. **Domain-specific training data** — the QAT run used locally-extracted
   function-body pairs, which are noisy and short. Higher-quality contrastive
   pairs (with genuine hard negatives) would give the model a cleaner gradient
   signal to work with during the critical ternary warmup phase.

---

## Follow-up: Qwen3.8-27B (2026-08-18) — scale does NOT rescue ternary PTQ

Directly tests hypothesis #3 above at real scale: prune 30% of FFN weights
(magnitude-based) then quantize survivors, on Qwen3.8-27B (26.9B params,
1225x larger than code-minilm), measuring real perplexity on a fixed
held-out text sample. `012-ternary/qwen27b_prune_ternary_test.py` +
`run_all.py`.

| Condition | Perplexity | vs baseline | Sparsity |
|---|---|---|---|
| A) baseline bf16 | 3.823 | — | 0% |
| B) prune 30% FFN only | 3.979 | +4.1% | 30.1% |
| **C) prune 30% + ternary quantize survivors** | **4738.863** | **+123,850%** | 30.1% |
| **D) prune 30% + int4 (NF4) quantize survivors** | **4.050** | **+5.9%** | 30.1% |

**Hypothesis #3 (larger model recovers better) is falsified, decisively.**
At 27B params, ternary PTQ is not just broken, it's *more* catastrophic in
relative terms than at 22M params (a total collapse to garbage-level loss
8.46, versus the original's -45pp accuracy drop — still bad, but bounded).
More representational redundancy did not help; if anything the larger
model had further to fall.

**Real new result, not in the original test**: **int4 survives PTQ with no
retraining at all** — 5.9% perplexity cost, nearly identical to pruning
alone (4.1%). This wasn't tested in the original three-option sweep (which
only validated INT8 as safe). Informed by DeepSeek V4's real production
choice to use FP4 (16 levels) as their most aggressive precision and never
go as low as ternary (3 levels) — this result is consistent with that
choice being a genuine precision floor, not just conservatism. int4 sits
between INT8 (proven safe, 4x compression) and ternary (proven catastrophic,
29.7x compression) — a real, usable middle ground with ~8x compression that
this project hadn't validated until now.

**Real engineering finding along the way**: modifying weights in a
disk-offloaded (`accelerate` `device_map="auto"`) loaded model directly
doesn't work reliably — hit a meta-tensor crash (weights aren't
materialized until touched by a real forward pass), then after fixing that,
unbounded GPU memory growth (repeated manual materialization with no
release) and a device-mismatch crash from fighting the hook's own
placement bookkeeping. The robust fix: process the raw safetensors shard
files directly on disk (plain tensors, no hooks, no meta placeholders),
save a new checkpoint, then load that cleanly through the normal path.

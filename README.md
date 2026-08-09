# 012 Ternary

A research and tooling project exploring ternary computing ({-1, 0, +1}) — from a hardware RTL design, through a software CPU emulator, to a working, shipping product: **OBSERVE**, a local semantic code search desktop app + MCP server. **→ [OBSERVE.md](OBSERVE.md) for the product: install, MCP setup, and the measured benchmarks (including where grep beats it).**

This repo contains real, measured, reproducible results — both positive and negative — not just claims. See [paper/resonance_domain.md](paper/resonance_domain.md) for an example of a fully falsifiable test series (5 honest experiments, 1 confirmed win, 4 disconfirmed hypotheses, with a precise explanation for the boundary). See [paper/cross_substrate_symmetry_findings.md](paper/cross_substrate_symmetry_findings.md) for a synthesis of this project's own symmetry/resonance results against an independent acoustic-MEMS-plate study — four substrates, four different answers to "does breaking symmetry help computation?" — now closed with a prospective confirmation: giving the Spikeling resonator bank the coupling richness its negative result said it lacked, and passing a pre-registered capability gate first, replicates the acoustic even/odd dichotomy at full strength (even-order gap +0.254, odd −0.003), making the selection rule a two-substrate result and the capability-bar precondition a *predictive* rule, not a post-hoc excuse. See [paper/npc_consensus_findings.md](paper/npc_consensus_findings.md) for the consensus-gate primitive applied, independently of any neural net, to a real game's (Tribe's) fight-or-flee NPC logic — a clean, statistically robust win (+1.8pp accuracy, 30/30 seeds) — and [paper/order_acceptance_findings.md](paper/order_acceptance_findings.md) for the same primitive tested on a *different* decision shape in the same game, where it loses just as decisively (-4.6pp, 30/30 seeds). The resulting scoping rule (weighted combination wins under calibrated evidence, voting wins under uncalibrated/contaminated evidence) was then checked against three independent fields: [Spikeling-Project/research/POPULATION_CODING_FINDINGS.md](https://github.com/tritsystem/Spikeling/blob/main/research/POPULATION_CODING_FINDINGS.md) (population-coding theory + robust statistics — confirmed, dramatically), [paper/tmr_findings.md](paper/tmr_findings.md) (classical Triple Modular Redundancy — confirmed, softly), and [paper/ensemble_ml_findings.md](paper/ensemble_ml_findings.md) (real scikit-learn ensembles under covariate shift — weak/inconclusive, which sharpened the rule: it's *rank-reordering* of relative reliability that matters, not miscalibration in general). That sharpened rule then survived its designed-to-kill test in [paper/rank_reorder_findings.md](paper/rank_reorder_findings.md): a shift engineered (on validation data only) to flip the previously-best classifier to worst reverses the weighted decoder's advantage (majority vote wins, t=-2.70, 30/30 seeds put the flipped classifier in the bottom two), while a damage-matched, rank-preserving shift from the *same* candidate family leaves weighted voting winning — same absolute damage, opposite outcome (paired contrast t=+2.55; pooled dose-response between rank preservation and the weighted advantage rho=+0.44, p=3.9e-07) — completing the five-check scoping-rule ladder with its final form: weighted combination beats voting exactly as long as the calibration-time reliability ranking still holds at decision time. The whole six-study program is assembled as a standalone preprint in [paper/scoping_rule_paper.tex](paper/scoping_rule_paper.tex) (upload fields in [paper/SCOPING_RULE_ZENODO.txt](paper/SCOPING_RULE_ZENODO.txt)). See [paper/weight_quantization_findings.md](paper/weight_quantization_findings.md) for a three-option study applying ternary quantization to the actual transformer model weights (PTQ, QAT, mixed precision) — all three options fail to recover accuracy without retraining, with a surprising finding that ternary attention-only is *worse* than ternary-all because FFN quantization noise partially compensates for attention damage. See [paper/token_reduction_findings.md](paper/token_reduction_findings.md) for a measured (not pitched) token-reduction result from OBSERVE's MCP server — 66.3% fewer tokens than the existing search tool, via dedup/relevance-cutoff/formatting, not the compression or consensus-gate mechanisms an earlier draft of this idea incorrectly credited — and [paper/quality_benchmark_findings.md](paper/quality_benchmark_findings.md) for the honest other half of that tradeoff: a ground-truth recall test showing the token savings are not free, with a real reproduced case where the compression's dedup/cutoff mechanism dropped a correct answer (86% vs 71% combined recall) — traced to a root cause (blind character-window chunking) and fixed in testing (function-boundary chunking recovers chunk-level recall from 29% to 86% at zero query-time cost). See [paper/grep_vs_semantic_findings.md](paper/grep_vs_semantic_findings.md) for the comparison against Grep — the tool an assistant actually reaches for by default — split into exact-identifier-known vs concept-only queries, then corrected with a stricter follow-up test: Grep wins outright when the identifier is known (5/5 vs 3/5 recall), and semantic search's real advantage narrows to genuine vocabulary mismatches (legacy renames, domain jargon, orphaned references) rather than any natural-language query in general, once obvious-synonym queries (which good naming conventions make grep-guessable anyway) are excluded from the test.

**Full documentation:** [DOCS.md](DOCS.md) (architecture + key results) and [FILES.md](FILES.md) (every file, what it does, how to run it).

---

## What's actually proven here

| Claim | Status | Where |
|---|---|---|
| Fine-tuned MiniLM beats baseline on code search | ✅ Measured: 96% vs 92% (hard benchmark), 92% vs 76.7% (real OSS code) | `trit_benchmark.py`, `trit_oss_test.py` |
| Ternary weight compression (20x) works, but costs accuracy | ✅ Measured: 7.35pp lost, vs INT8's 0.19pp gain at 4x compression | `precision_loss_test.py`, `int8_vs_ternary_test.py` |
| Triadic structure (not just ternary weights) causes CIFAR-10 rotation robustness | ✅ Measured ablation in `experiments.py` | `experiments.py` |
| "Resonating cell" hybrid improves decision-making generally | ❌ Disproven — wins only when the signal is genuinely time-varying (1 of 5 tests) | `paper/resonance_domain.md` |
| Triadic architecture beats pretrained models on text embedding from scratch | ❌ Disproven — scored 60% vs MiniLM's 96% | `trit_triadic_encoder.py` |
| Ternary CPU's native CONSENSUS instruction is more efficient than software fallback | ✅ Measured: 1.4x fewer instructions (modest, not the "10x" sometimes claimed informally) | `trit_emulator_tests.py` |

---

## Quick start

```bash
pip install -r requirements.txt          # full project
# or: pip install -r requirements_app.txt   # just the OBSERVE search app

python trit_app.py                       # launch OBSERVE desktop search
python experiments.py                    # CIFAR-10 ternary/triadic ablation study
python trit_emulator.py                  # balanced-ternary CPU emulator demo
```

GPU users: install `torch`/`torchvision` matching your CUDA version first (see comment in `requirements.txt`).

---

## Structure

```
012-ternary/
├── trit_app.py              OBSERVE — desktop semantic code search (the shipping product)
├── trit_search.py           Core search engine (CLI/HTTP API)
├── trit_embed_train.py      Fine-tunes MiniLM on code for search
├── experiments.py           Core ternary/triadic learning system (CIFAR-10 ablations)
├── trit_emulator.py         Balanced-ternary CPU emulator
├── trit_assembler.py        Text assembly syntax for the emulator
├── trit_*.py                LLM fine-tuning, language model, memory store, etc.
├── trit_resonant_*.py       The 5-test resonance hypothesis series (see paper/)
├── hardware/                SystemVerilog RTL — consensus gate, ternary ALU, FPGA target
├── paper/                   Research write-ups (resonance_domain.md, 012_paper.md)
├── DOCS.md                  Full architecture + results documentation
└── FILES.md                 Every file, what it does, how to run it
```

## Hardware validation

```bash
iverilog -g2012 -o sim hardware/trit_pkg.sv hardware/trit_register.sv \
    hardware/trit_not.sv hardware/trit_add.sv \
    hardware/consensus_gate.sv hardware/testbench.sv
vvp sim
# 36/36 testbench cases pass in simulation
```

FPGA synthesis (`hardware/vivado_synth.tcl`) targets Xilinx Ultrascale+; this has not been run on real silicon — see DOCS.md for what's simulated vs hardware-verified.

## License

MIT — see [LICENSE](LICENSE).

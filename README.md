# 012-ternary / OBSERVE

![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
[![test](https://github.com/tritsystem/012-trit-search/actions/workflows/test.yml/badge.svg)](https://github.com/tritsystem/012-trit-search/actions/workflows/test.yml)

**OBSERVE** is local semantic code search — a desktop app, a CLI, and an MCP
server for coding agents. Point it at a codebase; it indexes on your machine and
nothing leaves it. It came out of a research project on ternary computing
({-1, 0, +1}), and that research lives in the same repo.

- **Product:** [OBSERVE.md](OBSERVE.md) — install, MCP setup, and the measured
  benchmarks (including where grep beats it).
- **Research trail:** [RESEARCH.md](RESEARCH.md) — every measured finding behind
  it, positive and negative, each tied to a script.
- **Deep docs:** [DOCS.md](DOCS.md) (architecture + results) · [FILES.md](FILES.md)
  (every file) · [CHANGELOG.md](CHANGELOG.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

This repo contains real, measured, reproducible results — both positive and
negative — not just claims.

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

## Install OBSERVE (one line)

**macOS / Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/tritsystem/012-trit-search/main/install.sh | bash
```
**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/tritsystem/012-trit-search/main/install.ps1 | iex
```

Installs into an isolated venv (`~/.observe`) — never touches your system Python — and
puts three commands on your PATH:

```bash
observe            # desktop GUI search
observe-search     # command-line search
observe-mcp        # MCP server for editors / coding agents
```

First run downloads ~300–500 MB one time (CPU PyTorch + the embedding model). Everything
after that is fully local — nothing leaves your machine. Requires Python 3.10+.

> Prefer pip? `pipx install "git+https://github.com/tritsystem/012-trit-search.git"`

## Develop / run the full research project

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

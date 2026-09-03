# Changelog

All notable changes to **OBSERVE** (and the wider 012-ternary research repo) are
recorded here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [1.0.0] — 2026-09-03

First tagged release. OBSERVE is the shipped product; the rest of the repo is the
research it came out of, kept in the open.

### OBSERVE — local semantic code search

- **One-line install** (`install.sh` / `install.ps1`) into an isolated
  `~/.observe` venv; `pipx` install also supported. Nothing leaves the machine
  after the one-time model download.
- **Three entry points** from one core: `observe` (desktop GUI), `observe-search`
  (CLI), `observe-mcp` (MCP server for editors and coding agents).
- **Fine-tuned MiniLM** code-search embeddings — measured 96% vs 92% (hard
  benchmark) and 92% vs 76.7% (real OSS code) against the stock model.
- **Ternary-quantized index** — ~20× smaller on disk than float32; the accuracy
  cost is measured and documented (7.35pp), not hidden.
- **Function-boundary chunking** — replaced blind character-window chunking after
  a ground-truth recall test; chunk-level recall recovered from 29% → 86% at zero
  query-time cost.
- **Hybrid search** — lexical + semantic, with a measured comparison against grep
  (grep wins 5/5 when the identifier is known; semantic's real edge is vocabulary
  mismatch).
- **Chunk-provenance lineage** across commits; **incremental re-indexing**.
- **MCP token cost** measured: 66.3% fewer tokens than the prior search tool, via
  dedup / relevance-cutoff / formatting.

### Research (same repo, `paper/`)

- Six-study **scoping-rule ladder** (weighted combination vs voting under
  calibrated vs contaminated evidence), assembled as a standalone preprint.
- Balanced-ternary CPU emulator + SystemVerilog RTL (36/36 testbench cases pass
  in simulation; not run on silicon).
- Cross-substrate symmetry-selection findings; ternary weight-quantization study
  (all three PTQ/QAT/mixed options fail to recover accuracy without retraining —
  reported as a negative).

[Unreleased]: https://github.com/tritsystem/012-trit-search/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/tritsystem/012-trit-search/releases/tag/v1.0.0

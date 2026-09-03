# Research trail

The 012-ternary repo is a research monorepo. OBSERVE is the product that came out
of it; this file is the trail of measured findings behind it — positive and
negative, each tied to a script in `paper/` or the repo root.

## Falsifiable test series

- **[paper/resonance_domain.md](paper/resonance_domain.md)** — a fully falsifiable
  series: 5 honest experiments, 1 confirmed win, 4 disconfirmed hypotheses, with a
  precise explanation for the boundary.
- **[paper/cross_substrate_symmetry_findings.md](paper/cross_substrate_symmetry_findings.md)**
  — this project's symmetry/resonance results synthesised against an independent
  acoustic-MEMS-plate study: four substrates, four different answers to "does
  breaking symmetry help computation?". Closed with a prospective confirmation —
  giving the Spikeling resonator bank the coupling richness its negative result
  said it lacked, and passing a pre-registered capability gate first, replicates
  the acoustic even/odd dichotomy at full strength (even-order gap +0.254, odd
  −0.003), making the selection rule a two-substrate result and the capability bar
  a *predictive* precondition rather than a post-hoc excuse.

## The scoping-rule ladder (six studies)

The consensus-gate primitive, applied and stress-tested across independent fields:

1. **[paper/npc_consensus_findings.md](paper/npc_consensus_findings.md)** — applied
   to a real game's (Tribe's) fight-or-flee NPC logic, independently of any neural
   net: a clean, statistically robust win (+1.8pp accuracy, 30/30 seeds).
2. **[paper/order_acceptance_findings.md](paper/order_acceptance_findings.md)** — the
   same primitive on a *different* decision shape in the same game, where it loses
   just as decisively (−4.6pp, 30/30 seeds).
3. **[Spikeling/research/POPULATION_CODING_FINDINGS.md](https://github.com/tritsystem/Spikeling/blob/main/research/POPULATION_CODING_FINDINGS.md)**
   — population-coding theory + robust statistics: confirmed, dramatically.
4. **[paper/tmr_findings.md](paper/tmr_findings.md)** — classical Triple Modular
   Redundancy: confirmed, softly.
5. **[paper/ensemble_ml_findings.md](paper/ensemble_ml_findings.md)** — real
   scikit-learn ensembles under covariate shift: weak/inconclusive, which sharpened
   the rule — it's *rank-reordering* of relative reliability that matters, not
   miscalibration in general.
6. **[paper/rank_reorder_findings.md](paper/rank_reorder_findings.md)** — the
   designed-to-kill test: a shift engineered (on validation data only) to flip the
   previously-best classifier to worst reverses the weighted decoder's advantage
   (majority vote wins, t=−2.70); a damage-matched, rank-preserving shift leaves
   weighted voting winning (paired contrast t=+2.55; dose-response rho=+0.44,
   p=3.9e−07).

**Final form of the rule:** weighted combination beats voting exactly as long as
the calibration-time reliability ranking still holds at decision time. The whole
program is assembled as a standalone preprint in
[paper/scoping_rule_paper.tex](paper/scoping_rule_paper.tex) (Zenodo upload fields
in [paper/SCOPING_RULE_ZENODO.txt](paper/SCOPING_RULE_ZENODO.txt)).

## Ternary quantization

- **[paper/weight_quantization_findings.md](paper/weight_quantization_findings.md)**
  — ternary quantization applied to real transformer weights (PTQ, QAT, mixed
  precision). All three fail to recover accuracy without retraining. Surprising
  sub-result: ternary attention-only is *worse* than ternary-all, because FFN
  quantization noise partially compensates for attention damage.

## OBSERVE's own measured tradeoffs

- **[paper/token_reduction_findings.md](paper/token_reduction_findings.md)** — 66.3%
  fewer tokens than the prior search tool, via dedup / relevance-cutoff /
  formatting — *not* the compression or consensus-gate mechanisms an earlier draft
  wrongly credited.
- **[paper/quality_benchmark_findings.md](paper/quality_benchmark_findings.md)** — the
  other half of that tradeoff: a ground-truth recall test showing the savings are
  not free. A real reproduced case where dedup/cutoff dropped a correct answer (86%
  vs 71% combined recall), root-caused to blind character-window chunking and fixed
  in testing (function-boundary chunking recovers chunk-level recall 29% → 86% at
  zero query-time cost).
- **[paper/grep_vs_semantic_findings.md](paper/grep_vs_semantic_findings.md)** — vs
  grep, the tool an assistant reaches for by default: grep wins outright when the
  identifier is known (5/5 vs 3/5 recall); semantic search's real advantage narrows
  to genuine vocabulary mismatches (legacy renames, domain jargon, orphaned
  references), once obvious-synonym queries are excluded.

## Deeper docs

- **[DOCS.md](DOCS.md)** — architecture + key results
- **[FILES.md](FILES.md)** — every file, what it does, how to run it

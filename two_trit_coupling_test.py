#!/usr/bin/env python3
"""two_trit_coupling_test.py -- Condition A of the "ternary x3=9" thread.

Composing two independent trits (each gate x sign, 3 states) gives 3x3=9
joint states -- pure combinatorics, nothing interesting on its own. The real
question: does COUPLING two trits together change computational capability
versus running them fully independently? This is Condition A only: a RIGID
(commensurate) coupling, where trit B's gate is forced to always equal trit
A's gate. Pre-registered prediction (from ternary_torus_arnold_finding /
consensus_scoping_rule_ladder): commensurate/locked coupling collapses
reachable degrees of freedom and should HURT capability relative to running
the two trits independently -- the same mechanism already established for
continuous phase-locking, now tested on a discrete gate coupling.

MECHANISM UNDER TEST: split a leaky-ESN reservoir's state vector into two
halves (GROUP_A, GROUP_B). Each half is ternary-GATED on/off over time
(a trit's "gate" dimension; "sign" is not exercised here -- this isolates
the gate-coupling question cleanly, sign composition is a separate test).
  INDEPENDENT : GROUP_A and GROUP_B toggle on/off on different, coprime
                periods -> the two groups are rarely BOTH off at once, so
                the readout almost always has some live state to read.
  RIGID (Condition A): GROUP_B's on/off schedule is forced identical to
                GROUP_A's -> whenever A is off, B is off too -> the readout
                gets a fully-zero state for that entire half of all
                timesteps. This is a real, structural reduction of
                reachable joint states (5 of 9 possible (gate_A,gate_B,sign)
                combinations survive under rigid coupling vs all 9 under
                independent gating -- see the printed DOF check below).
  DENSE (ceiling): both groups always on (no gating at all).

TASK: classical Jaeger memory capacity (MC) -- how well a linear readout on
reservoir state can reconstruct u(t-k) for k=1..K, summed R^2 across k.
Chosen because it's a standard, trusted reservoir-capability metric already
used in this portfolio's other reservoir work, not a new one-off metric.

PRE-REGISTERED (Condition A):
  CONFIRM     MC_rigid meaningfully below MC_independent (rigid coupling
              hurts, consistent with the degeneracy-collapse mechanism).
  DISCONFIRM  MC_rigid >= MC_independent (coupling doesn't hurt here --
              report the null honestly, it would mean the collapse
              mechanism doesn't transfer to this discrete gating case).
              CONFIRMED 2026-07-17: rigid lost ~24% MC vs independent.

CONDITION B (added 2026-07-17): is it COUPLING itself that hurts, or
specifically a COMMENSURATE/rational relationship between the two trits'
clocks? Two new conditions, both "coupled" (trit B's period is DERIVED
from trit A's, unlike "independent" which uses two unrelated periods):
  golden           : period_B = period_A * phi (golden ratio) -- coupled,
                     but INCOMMENSURATE, so relative phase between the two
                     gates never exactly re-syncs (KAM: least resonant).
  rational_coupled : period_B = period_A * 1.5 -- coupled AND
                     COMMENSURATE (a simple rational ratio), so relative
                     phase repeats on a short cycle -- this is the "coupled
                     but still locks" control that isolates whether
                     coupling per se is the problem, or specifically
                     commensurability is.

PRE-REGISTERED (Condition B):
  P1  golden MC within ~10% of independent's -- being coupled to A's clock
      doesn't cost capability AS LONG AS the ratio is incommensurate.
  P2  rational_coupled MC meaningfully below golden/independent, similar
      magnitude to rigid's ~24% loss -- confirms it's the commensurate
      RATIO that collapses degrees of freedom, not coupling itself.
  DISCONFIRM  golden loses about as much as rational_coupled -- would mean
      incommensurateness doesn't rescue capability once ANY coupling
      exists in this discrete gating substrate (a real boundary against
      the continuous-rotor version of this claim in
      ternary_torus_arnold_finding.md -- report honestly, don't spin).
"""
import numpy as np

N_RES = 100
LEAK = 0.3
TOTAL = 6000
WASHOUT = 300
K_MAX = 20          # memory capacity lags to test
SEEDS = 5

rng_master = np.random.default_rng(0)


def build_reservoir(seed: int):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N_RES, N_RES)) * (rng.random((N_RES, N_RES)) < 0.1)
    W *= 0.9 / max(abs(np.linalg.eigvals(W)))
    win = 0.5 * rng.standard_normal(N_RES)
    return W, win


def run_states(u, W, win):
    s = np.zeros(N_RES)
    out = np.empty((len(u), N_RES))
    for t, v in enumerate(u):
        s = (1 - LEAK) * s + LEAK * np.tanh(W @ s + win * v)
        out[t] = s
    return out


def gate_schedule(period: int, length: int, phase: int = 0) -> np.ndarray:
    """1 for the first half of each `period`-tick block, 0 for the second
    half -- a simple 50% duty-cycle square wave, phase-shifted."""
    t = (np.arange(length) + phase) % period
    return (t < period // 2).astype(float)


def apply_gating(states: np.ndarray, gate_A: np.ndarray, gate_B: np.ndarray) -> np.ndarray:
    half = N_RES // 2
    masked = states.copy()
    masked[:, :half] *= gate_A[:, None]
    masked[:, half:] *= gate_B[:, None]
    return masked


def memory_capacity(states: np.ndarray, u: np.ndarray) -> float:
    """Standard Jaeger MC: sum over lags k of R^2 reconstructing u(t-k) from
    state(t), on a held-out split, with ridge regression."""
    T = len(u)
    split = T // 2
    total_mc = 0.0
    for k in range(1, K_MAX + 1):
        X = states[k:]
        y = u[:-k] if k > 0 else u
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]
        Xtr, Xte = X[:split], X[split:n]
        ytr, yte = y[:split], y[split:n]
        A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ ytr)
        Ate = np.hstack([Xte, np.ones((len(Xte), 1))])
        pred = Ate @ w
        var_y = yte.var()
        if var_y < 1e-12:
            continue
        r2 = max(0.0, 1.0 - np.mean((pred - yte) ** 2) / var_y)
        total_mc += r2
    return total_mc


def dof_check() -> None:
    """Cheap, deterministic sanity check BEFORE the stochastic benchmark --
    'validate the instrument' (lesson from ternary_torus_arnold_finding:
    a prior lock-detector gave a false null before this discipline was
    added). Directly counts reachable (gate_A, gate_B, sign_A, sign_B)
    joint states under each condition."""
    reachable_independent = 0
    reachable_rigid = 0
    for gA in (0, 1):
        for gB in (0, 1):
            signs_A = (1,) if gA == 0 else (-1, 1)   # gate off -> sign irrelevant, 1 state
            signs_B = (1,) if gB == 0 else (-1, 1)
            reachable_independent += len(signs_A) * len(signs_B)
            if gA == gB:      # rigid: only diagonal (gA==gB) pairs are reachable at all
                reachable_rigid += len(signs_A) * len(signs_B)
    print(f"  DOF check: independent gating reaches {reachable_independent}/9 joint states; "
          f"rigid coupling reaches {reachable_rigid}/9.")
    assert reachable_independent == 9, "independent gating should reach all 9 combinatorial states"
    assert reachable_rigid < reachable_independent, "rigid coupling should be a strict subset"


def run_condition(seed: int, mode: str) -> float:
    rng = np.random.default_rng(1000 + seed)
    u = rng.uniform(-1, 1, size=TOTAL)
    W, win = build_reservoir(seed)
    states = run_states(u, W, win)
    states, u_used = states[WASHOUT:], u[WASHOUT:]

    phi = (1 + 5 ** 0.5) / 2

    if mode == "dense":
        masked = states
    else:
        period_A = 17    # coprime-ish periods so independent gating rarely
        period_B = 23    # goes fully dark at the same time
        gate_A = gate_schedule(period_A, len(states))
        if mode == "independent":
            gate_B = gate_schedule(period_B, len(states), phase=7)
        elif mode == "rigid":
            gate_B = gate_A.copy()   # Condition A: forced identical schedule
        elif mode == "golden":
            # Condition B: B's period is DERIVED from A's (coupled) but at
            # an incommensurate ratio -- relative phase never exactly repeats
            gate_B = gate_schedule(period_A * phi, len(states))
        elif mode == "rational_coupled":
            # Condition B control: coupled AND commensurate (simple 3/2 ratio)
            gate_B = gate_schedule(period_A * 1.5, len(states))
        else:
            raise ValueError(mode)
        masked = apply_gating(states, gate_A, gate_B)

    return memory_capacity(masked, u_used)


def main():
    print("=" * 78)
    print("  TWO-TRIT COUPLING TEST -- Condition A (rigid/commensurate gate coupling)")
    print("=" * 78)
    dof_check()
    print()

    modes = ["dense", "independent", "rigid", "golden", "rational_coupled"]
    results = {m: [] for m in modes}
    for seed in range(SEEDS):
        for mode in modes:
            results[mode].append(run_condition(seed, mode))

    print(f"  {'condition':<18}{'mean MC':>10}{'std':>8}   (K_max={K_MAX}, {SEEDS} seeds)")
    for mode in modes:
        vals = np.array(results[mode])
        print(f"  {mode:<18}{vals.mean():>10.3f}{vals.std():>8.3f}")

    ind_mean = np.mean(results["independent"])
    rig_mean = np.mean(results["rigid"])
    gold_mean = np.mean(results["golden"])
    rat_mean = np.mean(results["rational_coupled"])

    drop_rigid = 100 * (ind_mean - rig_mean) / ind_mean if ind_mean else float("nan")
    drop_golden = 100 * (ind_mean - gold_mean) / ind_mean if ind_mean else float("nan")
    drop_rational = 100 * (ind_mean - rat_mean) / ind_mean if ind_mean else float("nan")

    print(f"\n  vs independent baseline (MC={ind_mean:.3f}):")
    print(f"    rigid             : {drop_rigid:+.1f}%  (Condition A, already confirmed)")
    print(f"    golden            : {drop_golden:+.1f}%  (Condition B: coupled, incommensurate)")
    print(f"    rational_coupled  : {drop_rational:+.1f}%  (Condition B control: coupled, commensurate)")

    print("\n--- Condition A verdict ---")
    if rig_mean < ind_mean * 0.9:
        print(f"CONFIRMED: rigid/commensurate coupling loses {drop_rigid:.1f}% memory capacity "
              f"vs independent gating.")
    else:
        print("DISCONFIRMED or negligible: rigid coupling did not meaningfully hurt capability here.")

    print("\n--- Condition B verdict ---")
    p1_ok = drop_golden < 10.0        # golden within ~10% of independent
    p2_ok = drop_rational > 10.0 and rat_mean < gold_mean * 0.95   # rational meaningfully worse than golden
    if p1_ok and p2_ok:
        print(f"CONFIRMED: golden/incommensurate coupling ({drop_golden:+.1f}%) stays close to "
              f"independent, while rational/commensurate coupling ({drop_rational:+.1f}%) loses "
              f"real capacity -- it's the RATIO, not coupling per se, that collapses degrees of "
              f"freedom. Transfers the continuous-rotor KAM story to this discrete gating substrate.")
    elif not p1_ok and drop_golden > 10.0:
        print(f"DISCONFIRMED: golden coupling ALSO lost meaningful capacity ({drop_golden:+.1f}%), "
              f"about as much as rational_coupled ({drop_rational:+.1f}%). Incommensurateness does "
              f"NOT rescue capability once any coupling exists in this discrete substrate -- a real "
              f"boundary against the continuous-phase version of this claim. Report honestly.")
    else:
        print(f"MIXED: golden={drop_golden:+.1f}%, rational_coupled={drop_rational:+.1f}% -- doesn't "
              f"cleanly confirm or disconfirm, report the raw numbers as-is, no spin.")


if __name__ == "__main__":
    main()

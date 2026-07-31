"""decision_arbitration.py -- the consensus-gate scoping rule, packaged.

Per PROJECT_decision_arbitration_library.md (Spikeling vault): this is
consolidation, not new discovery. Every formula here is lifted verbatim
from an already-validated test script, not reimplemented from scratch:

  majority_vote()      <- trit_tmr_test.py's decode(): sign(sum(votes))
  weighted_combine()   <- trit_tmr_test.py's decode(): sign(sum(votes*weights))
  calibrate_log_odds_weights()  <- trit_tmr_test.py's calibrate()
  k_of_n_consensus()    <- trit_npc_consensus_test.py's policy_b_flee
                          (sum([...]) >= 2), generalized to any k-of-n

THE RULE (six independent confirmations, see the vault doc for all six
with real numbers): weighted combination beats majority vote exactly as
long as the calibration-time reliability RANKING of the inputs still
holds at decision time. Once that ranking flips, majority vote wins
instead. rank_stability() below is the one genuinely new piece -- a
diagnostic for the actual quantity the rule hinges on, not present as a
reusable function in any of the six source scripts (each one tests a
FIXED regime, stable or drifted, decided in advance; nothing there
diagnoses which regime you're currently in).
"""
from typing import Sequence

import numpy as np


def majority_vote(votes: np.ndarray) -> np.ndarray:
    """votes: (..., n_units) array of +1/-1 (or any signed values -- only
    the sign of each vote matters). Returns sign(sum(votes)) along the
    last axis. Verbatim: trit_tmr_test.py's decode() majority_pred line."""
    return np.sign(np.sum(votes, axis=-1))


def weighted_combine(votes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """votes: (..., n_units). weights: (n_units,), e.g. from
    calibrate_log_odds_weights(). Returns sign(sum(votes*weights)) along
    the last axis. Verbatim: trit_tmr_test.py's decode() weighted_pred line."""
    return np.sign(np.sum(votes * weights, axis=-1))


def calibrate_log_odds_weights(votes: np.ndarray, truth: np.ndarray,
                                clip: tuple = (0.05, 0.95)) -> np.ndarray:
    """votes: (n_calib_trials, n_units) of +1/-1. truth: (n_calib_trials,)
    of +1/-1. Estimates each unit's reliability from this FINITE
    calibration data (not ground-truth knowledge of the true rate -- a
    realistic estimate, exactly as trit_tmr_test.py's calibrate() does),
    returns log-odds weights: log(r_hat / (1 - r_hat)), clipped to avoid
    +/-inf at r_hat=0 or 1."""
    correct = (votes == truth[:, None])
    r_hat = np.clip(correct.mean(axis=0), clip[0], clip[1])
    return np.log(r_hat / (1 - r_hat))


def k_of_n_consensus(signals: Sequence[bool], k: int = 2) -> bool:
    """signals: any sequence of booleans. True iff at least k of them are
    True. Verbatim generalization of trit_npc_consensus_test.py's
    policy_b_flee = sum([low_hp, outnumbered, weak_estimate]) >= 2 (that
    test used k=2, n=3; this is the same primitive at any k/n)."""
    return sum(bool(s) for s in signals) >= k


def rank_stability(calibration_reliabilities: np.ndarray,
                    current_reliabilities: np.ndarray) -> dict:
    """THE NEW piece -- not lifted from any source script, because none
    of the six tests needed to detect which regime they were in; each
    one was TOLD which regime (stable/calibrated vs. drifted/uncalibrated)
    it was testing. A real deployment doesn't get told -- this diagnostic
    is what the rule actually needs to be usable, not just demonstrated.

    Compares the RANK ORDER (not the absolute values) of reliability at
    calibration time vs. now -- the rank-reorder test (vault doc, test #6)
    established that it's specifically rank-reordering that breaks the
    weighted combiner's advantage, not miscalibration/drift in general.

    Returns {"rank_stable": bool, "spearman_rho": float,
    "recommend": "weighted"|"majority"}. rank_stable is True iff the
    argsort order is IDENTICAL (the strict version the rank-reorder test
    actually manipulated); spearman_rho is included as a softer,
    continuous signal for cases near the boundary, not a stated bar from
    any of the six tests -- read it, don't just threshold it blindly."""
    calib_order = np.argsort(calibration_reliabilities)
    current_order = np.argsort(current_reliabilities)
    rank_stable = bool(np.array_equal(calib_order, current_order))

    n = len(calibration_reliabilities)
    calib_ranks = np.argsort(np.argsort(calibration_reliabilities))
    current_ranks = np.argsort(np.argsort(current_reliabilities))
    d_squared = np.sum((calib_ranks - current_ranks) ** 2)
    spearman_rho = 1 - (6 * d_squared) / (n * (n**2 - 1)) if n > 1 else 1.0

    return {
        "rank_stable": rank_stable,
        "spearman_rho": float(spearman_rho),
        "recommend": "weighted" if rank_stable else "majority",
    }

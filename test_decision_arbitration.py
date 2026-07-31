"""Validates decision_arbitration.py's extracted functions reproduce the
ORIGINAL scripts' behavior on the same inputs -- step 2 of
PROJECT_decision_arbitration_library.md's build order: "re-run all six
tests against the library's own functions... to confirm no behavior
changed in extraction." Not a re-run of all six studies (that's a bigger
task); this checks the extraction itself is faithful, the prerequisite
for trusting any re-run.
"""
import numpy as np

from decision_arbitration import (
    calibrate_log_odds_weights,
    k_of_n_consensus,
    majority_vote,
    rank_stability,
    weighted_combine,
)


def test_majority_vote_matches_inline_sign_sum():
    rng = np.random.default_rng(0)
    votes = rng.choice([-1, 1], size=(500, 5))
    expected = np.sign(np.sum(votes, axis=1))  # trit_tmr_test.py's decode(), inline
    actual = majority_vote(votes)
    assert np.array_equal(expected, actual)
    print("[PASS] majority_vote matches trit_tmr_test.py's inline majority_pred")


def test_weighted_combine_matches_inline_sign_sum():
    rng = np.random.default_rng(1)
    votes = rng.choice([-1, 1], size=(500, 5)).astype(float)
    weights = rng.uniform(0.1, 2.0, size=5)
    expected = np.sign(np.sum(votes * weights[None, :], axis=1))
    actual = weighted_combine(votes, weights)
    assert np.array_equal(expected, actual)
    print("[PASS] weighted_combine matches trit_tmr_test.py's inline weighted_pred")


def test_calibrate_matches_original_tmr_calibrate():
    """Full pipeline check: sample reliabilities, run calibration trials,
    derive weights via the library function, compare against the
    ORIGINAL calibrate()'s exact formula re-implemented inline here."""
    rng = np.random.default_rng(2)
    n_units, n_calib = 5, 2000
    true_r = rng.uniform(0.55, 0.95, size=n_units)

    truth = rng.integers(0, 2, size=n_calib) * 2 - 1
    correct_mask = rng.uniform(size=(n_calib, n_units)) < true_r[None, :]
    votes = np.where(correct_mask, truth[:, None], -truth[:, None])

    actual_weights = calibrate_log_odds_weights(votes, truth)

    correct = (votes == truth[:, None])
    r_hat = np.clip(correct.mean(axis=0), 0.05, 0.95)
    expected_weights = np.log(r_hat / (1 - r_hat))

    assert np.allclose(actual_weights, expected_weights)
    print("[PASS] calibrate_log_odds_weights matches trit_tmr_test.py's calibrate() exactly")


def test_k_of_n_matches_npc_policy_b():
    """Reproduces trit_npc_consensus_test.py's policy_b_flee logic via
    k_of_n_consensus(k=2, n=3) and checks agreement across real trials
    from that exact scenario generator."""
    rng = np.random.default_rng(3)
    OUTNUMBER_THRESHOLD, LOW_HP_FRAC = 4, 0.3
    agree = 0
    n_trials = 5000
    for _ in range(n_trials):
        hp_ratio = rng.uniform(0.0, 1.0)
        rival_count = rng.integers(0, 8)
        base_power_self = max(0.1, rng.normal(1.0, 0.15))
        avg_rival_power = rng.uniform(0.5, 1.3)
        npc_power = hp_ratio * base_power_self
        rival_power = rival_count * avg_rival_power
        low_hp = hp_ratio < LOW_HP_FRAC
        outnumbered = rival_count >= OUTNUMBER_THRESHOLD
        est_self = npc_power + rng.normal(0, 0.25)
        est_rival = rival_power + rng.normal(0, 0.5)
        weak_estimate = est_self < est_rival

        original = sum([low_hp, outnumbered, weak_estimate]) >= 2
        library = k_of_n_consensus([low_hp, outnumbered, weak_estimate], k=2)
        agree += (original == library)
    assert agree == n_trials
    print(f"[PASS] k_of_n_consensus(k=2) matches trit_npc_consensus_test.py's "
          f"policy_b_flee on {n_trials}/{n_trials} real trials")


def test_rank_stability_diagnostic_sane():
    """Sanity check on the ONE genuinely new function: stable ranking ->
    recommend weighted; a real reordering -> recommend majority."""
    calib = np.array([0.9, 0.7, 0.5])
    same_order = np.array([0.95, 0.6, 0.4])  # same rank order, different values
    flipped = np.array([0.4, 0.6, 0.95])     # rank order reversed

    r1 = rank_stability(calib, same_order)
    assert r1["rank_stable"] is True and r1["recommend"] == "weighted"

    r2 = rank_stability(calib, flipped)
    assert r2["rank_stable"] is False and r2["recommend"] == "majority"
    print(f"[PASS] rank_stability: stable order -> 'weighted' (rho={r1['spearman_rho']:.2f}), "
          f"flipped order -> 'majority' (rho={r2['spearman_rho']:.2f})")


if __name__ == "__main__":
    test_majority_vote_matches_inline_sign_sum()
    test_weighted_combine_matches_inline_sign_sum()
    test_calibrate_matches_original_tmr_calibrate()
    test_k_of_n_matches_npc_policy_b()
    test_rank_stability_diagnostic_sane()
    print("\nALL EXTRACTION-FIDELITY CHECKS PASSED")

"""
Runs each condition of qwen27b_prune_ternary_test.py as a SEPARATE, fresh
Python process -- disk-offloaded model loading a second time within the
same process segfaults (real, reproduced: Condition A succeeded cleanly,
Condition B crashed instantly on its second load attempt, same process).
One process per condition sidesteps whatever stale accelerate/CUDA state
causes that, at the cost of reloading the tokenizer/model machinery each
time (cheap relative to the model weights themselves).

Usage:
  python run_all.py --model-path C:/models/qwen3.8-27B-bf16
  python run_all.py --model-path ... --conditions A B C D    (skip E/F, the slow training ones)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

LABELS = {
    "A": "A) baseline",
    "B": "B) prune-only",
    "C": "C) prune+ternary",
    "D": "D) prune+int4",
    "E": "E) prune+LoftQ+Muon recovery",
    "F": "F) temporal prune ramp+LoftQ+Muon",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--offload-folder", default="C:/models/offload-tmp")
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    parser.add_argument("--recovery-steps", type=int, default=150)
    args = parser.parse_args()

    results = {}
    result_dir = Path(args.offload_folder).parent / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions:
        result_file = result_dir / f"condition_{cond}.json"
        cmd = [
            sys.executable, "qwen27b_prune_ternary_test.py",
            "--model-path", args.model_path,
            "--offload-folder", args.offload_folder,
            "--condition", cond,
            "--recovery-steps", str(args.recovery_steps),
            "--result-file", str(result_file),
        ]
        print(f"\n{'#'*70}\n# Launching condition {cond} ({LABELS[cond]}) as a fresh process\n{'#'*70}")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"!! Condition {cond} FAILED (exit {proc.returncode}) after {elapsed:.0f}s -- skipping, continuing with remaining conditions")
            continue
        if result_file.exists():
            with open(result_file) as f:
                results[cond] = json.load(f)
            print(f"Condition {cond} done in {elapsed:.0f}s: ppl={results[cond]['ppl']:.3f}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if "A" not in results:
        print("  Baseline (A) never completed -- can't compute deltas.")
        for cond, r in results.items():
            print(f"  {LABELS[cond]:36s}: ppl={r['ppl']:8.3f}  sparsity={r['sparsity']*100:.1f}%")
        return

    ppl_base = results["A"]["ppl"]
    for cond in ["A", "B", "C", "D", "E", "F"]:
        if cond not in results:
            print(f"  {LABELS[cond]:36s}: (did not complete)")
            continue
        r = results[cond]
        delta = (r["ppl"] / ppl_base - 1) * 100
        print(f"  {LABELS[cond]:36s}: ppl={r['ppl']:8.3f}  ({delta:+.1f}% vs baseline)  sparsity={r['sparsity']*100:.1f}%")


if __name__ == "__main__":
    main()

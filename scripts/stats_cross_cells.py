# -*- coding: utf-8 -*-
"""Per-cell confidence intervals and n for the cross-cohort matrix (Table 3).

Why this file exists
--------------------
Reviewers asked for a CI on every cross-cohort cell and an explicit target-set size
(R1-M6, action item 11). The original matrix runs stored only per-seed AUCs, so no
interval could be computed from artefacts. The add-on runs (train_cross_cohort_ext.py)
save the per-seed predicted probabilities, and this script turns them into:

  * per-cell 95% bootstrap CI **of the reported point estimate**: the point estimate is
    the mean over seeds of the per-seed AUC, so each resample recomputes the AUC of every
    seed on that resample and averages over seeds (2,000 resamples of the TARGET
    evaluation set; percentile method; RandomState(0); resampling unit = image);
  * per-cell mean +/- SD over seeds (as in the draft);
  * per-cell n and number of positives;
  * the same statistics for the same-cohort (diagonal) cells.

Estimand note (2026-10-07). Earlier revisions put the mean-over-seeds AUC next to a CI
computed on the seed-averaged probability vector. Those are two different estimators, and
in one cell the reported point estimate fell outside its own interval. The CI below is
computed for the same estimator as the point estimate; `auc_ensemble` is still reported,
but its interval (`ci95_ensemble_reference`) is no longer used as the CI of the seed-mean.

Paths are resolved relative to the repository root (see `scripts/paths.py`); the per-seed
probabilities of a run live in `predictions/cross_cohort/<prefix>/` and the outputs are
written to `results/cross/`.

Outputs: results/cross/cells_ci_<prefix>.json / .txt
Usage:   python scripts/stats_cross_cells.py --prefix resnet50_full
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P  # noqa: E402

import glob
import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

OUT_DIR = str(P.RESULTS / "cross")
PRED_ROOT = str(P.PREDICTIONS / "cross_cohort")
COHORTS = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]
BOOT, SEED = 2000, 0


def boot_ci_seedmean(P_arr, labels, iters=BOOT, seed=SEED):
    """CI of the mean-over-seeds AUC (same estimator as `mean_over_seeds`).

    P_arr : (n_seeds, n) per-seed predicted probabilities for one cell.
    """
    n = labels.shape[0]
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(iters):
        idx = rng.choice(n, n, replace=True)
        y = labels[idx]
        if y.min() == y.max():
            continue
        vals.append(float(np.mean([roc_auc_score(y, P_arr[s][idx]) for s in range(P_arr.shape[0])])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def boot_ci_ensemble(probs, labels, iters=BOOT, seed=SEED):
    """CI of the AUC of the seed-averaged probability vector (kept for reference)."""
    n = len(labels)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(iters):
        idx = rng.choice(n, n, replace=True)
        if labels[idx].min() == labels[idx].max():
            continue
        vals.append(roc_auc_score(labels[idx], probs[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="resnet50_full")
    args = ap.parse_args()
    pred_dir = os.path.join(PRED_ROOT, args.prefix)
    assert os.path.isdir(pred_dir), f"missing {pred_dir} (run train_cross_cohort_ext.py --npz 1)"

    cells = {}
    for src in COHORTS:
        row = {}
        for tgt in COHORTS:
            files = sorted(glob.glob(os.path.join(pred_dir, f"s*_{src}_to_{tgt}.npz")))
            assert files, f"no predictions for {src} -> {tgt}"
            per_seed, rows, labels = [], [], None
            for f in files:
                d = np.load(f)
                p, y = d["probs"].astype(np.float64)[:, 1], d["labels"]
                if labels is None:
                    labels = y
                else:
                    assert np.array_equal(y, labels), f"label mismatch across seeds: {f}"
                rows.append(p)
                per_seed.append(float(roc_auc_score(y, p)))
            P_seeds = np.vstack(rows)
            probs = P_seeds.mean(axis=0)
            lo, hi, kept = boot_ci_seedmean(P_seeds, labels)
            ens_lo, ens_hi, _ = boot_ci_ensemble(probs, labels)
            row[tgt] = {
                "mean_over_seeds": float(np.mean(per_seed)),
                "sd_over_seeds": float(np.std(per_seed, ddof=1)),
                "n_seeds": len(files),
                "per_seed": per_seed,
                "auc_ensemble": float(roc_auc_score(labels, probs)),
                "ci95_low": lo, "ci95_high": hi,
                "ci95_estimand": "mean-over-seeds AUC",
                "ci95_ensemble_reference": [ens_lo, ens_hi],
                "n_test": int(len(labels)), "n_pos": int(labels.sum()),
                "bootstrap_iters_used": kept,
            }
        cells[src] = row

    payload = {"date": "2026-10-07", "prefix": args.prefix,
               "script": "stats_cross_cells.py",
               "protocol": {"bootstrap": {"iterations": BOOT, "method": "percentile",
                                          "ci": [2.5, 97.5], "random_state": SEED,
                                          "resampling_unit": "image",
                                          "estimand": "mean over seeds of the per-seed AUC"},
                            "point_estimate": "mean +/- SD of the per-seed AUC; the 95% CI is "
                                              "the bootstrap interval of the same mean-over-seeds AUC",
                            "revision": "2026-10-07: CI moved off the seed-averaged probability "
                                        "vector onto the reported point estimate"},
               "cells": cells}
    with open(os.path.join(OUT_DIR, f"cells_ci_{args.prefix}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    L = [f"CROSS-COHORT CELLS: n, mean+/-SD (5 seeds) and 95% CI  [{args.prefix}]",
         "CI: 2000 bootstrap resamples of the target evaluation set, estimand = mean over seeds "
         "of the per-seed AUC, percentile method, RandomState(0)", ""]
    L.append("src \\ tgt".ljust(12) + "".join(c.center(34) for c in COHORTS))
    for src in COHORTS:
        line = src.ljust(12)
        for tgt in COHORTS:
            c = cells[src][tgt]
            line += f"{c['mean_over_seeds']:.3f}+/-{c['sd_over_seeds']:.3f} [{c['ci95_low']:.3f},{c['ci95_high']:.3f}]".center(34)
        L.append(line)
    L.append("")
    for tgt in COHORTS:
        c = cells["TBX11K"][tgt]
        L.append(f"target {tgt:11s} n={c['n_test']:5d} (positives {c['n_pos']:4d})")
    with open(os.path.join(OUT_DIR, f"cells_ci_{args.prefix}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[OK] wrote {OUT_DIR}/cells_ci_{args.prefix}.json/.txt")


if __name__ == "__main__":
    main()

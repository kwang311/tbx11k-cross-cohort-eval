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

Since v2.0.0 the numeric blocks live in ``code/p2_lib.py``; this file is the entry point:
it parses ``--prefix``, walks the 4x4 cell grid and formats the two artefacts.

Paths are resolved relative to the repository root (see `code/paths.py`); the per-seed
probabilities of a run live in `predictions/cross_cohort/<prefix>/` and the outputs are
written to `results/cross/`.

Outputs: results/cross/cells_ci_<prefix>.json / .txt
Usage:   python code/stats_cross_cells.py --prefix resnet50_full
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P  # noqa: E402
import p2_lib as L  # noqa: E402

import argparse  # noqa: E402
import glob  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = str(P.RESULTS / "cross")
PRED_ROOT = str(P.PREDICTIONS / "cross_cohort")
COHORTS = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]
BOOT, SEED = L.BOOT_ITERS, L.BOOT_SEED


def cell_stats(P_seeds, labels):
    """All reported numbers for one matrix cell (n, seeds, CI, reference CI).

    Input  : ``P_seeds`` (n_seeds, n) per-seed positive-class scores, ``labels`` (n,).
    Output : dict with mean/SD over seeds, per-seed AUCs, ensemble AUC, the CI of the
             mean-over-seeds AUC, its reference CI, n and positives.
    Caliber: CI of the *same estimator* as the point estimate (see module docstring);
             ensemble interval kept only as ``ci95_ensemble_reference``.
    """
    probs = P_seeds.mean(axis=0)
    per_seed = [L.binary_auc(labels, P_seeds[s]) for s in range(P_seeds.shape[0])]
    lo, hi, kept = L.boot_ci_seedmean(P_seeds, labels)
    ens_lo, ens_hi, _ = L.boot_ci_ensemble(probs, labels)
    mean, sd = L.mean_sd(per_seed)
    return {
        "mean_over_seeds": mean,
        "sd_over_seeds": sd,
        "n_seeds": P_seeds.shape[0],
        "per_seed": per_seed,
        "auc_ensemble": L.binary_auc(labels, probs),
        "ci95_low": lo, "ci95_high": hi,
        "ci95_estimand": "mean-over-seeds AUC",
        "ci95_ensemble_reference": [ens_lo, ens_hi],
        "n_test": int(len(labels)), "n_pos": int(labels.sum()),
        "bootstrap_iters_used": kept,
    }


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
            P_seeds, labels = L.load_binary_seed_scores(files)
            row[tgt] = cell_stats(P_seeds, labels)
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
    L.write_json(os.path.join(OUT_DIR, f"cells_ci_{args.prefix}.json"), payload)

    S = [f"CROSS-COHORT CELLS: n, mean+/-SD (5 seeds) and 95% CI  [{args.prefix}]",
         "CI: 2000 bootstrap resamples of the target evaluation set, estimand = mean over seeds "
         "of the per-seed AUC, percentile method, RandomState(0)", ""]
    S.append("src \\ tgt".ljust(12) + "".join(c.center(34) for c in COHORTS))
    for src in COHORTS:
        line = src.ljust(12)
        for tgt in COHORTS:
            c = cells[src][tgt]
            line += f"{c['mean_over_seeds']:.3f}+/-{c['sd_over_seeds']:.3f} [{c['ci95_low']:.3f},{c['ci95_high']:.3f}]".center(34)
        S.append(line)
    S.append("")
    for tgt in COHORTS:
        c = cells["TBX11K"][tgt]
        S.append(f"target {tgt:11s} n={c['n_test']:5d} (positives {c['n_pos']:4d})")
    L.write_lines(os.path.join(OUT_DIR, f"cells_ci_{args.prefix}.txt"), S)
    print("\n".join(S))
    print(f"\n[OK] wrote {OUT_DIR}/cells_ci_{args.prefix}.json/.txt")


if __name__ == "__main__":
    main()

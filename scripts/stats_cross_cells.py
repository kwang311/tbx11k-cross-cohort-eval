# -*- coding: utf-8 -*-
"""Per-cell confidence intervals and n for the cross-cohort matrix (Table 3).

Why this file exists
--------------------
Reviewers asked for a CI on every cross-cohort cell and an explicit target-set size
(R1-M6, action item 11). The original matrix runs stored only per-seed AUCs, so no
interval could be computed from artefacts. The add-on runs (train_cross_cohort_ext.py)
save the per-seed predicted probabilities, and this script turns them into:

  * per-cell 95% bootstrap CI: 2,000 resamples of the TARGET evaluation set, computed on
    the probabilities averaged over seeds (the same ensemble convention as the rest of
    the paper); percentile method, RandomState(0), resampling unit = image;
  * per-cell mean +/- SD over seeds (as in the draft);
  * per-cell n and number of positives;
  * the same statistics for the same-cohort (diagonal) cells.

Outputs: output_cross/cells_ci_<prefix>.json / .txt
Usage:   python stats_cross_cells.py --prefix resnet50_full
"""
import os
import glob
import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output_cross")
COHORTS = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]
BOOT, SEED = 2000, 0


def boot_ci(probs, labels, iters=BOOT, seed=SEED):
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
    pred_dir = os.path.join(OUT_DIR, f"pred_{args.prefix}")
    assert os.path.isdir(pred_dir), f"missing {pred_dir} (run train_cross_cohort_ext.py with --npz 1)"

    cells = {}
    for src in COHORTS:
        row = {}
        for tgt in COHORTS:
            files = sorted(glob.glob(os.path.join(pred_dir, f"s*_{src}_to_{tgt}.npz")))
            assert files, f"no predictions for {src} -> {tgt}"
            per_seed, probs, labels = [], None, None
            for f in files:
                d = np.load(f)
                p, y = d["probs"].astype(np.float64)[:, 1], d["labels"]
                if probs is None:
                    probs, labels = p, y
                else:
                    assert np.array_equal(y, labels), f"label mismatch across seeds: {f}"
                    probs = probs + p
                per_seed.append(float(roc_auc_score(y, p)))
            probs /= len(files)
            lo, hi, kept = boot_ci(probs, labels)
            row[tgt] = {
                "mean_over_seeds": float(np.mean(per_seed)),
                "sd_over_seeds": float(np.std(per_seed, ddof=1)),
                "n_seeds": len(files),
                "per_seed": per_seed,
                "auc_ensemble": float(roc_auc_score(labels, probs)),
                "ci95_low": lo, "ci95_high": hi,
                "n_test": int(len(labels)), "n_pos": int(labels.sum()),
                "bootstrap_iters_used": kept,
            }
        cells[src] = row

    payload = {"date": "2026-09-13", "prefix": args.prefix,
               "script": "stats_cross_cells.py",
               "protocol": {"bootstrap": {"iterations": BOOT, "method": "percentile",
                                          "ci": [2.5, 97.5], "random_state": SEED,
                                          "resampling_unit": "image"},
                            "point_estimate": "mean +/- SD of the per-seed AUC; CI computed on "
                                              "the seed-averaged probability vector"},
               "cells": cells}
    with open(os.path.join(OUT_DIR, f"cells_ci_{args.prefix}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    L = [f"CROSS-COHORT CELLS: n, mean+/-SD (5 seeds) and 95% CI  [{args.prefix}]",
         "CI: 2000 bootstrap resamples of the target evaluation set, seed-averaged probs, "
         "percentile method, RandomState(0)", ""]
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

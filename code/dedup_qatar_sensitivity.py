# -*- coding: utf-8 -*-
"""Within-class duplicate-image sensitivity for Qatar

Background: the TB class of Qatar contains 3 groups of byte-identical images, 2 of which straddle the train/test split
per qatar_diagnostics.json, md5_within_class.Tuberculosis:
    group A: Tuberculosis-243(train) == Tuberculosis-244(test)
    group B: Tuberculosis-346(train) == Tuberculosis-348(train)
    group C: Tuberculosis-509(test)  == Tuberculosis-510(train)

Question: do these 2 "twin" images (together with the duplicates inside the training set) inflate the in-cohort Qatar AUC (CNN 1.000)
      and the cross-cohort rows (0.408 / 0.511 / 0.566)?

This script does two things:
  Part 1 (no retraining, pure recomputation): using the stored per-seed probabilities `predictions/cross_cohort/resnet50_full/s*_Qatar_to_Qatar.npz`,
      recompute the AUC on the full test set (840) and on the deduplicated test set (838, dropping 244/509).
  Part 2 (retraining, GPU): drop the duplicate files from the Qatar training set (one member of each group, see below),
      keep the test set unchanged (840), retrain Qatar->{TBX11K,Shenzhen,Montgomery,Qatar} x 5 seeds,
      and evaluate under both test-set conventions.

Removal rules (training set only; the test set is untouched so the numbers stay comparable with the published ones):
    group A: drop 243 (its twin 244 is in the test set -> the leakage channel is removed outright)
    group B: drop 348 (both members are in the training set; drop one, keep one, so the training set has no duplicated content)
    group C: drop 510 (its twin 509 is in the test set)
  -> training set 3,360 -> 3,357

Output: results/cross/dedup_qatar.{json,txt}
      predictions/cross_cohort/dedup_qatar/s*_Qatar_to_*.npz
Usage: python dedup_qatar_sensitivity.py [--epochs 15] [--seeds 42,43,44,45,46]
"""
import os
import sys
import json
import time
import glob
import hashlib
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

import train_cross_cohort_ext as T   # reuse the already verified split/training/prediction code (same code path)

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
OUT = str(P.RESULTS / "cross")
PRED_ROOT = str(P.PREDICTIONS / "cross_cohort")
COHORTS = T.COHORTS
DUP_DROP = ["Tuberculosis-243.png", "Tuberculosis-348.png", "Tuberculosis-510.png"]
DUP_TEST = ["Tuberculosis-244.png", "Tuberculosis-509.png"]


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def find_duplicate_groups(paths, labels):
    """Find duplicate groups by MD5 inside the TB class only (same convention as qatar_diagnostics)."""
    tb = [(p, y) for p, y in zip(paths, labels) if y == 1]
    by = {}
    for p, _ in tb:
        by.setdefault(md5(p), []).append(os.path.basename(p))
    return {h: v for h, v in by.items() if len(v) > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rep = {"date": "2026-09-13", "script": os.path.basename(__file__),
           "purpose": "effect of byte-identical duplicates within the Qatar TB class on in-cohort and cross-cohort AUC (part 1 recomputation + part 2 retraining)",
           "drop_from_train": DUP_DROP, "drop_from_test": DUP_TEST}

    # ---------- split (same code path as the published run) ----------
    print("=== building splits (full) ===", flush=True)
    splits = T.build_splits("full")
    q_tr_p, q_tr_y = list(splits["Qatar"]["train"][0]), list(splits["Qatar"]["train"][1])
    q_te_p, q_te_y = list(splits["Qatar"]["test"][0]), list(splits["Qatar"]["test"][1])
    print(f"Qatar split: train={len(q_tr_p)} (pos={sum(q_tr_y)}) test={len(q_te_p)} (pos={sum(q_te_y)})",
          flush=True)

    # ---------- duplicate group verification ----------
    groups = find_duplicate_groups(q_tr_p + q_te_p, q_tr_y + q_te_y)
    rep["duplicate_groups"] = {h[:12]: v for h, v in groups.items()}
    print("duplicate groups:", rep["duplicate_groups"], flush=True)
    assert len(groups) == 3, "expected 3 duplicate groups (see qatar_diagnostics.json)"

    te_names = [os.path.basename(p) for p in q_te_p]
    idx_drop_test = [te_names.index(n) for n in DUP_TEST if n in te_names]
    assert len(idx_drop_test) == 2, "both duplicate test twins must be present"
    print("test indices to drop:", idx_drop_test, flush=True)

    # ---------- Part 1: recomputation from stored probabilities (no retraining) ----------
    part1 = {}
    for seed in seeds:
        f = os.path.join(OUT, "pred_resnet50_full", f"s{seed}_Qatar_to_Qatar.npz")
        d = np.load(f)
        probs, labels = d["probs"], d["labels"]
        # Strict check: the npz label order must match the split test order position by position
        assert np.array_equal(labels, np.asarray(q_te_y)), f"label order mismatch in {f}"
        keep = [i for i in range(len(labels)) if i not in idx_drop_test]
        part1[seed] = {
            "auc_full_test": float(roc_auc_score(labels, probs[:, 1])),
            "auc_dedup_test": float(roc_auc_score(labels[keep], probs[keep, 1])),
        }
        print(f"  [part1] seed {seed}: full={part1[seed]['auc_full_test']:.6f} "
              f"dedup={part1[seed]['auc_dedup_test']:.6f}", flush=True)
    ens = np.mean([np.load(os.path.join(OUT, "pred_resnet50_full", f"s{s}_Qatar_to_Qatar.npz"))["probs"]
                   for s in seeds], axis=0)
    lab = np.load(os.path.join(OUT, "pred_resnet50_full", f"s{seeds[0]}_Qatar_to_Qatar.npz"))["labels"]
    keep = [i for i in range(len(lab)) if i not in idx_drop_test]
    part1["ensemble"] = {"auc_full_test": float(roc_auc_score(lab, ens[:, 1])),
                         "auc_dedup_test": float(roc_auc_score(lab[keep], ens[keep, 1]))}
    rep["part1_reevaluate_existing"] = part1
    print(f"  [part1] ensemble: full={part1['ensemble']['auc_full_test']:.6f} "
          f"dedup={part1['ensemble']['auc_dedup_test']:.6f}", flush=True)

    # ---------- Part 2: retraining on the deduplicated set ----------
    tr_keep = [(p, y) for p, y in zip(q_tr_p, q_tr_y)
               if os.path.basename(p) not in DUP_DROP]
    dropped = [(os.path.basename(p), int(y)) for p, y in zip(q_tr_p, q_tr_y)
               if os.path.basename(p) in DUP_DROP]
    assert len(dropped) == 3, f"expected 3 files dropped, got {dropped}"
    rep["part2_retrain"] = {"n_train_before": len(q_tr_p), "n_train_after": len(tr_keep),
                            "dropped": dropped,
                            "n_pos_before": int(sum(q_tr_y)),
                            "n_pos_after": int(sum(y for _, y in tr_keep))}
    print(f"\n=== part2 retrain: train {len(q_tr_p)} -> {len(tr_keep)} (dropped {dropped}) ===",
          flush=True)

    t0 = time.time()
    cache = {}
    for c in COHORTS:
        # Only Qatar uses the deduplicated training set; the other target cohorts only use test
        if c == "Qatar":
            cache[c] = {"train": T.build_ram_cache([p for p, _ in tr_keep], [y for _, y in tr_keep]),
                        "test": T.build_ram_cache(q_te_p, q_te_y)}
        else:
            cache[c] = {"train": T.build_ram_cache(*splits[c]["train"]),
                        "test": T.build_ram_cache(*splits[c]["test"])}
        print(f"  cache {c}: train={len(cache[c]['train'][0])} test={len(cache[c]['test'][0])}",
              flush=True)
    print(f"  cache built in {time.time()-t0:.1f}s", flush=True)

    device = "cuda" if T.torch.cuda.is_available() else "cpu"
    print(f"  device={device}", flush=True)
    pred_dir = os.path.join(OUT, "pred_dedup_qatar")
    os.makedirs(pred_dir, exist_ok=True)
    results = {}
    for seed in seeds:
        model = T.train_one(cache["Qatar"]["train"], seed, args.epochs, device, "resnet50")
        row = {}
        for tgt in COHORTS:
            probs, labels = T.predict(model, cache[tgt]["test"], device)
            row[tgt] = {"auc": float(roc_auc_score(labels, probs[:, 1])), "n_test": int(len(labels))}
            if tgt == "Qatar":
                k = [i for i in range(len(labels)) if i not in idx_drop_test]
                row[tgt]["auc_dedup_test"] = float(roc_auc_score(labels[k], probs[k, 1]))
                row[tgt]["n_test_dedup"] = int(len(k))
            np.savez_compressed(os.path.join(pred_dir, f"s{seed}_Qatar_to_{tgt}.npz"),
                                probs=probs.astype(np.float32), labels=labels.astype(np.int64))
        results[str(seed)] = row
        print(f"  seed {seed}: " + "  ".join(f"{t}={row[t]['auc']:.3f}" for t in COHORTS), flush=True)
    rep["part2_retrain"]["per_seed"] = results

    summary = {}
    for tgt in COHORTS:
        vals = [results[str(s)][tgt]["auc"] for s in seeds]
        summary[tgt] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                        "per_seed": {str(s): results[str(s)][tgt]["auc"] for s in seeds}}
    vals = [results[str(s)]["Qatar"]["auc_dedup_test"] for s in seeds]
    summary["Qatar_dedup_test"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                                   "per_seed": {str(s): results[str(s)]["Qatar"]["auc_dedup_test"]
                                                for s in seeds}}
    rep["part2_retrain"]["summary"] = summary

    # ---------- Comparison table ----------
    base = json.load(open(os.path.join(OUT, "matrix_summary_resnet50_full.json"), encoding="utf-8"))
    rep["comparison"] = {
        "published_qatar_row": {t: base["Qatar"][t]["mean"] for t in COHORTS},
        "retrained_dedup_qatar_row": {t: summary[t]["mean"] for t in COHORTS},
        "delta": {t: summary[t]["mean"] - base["Qatar"][t]["mean"] for t in COHORTS},
        "published_qatar_same_cohort_test_full": base["Qatar"]["Qatar"]["mean"],
        "existing_model_on_dedup_test_ensemble": part1["ensemble"]["auc_dedup_test"],
        "retrained_on_full_test": summary["Qatar"]["mean"],
        "retrained_on_dedup_test": summary["Qatar_dedup_test"]["mean"],
    }
    print("\n=== comparison ===", flush=True)
    print(json.dumps(rep["comparison"], indent=2), flush=True)

    with open(os.path.join(OUT, "dedup_qatar.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "dedup_qatar.txt"), "w", encoding="utf-8") as f:
        f.write("Qatar duplicate-image sensitivity (2026-09-13)\n")
        f.write(f"dropped from train: {DUP_DROP}\n")
        f.write("part1 (existing models, no retrain):\n")
        for s in seeds:
            f.write(f"  seed {s}: full={part1[s]['auc_full_test']:.6f} "
                    f"dedup={part1[s]['auc_dedup_test']:.6f}\n")
        f.write(f"  ensemble: full={part1['ensemble']['auc_full_test']:.6f} "
                f"dedup={part1['ensemble']['auc_dedup_test']:.6f}\n")
        f.write("part2 (retrained without duplicate training images):\n")
        for t in COHORTS:
            f.write(f"  Qatar->{t:11s} {summary[t]['mean']:.4f} +/- {summary[t]['std']:.4f}\n")
        f.write(f"  Qatar->Qatar (dedup test): {summary['Qatar_dedup_test']['mean']:.4f} "
                f"+/- {summary['Qatar_dedup_test']['std']:.4f}\n")
        f.write("comparison:\n" + json.dumps(rep["comparison"], indent=2) + "\n")
    print("\n[OK] wrote dedup_qatar.json/.txt and pred_dedup_qatar/*.npz", flush=True)


if __name__ == "__main__":
    main()

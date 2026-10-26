# -*- coding: utf-8 -*-
"""Statistical tests for the single-label 9-class task: bootstrap 95% CI + paired Wilcoxon + McNemar + BH-FDR.

Input: output/predictions/{model}_s{seed}.npz (probs (N,9), labels (N,) single class)
Usage: python stats.py
"""
import os
import glob
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
PRED_DIR = str(P.PREDICTIONS / "four_class")
MODELS = ["baseline", "se", "cbam", "bam", "gcn", "gat"]
NUM_CLASSES = 9
BOOTSTRAP_ITERS = 2000
CLASS_NAMES = ["Normal", "Atelectasis", "Cardiomegaly", "Effusion",
               "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]


def load_preds(model):
    files = sorted(glob.glob(os.path.join(PRED_DIR, f"{model}_s*.npz")))
    return [np.load(f) for f in files]


def macro_auc(probs, labels):
    return roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                         labels=list(range(NUM_CLASSES)))


def bootstrap_ci(probs_mean, labels, n_iter=BOOTSTRAP_ITERS):
    n = len(labels)
    vals = []
    rng = np.random.RandomState(0)
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        try:
            vals.append(macro_auc(probs_mean[idx], labels[idx]))
        except ValueError:
            pass
    return np.percentile(vals, [2.5, 97.5])


def mcnemar_p(pred_a, pred_b, labels):
    a = pred_a == labels
    b = pred_b == labels
    a_only = int((a & ~b).sum())
    b_only = int((b & ~a).sum())
    disc = a_only + b_only
    if disc == 0:
        return 1.0
    return stats.binomtest(min(a_only, b_only), n=disc, p=0.5).pvalue


def bh_fdr(p_values, alpha=0.05):
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    thresholds = np.arange(1, n + 1) / n * alpha
    below = sorted_p <= thresholds
    if np.any(below):
        k_max = np.max(np.where(below))
        cutoff = sorted_p[k_max]
        return p <= cutoff, cutoff
    return np.zeros(n, dtype=bool), None


def main():
    print("=" * 70)
    print("1) macro-AUC per model (each seed + mean±std)")
    print("=" * 70)
    model_aucs, model_preds = {}, {}
    for m in MODELS:
        preds = load_preds(m)
        if not preds:
            print(f"  {m}: no prediction files, skipping")
            continue
        aucs = [macro_auc(d["probs"], d["labels"]) for d in preds]
        model_aucs[m] = aucs
        model_preds[m] = preds
        print(f"  {m:8s} mean={np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
              f"seeds={[round(a, 4) for a in aucs]}")

    if "baseline" not in model_aucs:
        print("baseline missing, cannot compare.")
        return

    print("\n" + "=" * 70)
    print(f"2) Bootstrap {BOOTSTRAP_ITERS} resamples, 95% CI (multi-seed ensemble probabilities)")
    print("=" * 70)
    labels_ref = model_preds["baseline"][0]["labels"]
    for m, preds in model_preds.items():
        probs_mean = np.mean(np.stack([d["probs"] for d in preds]), axis=0)
        lo, hi = bootstrap_ci(probs_mean, labels_ref)
        auc_ens = macro_auc(probs_mean, labels_ref)
        print(f"  {m:8s} macroAUC={auc_ens:.4f}  CI=[{lo:.4f}, {hi:.4f}]")

    print("\n" + "=" * 70)
    print("3) Paired Wilcoxon signed-rank (baseline vs augmented)")
    print("=" * 70)
    base_aucs = np.array(model_aucs["baseline"])
    p_wilcox = {}
    for m in MODELS:
        if m == "baseline":
            continue
        diff = np.array(model_aucs[m]) - base_aucs
        p = stats.wilcoxon(diff).pvalue if (len(diff) >= 3 and np.any(diff != 0)) else 1.0
        p_wilcox[m] = p
        print(f"  {m:8s} vs baseline: ΔAUC={np.mean(diff):+.4f}±{np.std(diff):.4f}  "
              f"Wilcoxon p={p:.4f}")

    print("\n" + "=" * 70)
    print("4) McNemar paired test (baseline vs augmented)")
    print("=" * 70)
    for m in MODELS:
        if m == "baseline":
            continue
        ps = []
        for da, db in zip(model_preds["baseline"], model_preds[m]):
            assert np.array_equal(da["labels"], db["labels"]), "labels mismatch"
            ps.append(mcnemar_p(da["probs"].argmax(1), db["probs"].argmax(1), da["labels"]))
        print(f"  {m:8s} McNemar p: median={np.median(ps):.4f}  "
              f"min={np.min(ps):.4f}  max={np.max(ps):.4f}")

    print("\n" + "=" * 70)
    print("5) Multiple-comparison correction (BH-FDR)")
    print("=" * 70)
    enh = [m for m in MODELS if m != "baseline"]
    p_vals = np.array([p_wilcox[m] for m in enh])
    sig, cutoff = bh_fdr(p_vals)
    bonf = 0.05 / len(p_vals)
    print(f"  Bonferroni threshold = {bonf:.4f}; BH-FDR threshold = {cutoff if cutoff else 'none'}")
    for i, m in enumerate(enh):
        mark = "***significant" if sig[i] else "not significant"
        print(f"  {m:8s} Wilcoxon p={p_vals[i]:.4f}  FDR: {mark}")

    print("\n===== Statistical tests complete =====")


if __name__ == "__main__":
    main()

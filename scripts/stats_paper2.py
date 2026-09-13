# -*- coding: utf-8 -*-
"""Paper 2 statistics artifact script (B1/B2 + S4).

Why this file exists
--------------------
The Paper 2 draft reported bootstrap confidence intervals whose generating code was
never written to disk (the earlier numbers lived in output/baseline_4class_stats_20260910.txt,
produced by an ad-hoc run). Reviewers (R1-M1/M2, R3-m1) could not reproduce the CI endpoints
and could not tell which score the restricted active-vs-latent AUC was computed on.
This script is the single, dumped source for every CI, the restricted-AUC score
definition, the decision rule behind the reported recalls, and the per-seed dispersion.

Protocol implemented here (stated so that any reader can reproduce it):
  * restricted active-vs-latent AUC score = p_latent / (p_latent + p_active)
    (the two class posteriors re-normalised within the active/latent sub-problem).
    Alternatives are computed and written out as well, for full transparency.
  * confusion counts / recalls = argmax over the class posteriors of the 10-seed
    ensemble (no tuned threshold).
  * bootstrap: 2,000 iterations, percentile method (2.5 / 97.5), RandomState(0),
    resampling unit = individual image (patients are not resampled; the TBX11K
    validation split carries no patient identifier).
  * seed-to-seed dispersion = mean +/- SD (ddof=1) over the 10 training seeds.

Outputs: output/paper2_stats_20260913.json and .txt
Usage:   G:\\GitHub\\venv_pytorch\\Scripts\\python.exe stats_paper2.py
"""
import os
import glob
import json
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
PRED4 = os.path.join(BASE, "output", "predictions")
PREDA = os.path.join(BASE, "output_a_vs_l", "predictions")
OUT = os.path.join(BASE, "output")

CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
NUM = len(CLASS_NAMES)
BOOT = 2000
BOOT_SEED = 0
CI = (2.5, 97.5)


def macro_auc(probs, labels):
    return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                               labels=list(range(NUM))))


def weighted_auc(probs, labels):
    """Frequency-weighted one-vs-rest AUC (same definition as the draft)."""
    pc, counts = [], []
    for c in range(NUM):
        y = (labels == c).astype(int)
        pc.append(roc_auc_score(y, probs[:, c]))
        counts.append(int((labels == c).sum()))
    pc, counts = np.array(pc), np.array(counts, dtype=float)
    return float((pc * counts / counts.sum()).sum())


def per_class_auc(probs, labels):
    out = {}
    for c in range(NUM):
        y = (labels == c).astype(int)
        out[CLASS_NAMES[c]] = float(roc_auc_score(y, probs[:, c]))
    return out


def restricted_auc(probs, labels, mode="renorm"):
    """Active-vs-latent AUC: only the active (2) and latent (3) images are kept."""
    mask = np.isin(labels, [2, 3])
    y = (labels[mask] == 3).astype(int)          # latent = positive
    p_act, p_lat = probs[mask][:, 2], probs[mask][:, 3]
    if mode == "renorm":                          # PRIMARY definition
        s = p_lat / (p_lat + p_act + 1e-12)
    elif mode == "latent_only":
        s = p_lat
    elif mode == "difference":
        s = p_lat - p_act
    else:
        raise ValueError(mode)
    return float(roc_auc_score(y, s)), y, s


def boot_ci(fn, n, iters=BOOT, seed=BOOT_SEED):
    """Percentile bootstrap CI; fn(idx) -> scalar; resampling unit = image."""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(iters):
        idx = rng.choice(n, n, replace=True)
        try:
            vals.append(fn(idx))
        except ValueError:
            pass
    lo, hi = np.percentile(vals, [CI[0], CI[1]])
    return float(lo), float(hi), len(vals)


def main():
    files4 = sorted(glob.glob(os.path.join(PRED4, "baseline_s*.npz")))
    filesa = sorted(glob.glob(os.path.join(PREDA, "baseline_s*.npz")))
    assert len(files4) == 10, f"expect 10 four-class npz, found {len(files4)}"
    assert len(filesa) == 10, f"expect 10 a-vs-l npz, found {len(filesa)}"

    d4 = [np.load(f) for f in files4]
    da = [np.load(f) for f in filesa]
    labels = d4[0]["labels"]
    for d in d4:
        assert np.array_equal(d["labels"], labels), "val labels differ between seeds"
    labelsa = da[0]["labels"]
    for d in da:
        assert np.array_equal(d["labels"], labelsa), "a-vs-l labels differ between seeds"

    probs_mean = np.mean(np.stack([d["probs"] for d in d4]), axis=0).astype(np.float64)
    probsa_mean = np.mean(np.stack([d["probs"] for d in da]), axis=0).astype(np.float64)

    # ---------- A. four-class, per seed ----------
    per_seed_macro = [macro_auc(d["probs"].astype(np.float64), d["labels"]) for d in d4]
    per_seed_acc = [float(accuracy_score(d["labels"], d["probs"].argmax(1))) for d in d4]
    per_seed_restricted = [restricted_auc(d["probs"].astype(np.float64), d["labels"])[0]
                           for d in d4]

    # ---------- B. four-class ensemble + CI ----------
    ens_macro = macro_auc(probs_mean, labels)
    ens_acc = float(accuracy_score(labels, probs_mean.argmax(1)))
    ens_w = weighted_auc(probs_mean, labels)
    ens_pc = per_class_auc(probs_mean, labels)

    macro_ci = boot_ci(lambda i: macro_auc(probs_mean[i], labels[i]), len(labels))
    acc_ci = boot_ci(lambda i: float(accuracy_score(labels[i], probs_mean[i].argmax(1))),
                     len(labels))
    pc_ci = {}
    for c in range(NUM):
        y_full = (labels == c).astype(int)
        pc_ci[CLASS_NAMES[c]] = boot_ci(
            lambda i, y=y_full, c=c: float(roc_auc_score(y[i], probs_mean[i][:, c])),
            len(labels))

    # ---------- C. restricted active-vs-latent ----------
    restricted = {}
    for mode in ("renorm", "latent_only", "difference"):
        auc, y_r, s_r = restricted_auc(probs_mean, labels, mode)
        lo, hi, _ = boot_ci(lambda i: float(roc_auc_score(y_r[i], s_r[i])), len(y_r))
        d_lo, d_hi, _ = boot_ci(lambda i: float(roc_auc_score(y_r[i], s_r[i])) - 0.5,
                               len(y_r))
        restricted[mode] = {"auc_ensemble": auc, "ci95": [lo, hi],
                            "auc_minus_chance_ci95": [d_lo, d_hi], "n": int(len(y_r))}

    # ---------- D. decision rule: argmax on the ensemble ----------
    pred = probs_mean.argmax(1)
    cm = confusion_matrix(labels, pred, labels=list(range(NUM))).tolist()
    recall = {CLASS_NAMES[c]: int(cm[c][c]) / int(sum(cm[c])) for c in range(NUM)}
    active_as_latent = int(cm[2][3])
    latent_as_active = int(cm[3][2])

    # ---------- E. dedicated active-vs-latent model ----------
    per_seed_a = [float(roc_auc_score(d["labels"], d["probs"].astype(np.float64)[:, 1]))
                  for d in da]
    y_a = (labelsa == 1).astype(int)
    s_a = probsa_mean[:, 1]
    ens_a = float(roc_auc_score(y_a, s_a))
    a_ci = boot_ci(lambda i: float(roc_auc_score(y_a[i], s_a[i])), len(y_a))
    a_d_ci = boot_ci(lambda i: float(roc_auc_score(y_a[i], s_a[i])) - 0.5, len(y_a))
    pred_a = probsa_mean.argmax(1)
    cm_a = confusion_matrix(labelsa, pred_a, labels=[0, 1]).tolist()

    payload = {
        "date": "2026-09-13",
        "script": "stats_paper2.py",
        "protocol": {
            "restricted_score": "p_latent / (p_latent + p_active) over the active/latent "
                                "subset only (alternatives p_latent and p_latent - p_active "
                                "also reported)",
            "decision_rule": "argmax over class posteriors of the 10-seed ensemble "
                             "(no tuned threshold)",
            "bootstrap": {"iterations": BOOT, "method": "percentile",
                          "ci": [CI[0], CI[1]], "random_state": BOOT_SEED,
                          "resampling_unit": "image (patients not resampled; no patient "
                                             "identifier in the TBX11K validation split)"},
            "dispersion": "mean +/- SD (ddof=1) over the 10 training seeds",
            "inputs": {"four_class": "output/predictions/baseline_s{42..51}.npz",
                       "active_vs_latent": "output_a_vs_l/predictions/baseline_s{42..51}.npz"},
        },
        "four_class": {
            "n": int(len(labels)),
            "class_counts": {CLASS_NAMES[c]: int((labels == c).sum()) for c in range(NUM)},
            "per_seed": {"macro_auc": per_seed_macro,
                         "accuracy": per_seed_acc,
                         "restricted_active_vs_latent_auc": per_seed_restricted},
            "per_seed_mean_sd": {
                "macro_auc": [float(np.mean(per_seed_macro)),
                              float(np.std(per_seed_macro, ddof=1))],
                "accuracy": [float(np.mean(per_seed_acc)),
                             float(np.std(per_seed_acc, ddof=1))],
                "restricted_active_vs_latent_auc": [
                    float(np.mean(per_seed_restricted)),
                    float(np.std(per_seed_restricted, ddof=1))],
            },
            "ensemble": {
                "macro_auc": ens_macro, "macro_auc_ci95": list(macro_ci[:2]),
                "accuracy": ens_acc, "accuracy_ci95": list(acc_ci[:2]),
                "weighted_auc": ens_w,
                "per_class_auc": ens_pc,
                "per_class_auc_ci95": {k: list(v[:2]) for k, v in pc_ci.items()},
            },
            "restricted_active_vs_latent": restricted,
            "ensemble_confusion_argmax": cm,
            "recall": recall,
            "counts": {"latent_as_active": latent_as_active,
                       "active_as_latent": active_as_latent},
        },
        "dedicated_active_vs_latent": {
            "n": int(len(labelsa)),
            "class_counts": {"active_tb": int((labelsa == 0).sum()),
                             "latent_tb": int((labelsa == 1).sum())},
            "per_seed_auc": per_seed_a,
            "per_seed_mean_sd": [float(np.mean(per_seed_a)),
                                 float(np.std(per_seed_a, ddof=1))],
            "per_seed_min_max": [float(np.min(per_seed_a)), float(np.max(per_seed_a))],
            "ensemble_auc": ens_a,
            "ensemble_auc_ci95": list(a_ci[:2]),
            "ensemble_auc_minus_chance_ci95": list(a_d_ci[:2]),
            "ensemble_confusion_argmax": cm_a,
        },
    }

    with open(os.path.join(OUT, "paper2_stats_20260913.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    L = []
    L.append("PAPER 2 STATISTICS ARTIFACT 2026-09-13  (stats_paper2.py)")
    L.append("bootstrap: %d iters, percentile, RandomState(%d), unit=image, 95%% CI"
             % (BOOT, BOOT_SEED))
    L.append("restricted a-vs-l score: p_lat/(p_lat+p_act);  decision rule: argmax (ensemble)")
    L.append("")
    L.append("=== FOUR-CLASS TBX11K (n=%d; %s) ===" % (len(labels),
             ", ".join("%s=%d" % (CLASS_NAMES[c], int((labels == c).sum())) for c in range(NUM))))
    L.append("per-seed macro-AUC      : %s" % [round(v, 4) for v in per_seed_macro])
    L.append("  mean +/- SD           : %.4f +/- %.4f" % tuple(payload["four_class"]["per_seed_mean_sd"]["macro_auc"]))
    L.append("per-seed accuracy       : %s" % [round(v, 4) for v in per_seed_acc])
    L.append("  mean +/- SD           : %.4f +/- %.4f" % tuple(payload["four_class"]["per_seed_mean_sd"]["accuracy"]))
    L.append("per-seed restricted AUC : %s" % [round(v, 4) for v in per_seed_restricted])
    L.append("  mean +/- SD           : %.4f +/- %.4f" % tuple(payload["four_class"]["per_seed_mean_sd"]["restricted_active_vs_latent_auc"]))
    L.append("")
    L.append("ensemble macro-AUC      : %.4f   95%% CI [%.4f, %.4f]" % (ens_macro, macro_ci[0], macro_ci[1]))
    L.append("ensemble accuracy       : %.4f   95%% CI [%.4f, %.4f]" % (ens_acc, acc_ci[0], acc_ci[1]))
    L.append("ensemble weighted AUC   : %.4f" % ens_w)
    for c in range(NUM):
        L.append("  per-class %-15s : %.4f   95%% CI [%.4f, %.4f]"
                 % (CLASS_NAMES[c], ens_pc[CLASS_NAMES[c]], pc_ci[CLASS_NAMES[c]][0], pc_ci[CLASS_NAMES[c]][1]))
    L.append("")
    for mode in ("renorm", "latent_only", "difference"):
        r = restricted[mode]
        tag = "  <-- PRIMARY (used in the manuscript)" if mode == "renorm" else ""
        L.append("restricted AUC [%-11s]: %.4f   95%% CI [%.4f, %.4f]   AUC-0.5 CI [%+.4f, %+.4f]  n=%d%s"
                 % (mode, r["auc_ensemble"], r["ci95"][0], r["ci95"][1],
                    r["auc_minus_chance_ci95"][0], r["auc_minus_chance_ci95"][1], r["n"], tag))
    L.append("")
    L.append("ensemble confusion (argmax rows=true cols=pred) order=%s" % CLASS_NAMES)
    for row in cm:
        L.append("   %s" % row)
    L.append("recall: %s" % {k: round(v, 4) for k, v in recall.items()})
    L.append("latent->active = %d/%d ; active->latent = %d/%d"
             % (latent_as_active, int(sum(cm[3])), active_as_latent, int(sum(cm[2]))))
    L.append("")
    L.append("=== DEDICATED ACTIVE-VS-LATENT MODEL (n=%d; active=%d latent=%d) ==="
             % (len(labelsa), int((labelsa == 0).sum()), int((labelsa == 1).sum())))
    L.append("per-seed AUC            : %s" % [round(v, 4) for v in per_seed_a])
    L.append("  mean +/- SD           : %.4f +/- %.4f   (min %.4f max %.4f)"
             % (np.mean(per_seed_a), np.std(per_seed_a, ddof=1), np.min(per_seed_a), np.max(per_seed_a)))
    L.append("ensemble AUC            : %.4f   95%% CI [%.4f, %.4f]   AUC-0.5 CI [%+.4f, %+.4f]"
             % (ens_a, a_ci[0], a_ci[1], a_d_ci[0], a_d_ci[1]))
    L.append("ensemble confusion (argmax) rows=true cols=pred order=[active,latent]: %s" % cm_a)
    with open(os.path.join(OUT, "paper2_stats_20260913.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(L))
    print("\n[OK] wrote %s/paper2_stats_20260913.json/.txt" % OUT)


if __name__ == "__main__":
    main()

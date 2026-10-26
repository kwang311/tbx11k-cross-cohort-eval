# -*- coding: utf-8 -*-
"""Paper 2 statistics artifact script (B1/B2 + S4).

Why this file exists
--------------------
The Paper 2 draft reported bootstrap confidence intervals whose generating code was
never written to disk (the earlier numbers lived in output/baseline_4class_stats.txt,
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

Since v2.0.0 the numeric blocks live in ``code/p2_lib.py``; this file is the entry
point: it fixes the inputs/outputs and assembles the payload, nothing else.

Outputs: results/four_class/paper2_stats.json and .txt
Usage:   python stats_paper2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P  # noqa: E402
import p2_lib as L  # noqa: E402

PRED4 = str(P.PREDICTIONS / "four_class")
PREDA = str(P.PREDICTIONS / "a_vs_l")
OUT = str(P.RESULTS / "four_class")

CLASS_NAMES = L.CLASS_NAMES
NUM = L.NUM_CLASSES
BOOT = L.BOOT_ITERS
BOOT_SEED = L.BOOT_SEED
CI = L.CI_PCT
N_SEEDS = 10


def main():
    # ---------- inputs ----------
    d4, labels = L.load_run_files(os.path.join(PRED4, "baseline_s*.npz"), expect=N_SEEDS)
    da, labelsa = L.load_run_files(os.path.join(PREDA, "baseline_s*.npz"), expect=N_SEEDS)

    probs_mean = L.ensemble_probabilities(d4)
    probsa_mean = L.ensemble_probabilities(da)

    # ---------- A. four-class, per seed ----------
    per_seed_macro = [L.macro_ovr_auc(d["probs"].astype("float64"), d["labels"]) for d in d4]
    per_seed_acc = [L.accuracy(d["labels"], d["probs"].argmax(1)) for d in d4]
    per_seed_restricted = [L.restricted_pair_auc(d["probs"].astype("float64"), d["labels"])[0]
                           for d in d4]

    # ---------- B. four-class ensemble + CI ----------
    ens_macro = L.macro_ovr_auc(probs_mean, labels)
    ens_acc = L.accuracy(labels, probs_mean.argmax(1))
    ens_w = L.weighted_ovr_auc(probs_mean, labels)
    ens_pc = L.per_class_ovr_auc(probs_mean, labels)

    macro_ci = L.boot_percentile_ci(lambda i: L.macro_ovr_auc(probs_mean[i], labels[i]),
                                    len(labels))
    acc_ci = L.boot_percentile_ci(lambda i: L.accuracy(labels[i], probs_mean[i].argmax(1)),
                                  len(labels))
    pc_ci = {}
    for c in range(NUM):
        pc_ci[CLASS_NAMES[c]] = L.boot_auc_ci((labels == c).astype(int), probs_mean[:, c])

    # ---------- C. restricted active-vs-latent ----------
    restricted = {}
    for mode in L.RESTRICTED_MODES:
        auc, y_r, s_r = L.restricted_pair_auc(probs_mean, labels, mode)
        lo, hi, _ = L.boot_auc_ci(y_r, s_r)
        d_lo, d_hi, _ = L.boot_auc_minus_chance_ci(y_r, s_r)
        restricted[mode] = {"auc_ensemble": auc, "ci95": [lo, hi],
                            "auc_minus_chance_ci95": [d_lo, d_hi], "n": int(len(y_r))}

    # ---------- D. decision rule: argmax on the ensemble ----------
    cm, recall, _ = L.confusion_argmax(labels, probs_mean, NUM)
    active_as_latent = int(cm[2][3])
    latent_as_active = int(cm[3][2])

    # ---------- E. dedicated active-vs-latent model ----------
    per_seed_a = [L.binary_auc(d["labels"], d["probs"].astype("float64")[:, 1]) for d in da]
    y_a = (labelsa == 1).astype(int)
    s_a = probsa_mean[:, 1]
    ens_a = L.binary_auc(y_a, s_a)
    a_ci = L.boot_auc_ci(y_a, s_a)
    a_d_ci = L.boot_auc_minus_chance_ci(y_a, s_a)
    cm_a, _, _ = L.confusion_argmax(labelsa, probsa_mean, 2)

    # ---------- payload ----------
    sd_macro, sd_acc, sd_restr = (L.mean_sd(per_seed_macro),
                                  L.mean_sd(per_seed_acc),
                                  L.mean_sd(per_seed_restricted))
    sd_a = L.mean_sd(per_seed_a)

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
                "macro_auc": list(sd_macro),
                "accuracy": list(sd_acc),
                "restricted_active_vs_latent_auc": list(sd_restr),
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
            "per_seed_mean_sd": list(sd_a),
            "per_seed_min_max": list(L.min_max(per_seed_a)),
            "ensemble_auc": ens_a,
            "ensemble_auc_ci95": list(a_ci[:2]),
            "ensemble_auc_minus_chance_ci95": list(a_d_ci[:2]),
            "ensemble_confusion_argmax": cm_a,
        },
    }
    L.write_json(os.path.join(OUT, "paper2_stats.json"), payload)

    # ---------- text artefact ----------
    S = []
    S.append("PAPER 2 STATISTICS ARTIFACT 2026-09-13  (stats_paper2.py)")
    S.append("bootstrap: %d iters, percentile, RandomState(%d), unit=image, 95%% CI"
             % (BOOT, BOOT_SEED))
    S.append("restricted a-vs-l score: p_lat/(p_lat+p_act);  decision rule: argmax (ensemble)")
    S.append("")
    S.append("=== FOUR-CLASS TBX11K (n=%d; %s) ===" % (len(labels),
             ", ".join("%s=%d" % (CLASS_NAMES[c], int((labels == c).sum())) for c in range(NUM))))
    S.append("per-seed macro-AUC      : %s" % [round(v, 4) for v in per_seed_macro])
    S.append("  mean +/- SD           : %.4f +/- %.4f" % tuple(sd_macro))
    S.append("per-seed accuracy       : %s" % [round(v, 4) for v in per_seed_acc])
    S.append("  mean +/- SD           : %.4f +/- %.4f" % tuple(sd_acc))
    S.append("per-seed restricted AUC : %s" % [round(v, 4) for v in per_seed_restricted])
    S.append("  mean +/- SD           : %.4f +/- %.4f" % tuple(sd_restr))
    S.append("")
    S.append("ensemble macro-AUC      : %.4f   95%% CI [%.4f, %.4f]"
             % (ens_macro, macro_ci[0], macro_ci[1]))
    S.append("ensemble accuracy       : %.4f   95%% CI [%.4f, %.4f]"
             % (ens_acc, acc_ci[0], acc_ci[1]))
    S.append("ensemble weighted AUC   : %.4f" % ens_w)
    for c in range(NUM):
        S.append("  per-class %-15s : %.4f   95%% CI [%.4f, %.4f]"
                 % (CLASS_NAMES[c], ens_pc[CLASS_NAMES[c]],
                    pc_ci[CLASS_NAMES[c]][0], pc_ci[CLASS_NAMES[c]][1]))
    S.append("")
    for mode in L.RESTRICTED_MODES:
        r = restricted[mode]
        tag = "  <-- PRIMARY (used in the manuscript)" if mode == "renorm" else ""
        S.append("restricted AUC [%-11s]: %.4f   95%% CI [%.4f, %.4f]   AUC-0.5 CI [%+.4f, %+.4f]  n=%d%s"
                 % (mode, r["auc_ensemble"], r["ci95"][0], r["ci95"][1],
                    r["auc_minus_chance_ci95"][0], r["auc_minus_chance_ci95"][1], r["n"], tag))
    S.append("")
    S.append("ensemble confusion (argmax rows=true cols=pred) order=%s" % CLASS_NAMES)
    for row in cm:
        S.append("   %s" % row)
    S.append("recall: %s" % {k: round(v, 4) for k, v in recall.items()})
    S.append("latent->active = %d/%d ; active->latent = %d/%d"
             % (latent_as_active, int(sum(cm[3])), active_as_latent, int(sum(cm[2]))))
    S.append("")
    S.append("=== DEDICATED ACTIVE-VS-LATENT MODEL (n=%d; active=%d latent=%d) ==="
             % (len(labelsa), int((labelsa == 0).sum()), int((labelsa == 1).sum())))
    S.append("per-seed AUC            : %s" % [round(v, 4) for v in per_seed_a])
    S.append("  mean +/- SD           : %.4f +/- %.4f   (min %.4f max %.4f)"
             % (sd_a[0], sd_a[1], L.min_max(per_seed_a)[0], L.min_max(per_seed_a)[1]))
    S.append("ensemble AUC            : %.4f   95%% CI [%.4f, %.4f]   AUC-0.5 CI [%+.4f, %+.4f]"
             % (ens_a, a_ci[0], a_ci[1], a_d_ci[0], a_d_ci[1]))
    S.append("ensemble confusion (argmax) rows=true cols=pred order=[active,latent]: %s" % cm_a)
    L.write_lines(os.path.join(OUT, "paper2_stats.txt"), S)

    print("\n".join(S))
    print("\n[OK] wrote %s/paper2_stats.json/.txt" % OUT)


if __name__ == "__main__":
    main()

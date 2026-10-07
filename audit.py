#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shape-free reference audit.

Quantifies how much of a same-cohort chest-radiograph classification score is
attainable without resolving pulmonary pathology, by comparing a model's AUC with
a shape-free reference classifier trained on the identical splits.

Definition
----------
    Delta = AUC_model - AUC_reference

on the same test split, with a paired bootstrap interval: the test set is
resampled with replacement B times and both AUCs are recomputed on the *same*
resampled indices, which preserves the pairing between model and reference.

Reading a cohort
----------------
    interval wholly below 0.05   -> appearance-dominated
    interval straddles zero      -> inconclusive
    otherwise                    -> signal beyond appearance

Failure mode: if the reference exceeds the model (Delta < 0), the reading is not
that the reference is a ceiling but that the model configuration has not
exploited even the appearance-level signal on that cohort.

Reference family
----------------
Features (as released in features/<cohort>_{train,test}.npz):
    thumb16      16x16 grayscale thumbnail, 256 dimensions, values in [0,1]
    stats5       five global intensity statistics (mean, SD, p5, p95, median)
Feature sets: thumb16_only, thumb16_plus_stats5 (concatenation).
Learners (fixed hyper-parameters, random_state = 0 where stochastic):
    logreg            LogisticRegression(class_weight="balanced", max_iter=2000)
    randomforest      RandomForestClassifier(n_estimators=500,
                        class_weight="balanced_subsample", random_state=0)
    gradientboosting  HistGradientBoostingClassifier(max_iter=200, random_state=0)
    knn15             Pipeline(StandardScaler, KNeighborsClassifier(15, weights="distance"))

Usage
-----
Single cohort, released layout::

    python audit.py --cohort TBX11K --resamples 10000 --out audit_TBX11K.json

All four cohorts::

    python audit.py --all --out audit_all.json

Explicit paths (any cohort, any model scores)::

    python audit.py \
        --features  features/Shenzhen_train.npz features/Shenzhen_test.npz \
        --model-preds 'predictions/cross_cohort/resnet50_full/*_Shenzhen_to_Shenzhen.npz' \
        --resamples 10000 --out audit_Shenzhen.json

Requirements: numpy, scikit-learn. No other dependencies.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COHORTS = ("TBX11K", "Shenzhen", "Montgomery", "Qatar")
FEATURE_SETS = ("thumb16_only", "thumb16_plus_stats5")
LEARNERS = ("logreg", "randomforest", "gradientboosting", "knn15")
DEFAULT_MODEL_DIR = {
    "TBX11K": "predictions/cross_cohort/resnet50_full",
    "Shenzhen": "predictions/cross_cohort/resnet50_full",
    "Montgomery": "predictions/cross_cohort/resnet50_full",
    "Qatar": "predictions/cross_cohort/dedup_qatar",
}
APPEARANCE_THRESHOLD = 0.05


def fast_auc(y, s):
    """Rank-based AUC (Mann-Whitney U form; average ranks for ties)."""
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks for tied scores
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def make_learner(name):
    if name == "logreg":
        return LogisticRegression(class_weight="balanced", max_iter=2000)
    if name == "randomforest":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                      random_state=0, n_jobs=-1)
    if name == "gradientboosting":
        return HistGradientBoostingClassifier(max_iter=200, random_state=0)
    if name == "knn15":
        return Pipeline([("scaler", StandardScaler()),
                         ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1))])
    raise ValueError(f"unknown learner: {name}")


def feature_matrix(features, feature_set):
    if feature_set == "thumb16_only":
        return features["thumb16"]
    if feature_set == "thumb16_plus_stats5":
        return np.hstack([features["thumb16"], features["stats5"]])
    raise ValueError(f"unknown feature set: {feature_set}")


def reference_aucs(train_npz, test_npz, learners=LEARNERS, feature_sets=FEATURE_SETS):
    """AUC of every feature-set x learner cell on the test split."""
    tr = np.load(train_npz, allow_pickle=True)
    te = np.load(test_npz, allow_pickle=True)
    y_tr, y_te = tr["labels"], te["labels"]
    out = {}
    for fs in feature_sets:
        X_tr, X_te = feature_matrix(tr, fs), feature_matrix(te, fs)
        out[fs] = {}
        for lname in learners:
            clf = make_learner(lname).fit(X_tr, y_tr)
            out[fs][lname] = fast_auc(y_te, clf.predict_proba(X_te)[:, 1])
    return out, te["labels"]


def model_scores(pred_globs):
    """Per-seed P(positive) columns from one or more prediction .npz files."""
    files = []
    for pat in pred_globs:
        files.extend(sorted(glob.glob(pat)))
    if not files:
        raise SystemExit(f"no prediction files matched: {pred_globs}")
    cols, labels = [], None
    for f in files:
        z = np.load(f, allow_pickle=True)
        cols.append(z["probs"][:, 1])
        labels = z["labels"]
    return np.vstack(cols).T, labels, files


def paired_bootstrap(y, model_prob, ref_prob, resamples, seed, model_per_seed=None):
    """Paired bootstrap of Delta = AUC_model - AUC_reference."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d_primary = []
    skipped = 0
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        if model_per_seed is None:
            a_model = fast_auc(yy, model_prob[idx])
        else:
            a_model = float(np.mean([fast_auc(yy, model_per_seed[idx, k])
                                     for k in range(model_per_seed.shape[1])]))
        d_primary.append(a_model - fast_auc(yy, ref_prob[idx]))
    d = np.asarray(d_primary)
    return {"mean": float(d.mean()),
            "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
            "p_lt_0": float((d < 0).mean()),
            "n_valid": int(len(d)), "n_skipped": int(skipped)}


def read_cohort(reading, lo, hi):
    if hi < APPEARANCE_THRESHOLD:
        return "appearance-dominated"
    if lo <= 0.0 <= hi:
        return "inconclusive"
    return "signal beyond appearance"


def audit_one(cohort, train_npz, test_npz, pred_globs, resamples, seed, model_per_seed_override=None):
    aucs, y_te = reference_aucs(train_npz, test_npz)
    if model_per_seed_override is not None:
        P = model_per_seed_override
        y_model = y_te
        files = []
    else:
        P, y_model, files = model_scores(pred_globs)
    if not np.array_equal(np.asarray(y_model), np.asarray(y_te)):
        raise SystemExit(f"{cohort}: model and reference test labels differ; cannot pair")
    per_seed = [fast_auc(y_te, P[:, k]) for k in range(P.shape[1])]
    a_model = float(np.mean(per_seed))
    best_fs, best_learner = max(((fs, ln) for fs in aucs for ln in aucs[fs]),
                                key=lambda k: aucs[k[0]][k[1]])
    tr = np.load(train_npz, allow_pickle=True)
    te = np.load(test_npz, allow_pickle=True)
    ref_primary = make_learner("logreg").fit(feature_matrix(tr, "thumb16_plus_stats5"),
                                             tr["labels"]).predict_proba(
        feature_matrix(te, "thumb16_plus_stats5"))[:, 1]
    ref_envelope = make_learner(best_learner).fit(feature_matrix(tr, best_fs),
                                                  tr["labels"]).predict_proba(
        feature_matrix(te, best_fs))[:, 1]
    boots_primary = paired_bootstrap(y_te, P, ref_primary, resamples, seed, model_per_seed=P)
    boots_envelope = paired_bootstrap(y_te, P, ref_envelope, resamples, seed, model_per_seed=P)
    res = {
        "cohort": cohort,
        "n_test": int(len(y_te)), "n_positive": int(y_te.sum()),
        "model": {"files": [f.split("/")[-1] for f in files], "n_seeds": int(P.shape[1]),
                  "per_seed_auc": [round(float(a), 6) for a in per_seed],
                  "mean_auc": round(a_model, 6)},
        "reference_auc": {fs: {ln: round(v, 6) for ln, v in d.items()} for fs, d in aucs.items()},
        "primary_reference": {
            "feature_set": "thumb16_plus_stats5", "learner": "logreg",
            "auc": round(fast_auc(y_te, ref_primary), 6),
            "delta": round(a_model - fast_auc(y_te, ref_primary), 6), **boots_primary},
        "strongest_reference": {
            "feature_set": best_fs, "learner": best_learner,
            "auc": round(fast_auc(y_te, ref_envelope), 6),
            "delta": round(a_model - fast_auc(y_te, ref_envelope), 6), **boots_envelope},
    }
    res["primary_reference"]["reading"] = read_cohort(
        None, res["primary_reference"]["lo"], res["primary_reference"]["hi"])
    res["strongest_reference"]["reading"] = read_cohort(
        None, res["strongest_reference"]["lo"], res["strongest_reference"]["hi"])
    res["delta_range_over_cells"] = [
        round(a_model - aucs[fs][ln], 6) for fs in aucs for ln in aucs[fs]]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Shape-free reference audit (paired bootstrap of Delta).")
    ap.add_argument("--cohort", choices=COHORTS)
    ap.add_argument("--all", action="store_true", help="run the four released cohorts")
    ap.add_argument("--features", nargs=2, metavar=("TRAIN_NPZ", "TEST_NPZ"))
    ap.add_argument("--model-preds", nargs="+", metavar="GLOB")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--out", default="audit_result.json")
    a = ap.parse_args(argv)

    # self-check: the rank-based AUC must agree with scikit-learn's implementation
    rng = np.random.default_rng(0)
    y0 = (rng.random(500) < 0.4).astype(int)
    s0 = rng.random(500) + 0.25 * y0
    assert abs(fast_auc(y0, s0) - roc_auc_score(y0, s0)) < 1e-12, "fast_auc disagrees with roc_auc_score"

    results = {}
    if a.all or (a.cohort and not a.features):
        cohorts = COHORTS if a.all else (a.cohort,)
        for c in cohorts:
            tr = f"features/{c}_train.npz"
            te = f"features/{c}_test.npz"
            preds = [f"{DEFAULT_MODEL_DIR[c]}/*_{c}_to_{c}.npz"]
            print(f"[audit] {c}: {tr} | {te} | {preds[0]}", flush=True)
            results[c] = audit_one(c, tr, te, preds, a.resamples, a.seed)
    else:
        if not (a.cohort and a.features and a.model_preds):
            ap.error("give --cohort with --features and --model-preds, or use --all")
        results[a.cohort] = audit_one(a.cohort, a.features[0], a.features[1],
                                      a.model_preds, a.resamples, a.seed)

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"definition": "Delta = AUC_model - AUC_reference, paired bootstrap",
                   "resamples": a.resamples, "seed": a.seed,
                   "threshold": APPEARANCE_THRESHOLD, "results": results},
                  fh, indent=2)

    for c, r in results.items():
        p, s = r["primary_reference"], r["strongest_reference"]
        print(f"\n{c}  n={r['n_test']} (pos {r['n_positive']})  model AUC {r['model']['mean_auc']:.4f} "
              f"({r['model']['n_seeds']} seeds)")
        print(f"  primary  (logreg, 16x16+5): AUC {p['auc']:.4f}  Delta {p['delta']:+.4f} "
              f"[{p['lo']:+.4f}, {p['hi']:+.4f}]  P(Delta<0)={p['p_lt_0']:.3f}  -> {p['reading']}")
        print(f"  strongest({s['learner']}, {s['feature_set']}): AUC {s['auc']:.4f}  "
              f"Delta {s['delta']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}]  -> {s['reading']}")
        print(f"  Delta over the 8 released cells: {min(r['delta_range_over_cells']):+.4f} "
              f"to {max(r['delta_range_over_cells']):+.4f}")
    print(f"\n[audit] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

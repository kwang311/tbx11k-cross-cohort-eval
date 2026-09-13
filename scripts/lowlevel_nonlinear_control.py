# -*- coding: utf-8 -*-
"""Add-on control 1: non-linear shape-free reference (reviewer request).

Why this file exists
--------------------
Section 5.4 of the Paper 2 draft states that whatever the CNNs add over the low-level
reference (0.023 AUC in TBX11K, 0.024 in Qatar) is "the only part of their performance
that requires finer structure". R1/R3 pointed out that a single un-tuned linear model
cannot support that strength: a non-linear classifier on the same low-level features
might close the gap. This script therefore repeats the reference with the SAME features,
SAME splits and SAME preprocessing as lowlevel_same_split.py, adding RandomForest,
gradient boosting and kNN next to logistic regression.

Pipeline (identical to lowlevel_same_split.py otherwise):
  * splits from train_cross_cohort.build_splits (TBX11K official train/val; others 80/20
    stratified, random_state=42), TBX_GRAY=1 preprocessing;
  * features: 16x16 grayscale thumbnail (256 dims) and/or five global intensity statistics;
  * classifiers: LogisticRegression (reference, must reproduce the dumped values),
    RandomForest(500 trees), HistGradientBoosting(200 iters), kNN(k=15, distance-weighted,
    features standardised inside a Pipeline fitted on the training split only).
  * AUC on the identical test split; random_state fixed at 0 for all stochastic learners.

Outputs: output_cross/lowlevel_nonlinear_control_20260913.json / .txt
Usage:   set TBX_GRAY=1 && python lowlevel_nonlinear_control.py
"""
import os
import json
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

assert os.environ.get("TBX_GRAY", "") == "1", "set TBX_GRAY=1 (main matrix preprocessing)"

import train_cross_cohort as T  # noqa: E402
from data_tbx11k import build_ram_cache, CACHE_SIZE, GRAYSCALE  # noqa: E402

OUT_DIR = T.OUT_DIR
COHORTS4 = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]
REF_JSON = os.path.join(OUT_DIR, "lowlevel_same_split_20260911.json")
TOL = 0.002          # fail-fast tolerance against the dumped LR reference values


def gray_thumb(images, size=16):
    out = np.zeros((len(images), size * size), dtype=np.float32)
    for i in range(len(images)):
        g = images[i].astype(np.float32).mean(axis=2)
        t = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((size, size), Image.BILINEAR),
                       dtype=np.float32)
        out[i] = t.ravel() / 255.0
    return out


def stats5(images):
    out = np.zeros((len(images), 5), dtype=np.float32)
    for i in range(len(images)):
        g = images[i].astype(np.float32).mean(axis=2)
        out[i] = [g.mean() / 255.0, g.std() / 255.0,
                  np.percentile(g, 5) / 255.0, np.percentile(g, 95) / 255.0,
                  np.median(g) / 255.0]
    return out


def classifiers():
    return {
        "logreg": lambda: LogisticRegression(class_weight="balanced", max_iter=2000),
        "randomforest": lambda: RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", random_state=0, n_jobs=-1),
        "gradientboosting": lambda: HistGradientBoostingClassifier(
            max_iter=200, random_state=0),
        "knn15": lambda: Pipeline([("scaler", StandardScaler()),
                                   ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance",
                                                                n_jobs=-1))]),
    }


def auc_of(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def main():
    splits = T.build_splits(quick=False)
    assert set(splits) == set(COHORTS4)
    ref = json.load(open(REF_JSON, encoding="utf-8"))["auc"]

    res, feats = {}, {}
    print("=== non-linear shape-free control (same splits/features as lowlevel_same_split.py) "
          f"(CACHE_SIZE={CACHE_SIZE}, GRAYSCALE={GRAYSCALE}) ===", flush=True)
    for c in COHORTS4:
        tr_img, tr_y = build_ram_cache(*splits[c]["train"])
        te_img, te_y = build_ram_cache(*splits[c]["test"])
        tr_y, te_y = np.asarray(tr_y), np.asarray(te_y)
        assert len(np.unique(tr_y)) == 2 and len(np.unique(te_y)) == 2, f"{c}: single-class split"
        assert len(splits[c]["train"][0]) - len(tr_img) <= 5, f"{c}: unreadable train images"
        assert len(splits[c]["test"][0]) - len(te_img) <= 5, f"{c}: unreadable test images"

        t_tr, t_te = gray_thumb(tr_img), gray_thumb(te_img)
        s_tr, s_te = stats5(tr_img), stats5(te_img)
        X = {"thumb16_only": (t_tr, t_te),
             "thumb16_plus_stats5": (np.hstack([t_tr, s_tr]), np.hstack([t_te, s_te]))}

        row = {}
        print(f"\n{c}: train={len(tr_img)} test={len(te_img)}", flush=True)
        for fset, (Xtr, Xte) in X.items():
            row[fset] = {}
            for name, mk in classifiers().items():
                a = auc_of(mk(), Xtr, tr_y, Xte, te_y)
                row[fset][name] = a
                print(f"    {fset:21s} {name:17s} AUC={a:.4f}", flush=True)

        # fail-fast: the LR reference must reproduce the dumped artifact
        for fset in X:
            ref_v = ref[c][fset]
            got = row[fset]["logreg"]
            assert abs(got - ref_v) <= TOL, \
                f"{c}/{fset}: LR {got:.4f} vs dumped {ref_v:.4f} (tolerance {TOL})"
        print(f"    [check] LR reproduces lowlevel_same_split_20260911.json "
              f"(tol {TOL}) OK", flush=True)
        res[c] = row
        feats[c] = {"n_train": int(len(tr_img)), "n_test": int(len(te_img))}

    payload = {
        "note": "non-linear shape-free control; same splits (train_cross_cohort.build_splits) and "
                "same features as lowlevel_same_split.py; TBX_GRAY=1; LogisticRegression and kNN "
                "carry class-balancing options, RandomForest uses balanced_subsample, "
                "HistGradientBoosting uses defaults; kNN features standardised inside the pipeline "
                "(fitted on the training split only); random_state=0 for all stochastic learners",
        "scripts": {"control": "lowlevel_nonlinear_control.py",
                    "reference_values": "lowlevel_same_split_20260911.json"},
        "splits": feats, "auc": res,
    }
    for ext in ("json", "txt"):
        with open(os.path.join(OUT_DIR, f"lowlevel_nonlinear_control_20260913.{ext}"), "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== summary: CNN same-cohort gain over the best shape-free model ===", flush=True)
    for c in COHORTS4:
        best = max(max(v.values()) for v in res[c].values())
        print(f"  {c:11s} LR(thumb+stats5)={res[c]['thumb16_plus_stats5']['logreg']:.4f}  "
              f"best shape-free={best:.4f}", flush=True)
    print(f"\n[OK] wrote {OUT_DIR}/lowlevel_nonlinear_control_20260913.json/.txt", flush=True)


if __name__ == "__main__":
    main()

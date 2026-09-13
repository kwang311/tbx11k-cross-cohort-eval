# -*- coding: utf-8 -*-
"""Low-level reference classifier, SAME splits + SAME preprocessing as the cross-cohort matrix.

Why this file exists
--------------------
The earlier low-level reference numbers cited in the Paper 2 draft
(TBX11K 0.976 / Qatar 0.965 / Shenzhen 0.832 / Montgomery 0.755) were produced by
lowlevel_baseline.py, which samples <=700 images per class and uses its OWN 70/30
split -- NOT the CNN evaluation splits that Methods claims ("exactly the same
train/test splits"). Its stdout was never written to a results file, and the
diagnostic outputs in output_cross/ were overwritten by the later 3-cohort rerun.
So those numbers have no artifact and do not match the stated protocol.

This script recomputes the low-level reference with:
  * the exact splits from train_cross_cohort.build_splits()  (TBX11K official train/val;
    other cohorts 80/20 stratified, random_state=42),
  * TBX_GRAY=1 preprocessing (same as the paper's main matrix),
  * the same feature extractor as train_cross_cohort.lowlevel_auc (16x16 thumbnail + mean/std).
It also reports feature ablations (thumbnail only / global stats only / thumbnail border
frame / thumbnail center) so that the Qatar confound numbers have a written definition.

Outputs: output_cross/lowlevel_same_split_20260911.json and .txt (ASCII only, GBK-safe).
Usage:  set TBX_GRAY=1 && python lowlevel_same_split.py
"""
import os
import sys
import json
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

assert os.environ.get("TBX_GRAY", "") == "1", "set TBX_GRAY=1 (main matrix preprocessing)"

import train_cross_cohort as T  # noqa: E402  (import after env check)
from data_tbx11k import build_ram_cache, CACHE_SIZE, GRAYSCALE  # noqa: E402

OUT_DIR = T.OUT_DIR
COHORTS4 = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]
rng = np.random.RandomState(0)


def gray_thumb(images, size=16):
    """(N,H,W,3) uint8 -> (N, size*size) float thumbnail of the grayscale image."""
    out = np.zeros((len(images), size * size), dtype=np.float32)
    for i in range(len(images)):
        g = images[i].astype(np.float32).mean(axis=2)
        t = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((size, size), Image.BILINEAR),
                       dtype=np.float32)
        out[i] = t.ravel() / 255.0
    return out


def stats5(images):
    """Five global intensity statistics per image (Methods: mean, SD, p5, p95, median)."""
    out = np.zeros((len(images), 5), dtype=np.float32)
    for i in range(len(images)):
        g = images[i].astype(np.float32).mean(axis=2)
        out[i] = [g.mean() / 255.0, g.std() / 255.0,
                  np.percentile(g, 5) / 255.0, np.percentile(g, 95) / 255.0,
                  np.median(g) / 255.0]
    return out


def auc_of(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def main():
    splits = T.build_splits(quick=False)
    assert set(splits) == set(COHORTS4), f"splits missing cohorts: {set(COHORTS4) - set(splits)}"

    res = {}
    feats = {}
    print("=== low-level reference on the SAME splits as the CNN matrix "
          f"(CACHE_SIZE={CACHE_SIZE}, GRAYSCALE={GRAYSCALE}) ===", flush=True)

    for c in COHORTS4:
        tr_img, tr_y = build_ram_cache(*splits[c]["train"])
        te_img, te_y = build_ram_cache(*splits[c]["test"])
        tr_y = np.asarray(tr_y)
        te_y = np.asarray(te_y)
        n_tr_dropped = len(splits[c]["train"][0]) - len(tr_img)
        n_te_dropped = len(splits[c]["test"][0]) - len(te_img)
        assert len(np.unique(tr_y)) == 2 and len(np.unique(te_y)) == 2, f"{c}: single-class split"
        assert n_tr_dropped <= 5 and n_te_dropped <= 5, f"{c}: too many unreadable images"

        t_tr = gray_thumb(tr_img)
        t_te = gray_thumb(te_img)
        st_tr = stats5(tr_img)
        st_te = stats5(te_img)

        row = {}
        row["thumb16_plus_stats5"] = auc_of(np.hstack([t_tr, st_tr]), tr_y,
                                            np.hstack([t_te, st_te]), te_y)
        row["thumb16_only"] = auc_of(t_tr, tr_y, t_te, te_y)
        row["global_stats5_only"] = auc_of(st_tr, tr_y, st_te, te_y)

        # thumbnail sub-regions (definition stated explicitly for the paper)
        idx = np.arange(256).reshape(16, 16)
        border = np.concatenate([idx[:2].ravel(), idx[-2:].ravel(), idx[2:-2, :2].ravel(),
                                 idx[2:-2, -2:].ravel()])          # outer 2-pixel frame
        center = idx[4:12, 4:12].ravel()                            # central 8x8 = 25% of field
        row["thumb16_border_frame"] = auc_of(t_tr[:, border], tr_y, t_te[:, border], te_y)
        row["thumb16_center_8x8"] = auc_of(t_tr[:, center], tr_y, t_te[:, center], te_y)

        res[c] = row
        feats[c] = {"n_train": int(len(tr_img)), "n_test": int(len(te_img)),
                    "train_pos": int((tr_y == 1).sum()), "test_pos": int((te_y == 1).sum()),
                    "unreadable_train": int(n_tr_dropped), "unreadable_test": int(n_te_dropped)}
        print(f"\n{c}: train={len(tr_img)} (TB={int((tr_y==1).sum())})  "
              f"test={len(te_img)} (TB={int((te_y==1).sum())})  drop={n_tr_dropped}/{n_te_dropped}",
              flush=True)
        for k, v in row.items():
            print(f"    {k:22s} AUC={v:.4f}", flush=True)

    payload = {"note": "same splits + TBX_GRAY=1 as the cross-cohort matrix; "
                       "features = grayscale 16x16 thumbnail (resized from the 256px cache) "
                       "and/or five global intensity statistics (mean, SD, p5, p95, median); "
                       "region ablations = outer 2-pixel frame of the thumbnail (image periphery) "
                       "and its central 8x8 (25% of the field); LogisticRegression(class_weight=balanced)",
               "scripts": {"lowlevel": "lowlevel_same_split.py",
                           "splits": "train_cross_cohort.build_splits"},
               "splits": feats, "auc": res}
    with open(os.path.join(OUT_DIR, "lowlevel_same_split_20260911.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "lowlevel_same_split_20260911.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n[OK] wrote {OUT_DIR}/lowlevel_same_split_20260911.json/.txt", flush=True)


if __name__ == "__main__":
    main()

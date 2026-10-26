# -*- coding: utf-8 -*-
"""Cross-cohort binary TB screening: train on one cohort -> test on several (angle 1).

- Task: normal(0) vs TB(1)
- Cohorts: TBX11K / Shenzhen / Montgomery / Qatar
- Split: official train/val for TBX11K; stratified 80/20 for the rest (fixed seed)
- Training: ResNet-50 baseline (frozen early layers, the project-standard recipe), multiple seeds
- Output: output_cross/matrix_{seed}.json (source x target AUC) + summary

Usage:
  python train_cross_cohort.py --quick           # smoke test (subsampling + 1 seed + 1 epoch)
  python train_cross_cohort.py                   # full run (5 seeds x 15 epochs)
  python train_cross_cohort.py --seeds 42,43     # explicit seeds
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data_tbx11k import build_ram_cache, RamDataset
from data_tbcohort import load_cohort, load_tbx_split
from models import Chest9Classifier

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
BASE_DIR = str(P.ROOT)
OUT_DIR = str(P.RESULTS / "cross")
# Qatar is excluded: image formats differ between classes (Normal is all RGB / nearly half of TB is grayscale) and provenance is mixed (low-level baseline AUC 0.97)
COHORTS = ["TBX11K", "Shenzhen", "Montgomery"]


def build_splits(quick=False):
    """Return {cohort: {"train": (paths,y), "test": (paths,y)}}."""
    splits = {}
    s = load_tbx_split()
    splits["TBX11K"] = {"train": s["train"], "test": s["val"]}
    for c in ["Shenzhen", "Montgomery", "Qatar"]:
        p, y = load_cohort(c)
        p, y = np.array(p), np.array(y)
        tr_p, te_p, tr_y, te_y = train_test_split(
            p, y, test_size=0.2, stratify=y, random_state=42)
        splits[c] = {"train": (list(tr_p), list(tr_y)), "test": (list(te_p), list(te_y))}
    if quick:
        rng = np.random.RandomState(0)
        for c in splits:
            for k in ("train", "test"):
                p, y = splits[c][k]
                p, y = np.array(p), np.array(y)
                n = min(len(p), 80)
                idx = rng.choice(len(p), n, replace=False)
                splits[c][k] = (list(p[idx]), list(y[idx]))
    # --- Guard assertions (fail-fast) ---
    for c in splits:
        trp, trY = np.array(splits[c]["train"][0]), np.array(splits[c]["train"][1])
        tep, teY = np.array(splits[c]["test"][0]), np.array(splits[c]["test"][1])
        assert set(np.unique(trY)) <= {0, 1} and set(np.unique(teY)) <= {0, 1}, f"{c} labels are not 0/1"
        assert len(set(trp) & set(tep)) == 0, f"{c} train/test paths overlap (leakage!)"
        assert len(trY) == len(trp) and len(teY) == len(tep), f"{c} path/label counts do not match"
    print("  [check] labels in {0,1} for every cohort, no train/test overlap, path/label counts consistent OK")
    return splits


def train_one(train_cache, seed, epochs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    imgs, lbls = train_cache
    ds = RamDataset(imgs, lbls, train=True)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    counts = np.bincount(lbls, minlength=2).astype(np.float32)
    w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    model = Chest9Classifier("baseline", num_classes=2).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
    model.train()
    for ep in range(epochs):
        for im, lb in dl:
            im, lb = im.to(device), lb.to(device)
            opt.zero_grad()
            loss = crit(model(im), lb)
            if torch.isnan(loss):
                continue
            loss.backward()
            opt.step()
        sch.step()
    return model


@torch.no_grad()
def predict(model, test_cache, device):
    model.eval()
    imgs, lbls = test_cache
    ds = RamDataset(imgs, lbls, train=False)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    probs = []
    for im, lb in dl:
        im = im.to(device)
        probs.append(torch.softmax(model(im), dim=1).cpu().numpy())
    return np.concatenate(probs), lbls


def lowlevel_auc(train_cache, test_cache):
    """16x16 thumbnails + mean/SD -> logistic-regression AUC (trivial low-level baseline control)."""
    from PIL import Image
    from sklearn.linear_model import LogisticRegression

    def feats(cache):
        imgs, lbls = cache
        X = np.zeros((len(imgs), 16 * 16 + 2), dtype=np.float32)
        for i in range(len(imgs)):
            g = imgs[i].astype(np.float32).mean(axis=2)
            small = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((16, 16), Image.BILINEAR),
                               dtype=np.float32).ravel() / 255.0
            X[i] = np.concatenate([small, [g.mean() / 255.0, g.std() / 255.0]])
        return X, np.asarray(lbls)

    Xtr, ytr = feats(train_cache)
    Xte, yte = feats(test_cache)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.quick:
        args.epochs = 1
        args.seeds = "42"
    seeds = [int(s) for s in args.seeds.split(",")]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    splits = build_splits(quick=args.quick)
    # Cache
    cache = {}
    for c in COHORTS:
        t0 = time.time()
        cache[c] = {
            "train": build_ram_cache(*splits[c]["train"]),
            "test": build_ram_cache(*splits[c]["test"]),
        }
        print(f"  cached {c}: train={len(cache[c]['train'][0])} test={len(cache[c]['test'][0])} "
              f"({time.time()-t0:.1f}s)", flush=True)

    results = {}  # results[source][seed][target] = auc
    for src in COHORTS:
        print(f"\n===== Training source cohort {src} =====")
        tr_cache = cache[src]["train"]
        for seed in seeds:
            model = train_one(tr_cache, seed, args.epochs, device)
            row = {}
            for tgt in COHORTS:
                probs, labels = predict(model, cache[tgt]["test"], device)
                auc = roc_auc_score(labels, probs[:, 1])
                row[tgt] = float(auc)
            results.setdefault(src, {})[str(seed)] = row
            print(f"  seed {seed}: " + "  ".join(f"{t}={row[t]:.3f}" for t in COHORTS), flush=True)
            with open(os.path.join(OUT_DIR, "matrix_raw.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary mean±std
    print("\n===== Cross-cohort AUC matrix (train rows x test columns, mean±std over seeds) =====")
    header = "src\\tgt      " + "".join(f"{t[:10]:>14s}" for t in COHORTS)
    print(header)
    summary = {}
    for src in COHORTS:
        cells = []
        for tgt in COHORTS:
            vals = [results[src][str(s)][tgt] for s in seeds]
            summary.setdefault(src, {})[tgt] = {"mean": float(np.mean(vals)),
                                                "std": float(np.std(vals, ddof=1))}
            cells.append(f"{np.mean(vals):.3f}±{np.std(vals,ddof=1):.3f}")
        print(f"{src:11s} " + "".join(f"{c:>14s}" for c in cells))
    with open(os.path.join(OUT_DIR, "matrix_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Low-level baseline (trivial control)
    print("\n===== Low-level baseline (16x16 + stats -> LR, trivial control) =====")
    ll = {}
    for c in COHORTS:
        a = lowlevel_auc(cache[c]["train"], cache[c]["test"])
        ll[c] = a
        print(f"  {c:11s} low-level baseline AUC={a:.3f}   in-cohort CNN AUC={summary[c][c]['mean']:.3f}   "
              f"CNN gain={summary[c][c]['mean'] - a:+.3f}")
    with open(os.path.join(OUT_DIR, "lowlevel.json"), "w", encoding="utf-8") as f:
        json.dump(ll, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {OUT_DIR}/matrix_raw.json, matrix_summary.json, lowlevel.json")


if __name__ == "__main__":
    main()

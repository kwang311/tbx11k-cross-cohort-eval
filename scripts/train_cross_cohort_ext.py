# -*- coding: utf-8 -*-
"""Paper 2 add-on runs: cross-cohort matrix with (a) alternative backbones and
(b) alternative TBX11K TB definitions (active-only / latent-only).

Derived from train_cross_cohort_qatar.py (verified code) with three deliberate changes:
  1. --backbone {resnet50,resnet18,efficientnet_b0}: the same pipeline paradigm
     (ImageNet-pretrained backbone, early stages frozen, projection + linear head,
     Adam 1e-4 / wd 1e-4, step decay /5, batch 16, class-weighted CE).
     Frozen for resnet18 = conv1/bn1/relu/maxpool/layer1/layer2 (identical to resnet50);
     for efficientnet_b0 = features[0:4] (first four of nine blocks) so that the frozen/trainable
     split is the same early-half / late-half proportion. This is the ONLY deviation from the
     ResNet-50 runs; it is recorded in the output json.
  2. --tbset {full,active,latent}: how TBX11K's TB label is defined when TBX11K is the source
     (full = active + latent, as in the main matrix; active = active_tb + active&latent_tb only;
     latent = latent_tb only). TBX11K's own test split follows the same definition; the other
     three cohorts are unchanged (normal vs their own TB label).
  3. --npz 1: store per-seed predicted probabilities for every (source, target) pair so that
     per-cell confidence intervals can be computed from artefacts (no re-training needed).

Outputs: output_cross/{raw,summary}_<prefix>.json, output_cross/pred_<prefix>/*.npz,
         output_cross/lowlevel_<prefix>.json
Usage:   python train_cross_cohort_ext.py --backbone resnet18 --tbset full --npz 1 --prefix resnet18_full
"""
import os
import csv
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models as tvm

from data_tbx11k import build_ram_cache, RamDataset
from data_tbcohort import load_cohort, TBY
from models import Chest9Classifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output_cross")
COHORTS = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]

# TBX11K tag sets per --tbset choice (mirrors data_tbcohort.TBX_TAG_POS)
TB_TAGS = {
    "full": {"active_tb", "latent_tb", "active&latent_tb"},
    "active": {"active_tb", "active&latent_tb"},
    "latent": {"latent_tb"},
}
NEG_TAGS = {"healthy"}


def load_tbx_split_tbset(tbset):
    """TBX11K official train/val with TB defined by tbset (see module docstring)."""
    pos = TB_TAGS[tbset]
    out = {}
    for split in ("train", "val"):
        paths, labels = [], []
        with open(os.path.join(TBY, f"manifest_{split}.csv"), newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                t = r["tag"]
                if t in pos:
                    lab = 1
                elif t in NEG_TAGS:
                    lab = 0
                else:
                    continue
                paths.append(os.path.join(TBY, split, "img", r["image"]))
                labels.append(lab)
        out[split] = (paths, labels)
    return out


def build_splits(tbset, quick=False):
    splits = {}
    s = load_tbx_split_tbset(tbset)
    splits["TBX11K"] = {"train": s["train"], "test": s["val"]}
    for c in ["Shenzhen", "Montgomery", "Qatar"]:
        p, y = load_cohort(c)
        p, y = np.array(p), np.array(y)
        tr_p, te_p, tr_y, te_y = train_test_split(p, y, test_size=0.2, stratify=y,
                                                  random_state=42)
        splits[c] = {"train": (list(tr_p), list(tr_y)), "test": (list(te_p), list(te_y))}
    if quick:
        rng = np.random.RandomState(0)
        for c in splits:
            for k in ("train", "test"):
                p, y = splits[c][k]
                p, y = np.array(p), np.array(y)
                idx = rng.choice(len(p), min(len(p), 80), replace=False)
                splits[c][k] = (list(p[idx]), list(y[idx]))
    for c in splits:
        trp, trY = np.array(splits[c]["train"][0]), np.array(splits[c]["train"][1])
        tep, teY = np.array(splits[c]["test"][0]), np.array(splits[c]["test"][1])
        assert set(np.unique(trY)) <= {0, 1} and set(np.unique(teY)) <= {0, 1}, f"{c} labels not 0/1"
        assert len(np.unique(trY)) == 2 and len(np.unique(teY)) == 2, f"{c} single-class split"
        assert len(set(trp) & set(tep)) == 0, f"{c} train/test path overlap (leak!)"
        assert len(trY) == len(trp) and len(teY) == len(tep), f"{c} path/label length mismatch"
    print("  [check] labels in {0,1}, both classes present, no train/test overlap: OK", flush=True)
    for c in splits:
        print(f"    {c:11s} train={len(splits[c]['train'][0])} "
              f"(TB={int(np.sum(splits[c]['train'][1]))})  "
              f"test={len(splits[c]['test'][0])} (TB={int(np.sum(splits[c]['test'][1]))})", flush=True)
    return splits


# ----------------------------------------------------------------------
# backbones
# ----------------------------------------------------------------------
class BackboneClassifier(nn.Module):
    """Same head as models.Chest9Classifier, alternative backbone / freeze split."""

    def __init__(self, backbone, num_classes=2, feature_depth=512):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
            self.layer2, self.layer3, self.layer4 = net.layer2, net.layer3, net.layer4
            ch = 512
            self.frozen_modules = [self.stem, self.layer2]   # conv1..layer2, as for resnet50
        elif backbone == "efficientnet_b0":
            net = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.stem = net.features[0:4]          # frozen early half (4 of 9 blocks)
            self.layer2 = nn.Identity()
            self.layer3 = net.features[4:7]
            self.layer4 = net.features[7:9]
            ch = 1280
            self.frozen_modules = [self.stem]
        else:
            raise ValueError(backbone)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(nn.Linear(ch, feature_depth), nn.ReLU(inplace=True),
                                        nn.Dropout(0.5))
        self.classifier = nn.Linear(feature_depth, num_classes)
        for mod in self.frozen_modules:
            for p in mod.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.stem(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(self.projection(x))


def make_model(backbone, num_classes=2):
    if backbone == "resnet50":
        return Chest9Classifier("baseline", num_classes=num_classes)
    return BackboneClassifier(backbone, num_classes=num_classes)


def train_one(train_cache, seed, epochs, device, backbone):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    imgs, lbls = train_cache
    dl = DataLoader(RamDataset(imgs, lbls, train=True), batch_size=16, shuffle=True,
                    num_workers=0, pin_memory=True)
    counts = np.bincount(lbls, minlength=2).astype(np.float32)
    w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    model = make_model(backbone, 2).to(device)
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
    dl = DataLoader(RamDataset(imgs, lbls, train=False), batch_size=32, shuffle=False,
                    num_workers=0)
    probs = []
    for im, lb in dl:
        probs.append(torch.softmax(model(im.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probs), np.asarray(lbls)


def lowlevel_auc(train_cache, test_cache):
    """Same definition as train_cross_cohort_qatar.lowlevel_auc (16x16 + mean/std, LR)."""
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
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "resnet18", "efficientnet_b0"])
    ap.add_argument("--tbset", default="full", choices=["full", "active", "latent"])
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--npz", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()
    prefix = args.prefix or f"{args.backbone}_{args.tbset}"
    if args.quick:
        args.epochs = 1
        args.seeds = "42"
    seeds = [int(s) for s in args.seeds.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== backbone={args.backbone} tbset={args.tbset} prefix={prefix} "
          f"seeds={seeds} epochs={args.epochs} device={device} npz={args.npz} ===", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    pred_dir = os.path.join(OUT_DIR, f"pred_{prefix}")
    if args.npz:
        os.makedirs(pred_dir, exist_ok=True)

    splits = build_splits(args.tbset, quick=args.quick)
    cache = {}
    for c in COHORTS:
        t0 = time.time()
        cache[c] = {"train": build_ram_cache(*splits[c]["train"]),
                    "test": build_ram_cache(*splits[c]["test"])}
        print(f"  cache {c}: train={len(cache[c]['train'][0])} test={len(cache[c]['test'][0])} "
              f"({time.time()-t0:.1f}s)", flush=True)

    results = {}
    for src in COHORTS:
        print(f"\n===== source cohort {src} =====", flush=True)
        for seed in seeds:
            model = train_one(cache[src]["train"], seed, args.epochs, device, args.backbone)
            row = {}
            for tgt in COHORTS:
                probs, labels = predict(model, cache[tgt]["test"], device)
                row[tgt] = float(roc_auc_score(labels, probs[:, 1]))
                if args.npz:
                    np.savez_compressed(os.path.join(pred_dir, f"s{seed}_{src}_to_{tgt}.npz"),
                                        probs=probs.astype(np.float32), labels=labels.astype(np.int64))
            results.setdefault(src, {})[str(seed)] = row
            print(f"  seed {seed}: " + "  ".join(f"{t}={row[t]:.3f}" for t in COHORTS), flush=True)
            with open(os.path.join(OUT_DIR, f"matrix_raw_{prefix}.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    summary = {}
    print(f"\n===== cross-cohort AUC ({prefix}; mean+/-SD over {len(seeds)} seeds) =====", flush=True)
    for src in COHORTS:
        cells = []
        for tgt in COHORTS:
            vals = [results[src][str(s)][tgt] for s in seeds]
            summary.setdefault(src, {})[tgt] = {"mean": float(np.mean(vals)),
                                                "std": float(np.std(vals, ddof=1)),
                                                "per_seed": {str(s): results[src][str(s)][tgt]
                                                             for s in seeds},
                                                "n_test": int(len(cache[tgt]["test"][1]))}
            cells.append(f"{np.mean(vals):.3f}+/-{np.std(vals,ddof=1):.3f}")
        print(f"{src:11s} " + "".join(f"{c:>16s}" for c in cells), flush=True)
    with open(os.path.join(OUT_DIR, f"matrix_summary_{prefix}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    ll = {}
    print("\n===== low-level reference (16x16 + mean/std -> LR), within cohort =====", flush=True)
    for c in COHORTS:
        a = lowlevel_auc(cache[c]["train"], cache[c]["test"])
        ll[c] = {"auc": a, "same_cohort_cnn": summary[c][c]["mean"],
                 "cnn_gain": summary[c][c]["mean"] - a}
        print(f"  {c:11s} lowlevel={a:.4f}  CNN={summary[c][c]['mean']:.4f}  "
              f"gain={summary[c][c]['mean']-a:+.4f}", flush=True)
    with open(os.path.join(OUT_DIR, f"lowlevel_{prefix}.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "same splits as this run; LogisticRegression(max_iter=1000), "
                           "16x16 grayscale thumbnail + mean/std",
                   "backbone": args.backbone, "tbset": args.tbset, "auc": ll}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[OK] wrote matrix_{{raw,summary}}_{prefix}.json, lowlevel_{prefix}.json"
          + (f", pred_{prefix}/*.npz" if args.npz else ""), flush=True)


if __name__ == "__main__":
    main()

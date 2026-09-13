# -*- coding: utf-8 -*-
"""跨库 TB 二元筛查：train 一库 → test 多库（角度①）。

- 任务：normal(0) vs TB(1)
- 库：TBX11K / Shenzhen / Montgomery / Qatar
- 划分：TBX11K 用官方 train/val；其余 80/20 分层（固定 seed）
- 训练：ResNet-50 baseline（冻结前层，同项目范式），多 seed
- 输出：output_cross/matrix_{seed}.json（source × target AUC）+ 汇总

用法：
  python train_cross_cohort.py --quick           # 冒烟（子采样 + 1 seed + 1 epoch）
  python train_cross_cohort.py                   # 全量（5 seed × 15 epoch）
  python train_cross_cohort.py --seeds 42,43     # 指定 seed
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "output_cross")
# ⚠️ 本副本（qatar 版）与 train_cross_cohort.py 的唯一差别：把 Qatar 放回 COHORTS，
#    并把三个输出文件改名（*_qatar.json），以便与"主矩阵=3 库"并存、互不覆写。
#    用途：为 Paper 2 的 Qatar case study 提供**有实物**的跨库格子（2026-09-12 建，
#    起因：08:28 的 3 库重跑覆写了原 4 库矩阵，Qatar 行列失去实物出处）。
COHORTS = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]


def build_splits(quick=False):
    """返回 {cohort: {"train": (paths,y), "test": (paths,y)}}。"""
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
    # --- 防护断言（fail-fast）---
    for c in splits:
        trp, trY = np.array(splits[c]["train"][0]), np.array(splits[c]["train"][1])
        tep, teY = np.array(splits[c]["test"][0]), np.array(splits[c]["test"][1])
        assert set(np.unique(trY)) <= {0, 1} and set(np.unique(teY)) <= {0, 1}, f"{c} 标签非 0/1"
        assert len(set(trp) & set(tep)) == 0, f"{c} train/test 路径重叠（泄漏！）"
        assert len(trY) == len(trp) and len(teY) == len(tep), f"{c} 路径/标签数不匹配"
    print("  [校验] 各库标签 in {0,1}、train/test 无重叠、路径/标签数一致 OK")
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
    """16x16 缩略图 + 均值/标准差 → 逻辑回归 AUC（trivial 低层基线对照）。"""
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
    # 缓存
    cache = {}
    for c in COHORTS:
        t0 = time.time()
        cache[c] = {
            "train": build_ram_cache(*splits[c]["train"]),
            "test": build_ram_cache(*splits[c]["test"]),
        }
        print(f"  缓存 {c}: train={len(cache[c]['train'][0])} test={len(cache[c]['test'][0])} "
              f"({time.time()-t0:.1f}s)", flush=True)

    results = {}  # results[source][seed][target] = auc
    for src in COHORTS:
        print(f"\n===== 训练源库 {src} =====")
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
            with open(os.path.join(OUT_DIR, "matrix_raw_qatar.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总 mean±std
    print("\n===== 跨库 AUC 矩阵（train 行 × test 列，mean±std over seeds）=====")
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
    with open(os.path.join(OUT_DIR, "matrix_summary_qatar.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 低层基线（trivial 对照）
    print("\n===== 低层基线（16x16 + 统计 -> LR，trivial 对照）=====")
    ll = {}
    for c in COHORTS:
        a = lowlevel_auc(cache[c]["train"], cache[c]["test"])
        ll[c] = a
        print(f"  {c:11s} 低层基线 AUC={a:.3f}   同库 CNN AUC={summary[c][c]['mean']:.3f}   "
              f"CNN增益={summary[c][c]['mean'] - a:+.3f}")
    with open(os.path.join(OUT_DIR, "lowlevel_qatar.json"), "w", encoding="utf-8") as f:
        json.dump(ll, f, ensure_ascii=False, indent=2)
    print(f"\n已存 {OUT_DIR}/matrix_raw_qatar.json、matrix_summary_qatar.json、lowlevel_qatar.json")


if __name__ == "__main__":
    main()

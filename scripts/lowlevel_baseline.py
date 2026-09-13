# -*- coding: utf-8 -*-
"""低层基线对照：16x16 缩略图 + 全局统计 → 线性分类器，四库都算。
若 LR(低层) 已≈1.0，说明类间有全局捷径，CNN 高分不可信。"""
import os
import csv
import glob
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def feats(paths, n=700):
    rng = np.random.RandomState(0)
    if len(paths) > n:
        paths = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
    rows, stds = [], []
    for p in paths:
        im = Image.open(p).convert("L")
        a = np.asarray(im, dtype=np.float32)
        small = np.asarray(im.resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
        rows.append(np.concatenate([small, [a.mean()/255, a.std()/255]]))
        stds.append(a.std())
    return np.array(rows), np.array(stds)


def run(name, pos, neg):
    Xp, sp = feats(pos); Xn, sn = feats(neg)
    X = np.vstack([Xp, Xn]); y = np.array([1]*len(Xp) + [0]*len(Xn))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    auc = roc_auc_score(yte, LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    flag = "  <== 捷径红旗" if auc > 0.95 else ""
    print(f"  {name:26s} LR(16x16) AUC={auc:.3f}   pos.std={sp.mean():5.1f} neg.std={sn.mean():5.1f}{flag}")


print("=== 低层基线（16x16 + 全局统计，线性分类器）===")
# TBX11K: healthy(neg) vs TB(active+latent, pos)
man = {}
with open(r"F:\datasets\TBX11K\manifest_val.csv", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        man[r["image"]] = r["tag"]
tb = [rf"F:\datasets\TBX11K\val\img\{i}" for i, t in man.items() if t in ("active_tb", "latent_tb", "active&latent_tb")]
hd = [rf"F:\datasets\TBX11K\val\img\{i}" for i, t in man.items() if t == "healthy"]
run("TBX11K (TB vs healthy)", tb, hd)

sz = glob.glob(r"F:\datasets\TB_public\Shenzhen\Shenzhen\img\*.png")
run("Shenzhen (_1 TB vs _0)", [p for p in sz if p[:-4].endswith("_1")], [p for p in sz if p[:-4].endswith("_0")])
mg = glob.glob(r"F:\datasets\TB_public\Montgomery\Montgomery\img\*.png")
run("Montgomery (_1 TB vs _0)", [p for p in mg if p[:-4].endswith("_1")], [p for p in mg if p[:-4].endswith("_0")])
qa = [p for p in glob.glob(r"G:\Xray\tawsifurrahman\Tuberculosis\*") if p.lower().endswith((".png", ".jpg"))]
qn = [p for p in glob.glob(r"G:\Xray\tawsifurrahman\Normal\*") if p.lower().endswith((".png", ".jpg"))]
run("Qatar (TB vs Normal)", qa, qn)

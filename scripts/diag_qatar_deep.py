# -*- coding: utf-8 -*-
"""Qatar 深层伪影诊断：低层统计 + 简单分类器能否直接分开两类。"""
import os
import glob
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def feats(paths, n=600):
    rng = np.random.RandomState(0)
    if len(paths) > n:
        paths = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
    rows, stds = [], []
    for p in paths:
        im = Image.open(p).convert("L")
        a = np.asarray(im, dtype=np.float32)
        small = np.asarray(im.resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
        rows.append(np.concatenate([small, [a.mean() / 255, a.std() / 255,
                                            np.percentile(a, 5) / 255, np.percentile(a, 95) / 255]]))
        stds.append(a.std())
    return np.array(rows), np.array(stds)


def run(name, pos_paths, neg_paths):
    Xp, sp = feats(pos_paths)
    Xn, sn = feats(neg_paths)
    X = np.vstack([Xp, Xn])
    y = np.array([1] * len(Xp) + [0] * len(Xn))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    print(f"{name:28s} LR(16x16+stats) AUC={auc:.3f}  pos.std={sp.mean():5.1f}  neg.std={sn.mean():5.1f}  "
          f"n_pos={len(Xp)} n_neg={len(Xn)}")


print("=== 低层统计分类器（≈1.0 = 类间存在全局捷径）===")
# Qatar
run("Qatar (TB vs Normal)",
    [p for p in glob.glob(r"G:\Xray\tawsifurrahman\Tuberculosis\*") if p.lower().endswith((".png", ".jpg", ".jpeg"))],
    [p for p in glob.glob(r"G:\Xray\tawsifurrahman\Normal\*") if p.lower().endswith((".png", ".jpg", ".jpeg"))])
# Shenzhen（按文件名 _1=TB / _0=normal）
sz = glob.glob(r"F:\datasets\TB_public\Shenzhen\Shenzhen\img\*.png")
run("Shenzhen (_1 TB vs _0 norm)",
    [p for p in sz if p[:-4].endswith("_1")],
    [p for p in sz if p[:-4].endswith("_0")])
# Montgomery
mg = glob.glob(r"F:\datasets\TB_public\Montgomery\Montgomery\img\*.png")
run("Montgomery (_1 TB vs _0 norm)",
    [p for p in mg if p[:-4].endswith("_1")],
    [p for p in mg if p[:-4].endswith("_0")])

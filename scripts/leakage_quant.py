# -*- coding: utf-8 -*-
"""量化 train/val 泄漏：重复图 + patient 级重叠（回应 R3-M2/M3）。"""
import os
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

F = r"F:\archive"
IMG = r"F:\Xray8_images"
CLASS = ["Normal", "Atelectasis", "Cardiomegaly", "Effusion",
         "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

df = pd.read_csv(rf"{F}\Filtered_Data_Entry_2017.csv")
name2pat = dict(zip(pd.read_csv(rf"{F}\Data_Entry_2017.csv")["Image Index"],
                    pd.read_csv(rf"{F}\Data_Entry_2017.csv")["Patient ID"]))
paths = np.array([os.path.join(IMG, fn) for fn in df["Image Index"].values])
oh = df[CLASS].values
y = oh.argmax(axis=1)
_, _, tr_idx, va_idx = train_test_split(np.arange(len(y)), np.arange(len(y)),
                                        test_size=0.2, stratify=y, random_state=42)
# 每类封顶
rng = np.random.RandomState(42)
def cap(idx, size):
    keep = []
    for c in range(len(CLASS)):
        ci = idx[y[idx] == c]
        if len(ci) > size:
            ci = rng.choice(ci, size, replace=False)
        keep.append(ci)
    return np.sort(np.concatenate(keep))
tr = cap(tr_idx, 1200); va = cap(va_idx, 300)
print(f"train={len(tr)} val={len(va)}  (与论文 9,532 / 2,383 对照)")

tr_paths = paths[tr]; va_paths = paths[va]
tr_names = df["Image Index"].values[tr]; va_names = df["Image Index"].values[va]

print("正在哈希 train...", flush=True)
tr_h = {}
for p, n in zip(tr_paths, tr_names):
    with open(p, "rb") as f:
        tr_h.setdefault(hashlib.md5(f.read()).hexdigest(), []).append(n)
print("正在哈希 val...", flush=True)
va_h = {}
for p, n in zip(va_paths, va_names):
    with open(p, "rb") as f:
        va_h.setdefault(hashlib.md5(f.read()).hexdigest(), []).append(n)

dup_between = set(tr_h) & set(va_h)
n_pairs = sum(len(tr_h[k]) * len(va_h[k]) for k in dup_between)
n_val_dup = sum(len(va_h[k]) for k in dup_between)
print(f"\n[泄漏] 内容相同的 train-val 图对: {n_pairs}  (占 val 的 {n_val_dup/len(va)*100:.2f}%)")
for k in list(dup_between)[:3]:
    print(f"   train{tr_h[k][:2]} == val{va_h[k][:2]}")

# patient 级
tr_pat = set(name2pat.get(n) for n in tr_names)
va_pat = set(name2pat.get(n) for n in va_names)
sh = tr_pat & va_pat
va_in_shared = sum(1 for n in va_names if name2pat.get(n) in sh)
print(f"\n[泄漏] 同时出现在 train 和 val 的患者数: {len(sh)}")
print(f"        这些患者的 val 图数: {va_in_shared}  (占 val 的 {va_in_shared/len(va)*100:.1f}%)")
print(f"        train 患者数={len(tr_pat)}  val 患者数={len(va_pat)}")

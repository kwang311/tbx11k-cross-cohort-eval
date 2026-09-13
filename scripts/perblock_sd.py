# -*- coding: utf-8 -*-
"""核对 SE/CBAM per-block 的逐 seed SD（未舍入原始值 vs 稿内 0.0082/0.0184）。"""
import json, numpy as np
R = json.load(open(r"G:\Xray\tb9_single_label_perblock\output\results.json"))
by = {}
for r in R:
    by.setdefault(r["model"], {})[r["model_seed"]] = r["best_macro_auc"]
seeds = sorted(next(iter(by.values())))
for m in ("baseline","se_block","cbam_block"):
    if m in by:
        v = np.array([by[m][s] for s in seeds])
        print(f"{m:12s} n={len(v)} mean={v.mean():.6f} sd(ddof1)={v.std(ddof=1):.6f} sd(ddof0)={v.std(ddof=0):.6f}")

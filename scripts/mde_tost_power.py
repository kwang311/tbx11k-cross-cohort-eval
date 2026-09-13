# -*- coding: utf-8 -*-
"""重算 P1-6 所需数字：每模块配对差 SD、MDE(80%功效,配对t)、TOST 等价的功效(真差=0, margin=0.01)。"""
import json, numpy as np
from scipy import stats
R = json.load(open(r"G:\Xray\tb9_single_label\output\results.json"))
by = {}
for r in R:
    by.setdefault(r["model"], {})[r["model_seed"]] = r["best_macro_auc"]
seeds = sorted(by["baseline"])
base = np.array([by["baseline"][s] for s in seeds])
z_a, z_b = 1.959964, 0.8416212
print(f"{'mod':6s} {'Δmean':>8s} {'sd(ddof1)':>9s} {'MDE80':>7s} {'SE':>8s} {'TOST_power(Δtrue=0)':>20s}")
for m in ["se","cbam","bam","gcn","gat"]:
    v = np.array([by[m][s] for s in seeds]); d = v - base
    sd = d.std(ddof=1); n = len(d)
    se = sd/np.sqrt(n)
    mde = (z_a+z_b)*se
    # TOST(95%CI 版, margin=0.01): 等价声明当 |dhat| + z_{0.95}*se < 0.01
    z95 = 1.6448536
    thr = 0.01 - z95*se
    power = float(2*stats.norm.cdf(thr/se) - 1) if se > 0 else 1.0
    print(f"{m:6s} {d.mean():+8.4f} {sd:9.4f} {mde:7.4f} {se:8.5f} {max(0.0,power):20.4f}")

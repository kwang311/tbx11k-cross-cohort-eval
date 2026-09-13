# -*- coding: utf-8 -*-
"""Paper 1 统计补充：每模块 ΔAUC 的 95%CI + 等价性(TOST) + MDE。
回应 R1-M1（Blocking）/R2-M3（difference test ≠ equivalence）。"""
import json
import numpy as np
from scipy import stats

R = json.load(open(r"G:\Xray\tb9_single_label\output\results.json", encoding="utf-8"))
by = {}
for r in R:
    by.setdefault(r["model"], {})[r["model_seed"]] = r["best_macro_auc"]
seeds = sorted(by["baseline"])
base = np.array([by["baseline"][s] for s in seeds])
print(f"baseline: mean={base.mean():.4f} sd={base.std(ddof=1):.4f}  (n={len(seeds)})")
print(f"逐 seed baseline: {[round(x,4) for x in base]}\n")

MARGIN = 0.01  # 预设定等价边界（1 个百分点 AUC）——审稿要求的 pre-specified margin
print(f"等价边界 margin = ±{MARGIN} 可检测: 主指标=逐 seed macro-AUC 均值")
print(f"{'model':7s} {'Δmean':>8s} {'Δsd':>7s} {'Δ95%CI(boot)':>20s} {'Wilcoxon':>9s} {'TOST':>12s}")
for m in ["se", "cbam", "bam", "gcn", "gat"]:
    v = np.array([by[m][s] for s in seeds])
    d = v - base
    rng = np.random.RandomState(0)
    bs = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    equiv = (lo > -MARGIN) and (hi < MARGIN)
    print(f"{m:7s} {d.mean():+8.4f} {d.std(ddof=1):7.4f}   [{lo:+.4f}, {hi:+.4f}] {wp:9.4f} "
          f"{'EQUIVALENT' if equiv else 'not equiv':>12s}")

print("\nMDE（配对设计，80% power，alpha=0.05，双侧）:")
for m in ["se", "cbam", "bam", "gcn", "gat"]:
    v = np.array([by[m][s] for s in seeds]); d = v - base
    sd = d.std(ddof=1); n = len(d)
    mde = (1.96 + 0.8416) * sd / np.sqrt(n)
    print(f"  {m:6s} sd(diff)={sd:.4f}  MDE≈{mde:.4f}")
sd_all = np.mean([ (np.array([by[m][s] for s in seeds])-base).std(ddof=1) for m in ["se","cbam","bam","gcn","gat"]])
print(f"  (模块间平均 sd)={sd_all:.4f} → 平均 MDE≈{(1.96+0.8416)*sd_all/np.sqrt(len(seeds)):.4f}")

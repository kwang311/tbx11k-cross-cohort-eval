# -*- coding: utf-8 -*-
"""Paper 2 Fig 2: what the aggregate hides (a) and the low-level reference (b).

数据源（全部现算，禁止抄表）：
  (a) tbX11k_proj/output/predictions/baseline_s{42..51}.npz  -> per-class one-vs-rest AUC (10 seed 均值±SD)
      以及 active-vs-latent 两两 AUC（只用标签 2/3 的子集）
  (b) tbX11k_proj/output_cross/lowlevel_same_split_20260911.json -> 五种低层特征 × 4 库（五统计口径）
输出：fig2_aggregate_and_lowlevel.{png,pdf}（同目录），PNG 300 dpi
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.environ.get("TBX_PROJ", r"G:\Xray\tbx11k_proj")
NPZ_DIR = os.path.join(PROJ, "output", "predictions")
LL_JSON = os.path.join(PROJ, "output_cross", "lowlevel_same_split_20260911.json")

CLASSES = ["Healthy", "Sick, non-TB", "Active TB", "Latent TB"]
COHORTS = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]

# ---------- (a) 从 npz 重算 ----------
# ⚠️ 口径（重要）：active vs latent 的两两 AUC 必须用**两类重归一化分数**
#    p_latent / (p_active + p_latent)（与 analyze_tbx11k.py 一致），不是原始 p[:,3]。
#    用原始 p[:,3] 会得到 ~0.642（per-seed 均值），与稿件 0.6499 不符。
def load_probs():
    files = sorted(glob.glob(os.path.join(NPZ_DIR, "baseline_s*.npz")))
    assert len(files) == 10, f"expected 10 npz, got {len(files)}"
    P, ref = [], None
    for f in files:
        d = np.load(f)
        if ref is None:
            ref = d["labels"]
        else:
            assert np.array_equal(ref, d["labels"]), "val labels differ across seeds"
        P.append(d["probs"])
    return np.stack(P), ref          # (10, N, 4), (N,)


def pair_score(pm, mask):
    """两类重归一化：latent / (active + latent)。"""
    return pm[mask][:, 3] / (pm[mask][:, 2] + pm[mask][:, 3] + 1e-9)


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    N = len(y)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, N, N)
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, ss))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def panel_a_stats():
    P, y = load_probs()
    pm = P.mean(0)                                    # ensemble
    macro_ens = roc_auc_score(y, pm, multi_class="ovr", average="macro", labels=list(range(4)))
    pc_ens = [roc_auc_score((y == c).astype(int), pm[:, c]) for c in range(4)]
    pc_ci = [boot_ci((y == c).astype(int), pm[:, c]) for c in range(4)]
    m = np.isin(y, [2, 3])
    y2 = (y[m] == 3).astype(int)
    pw_ens = roc_auc_score(y2, pair_score(pm, m))
    pw_ci = boot_ci(y2, pair_score(pm, m))
    # per-seed（用于交叉核对，不进图）
    pw_seed = [roc_auc_score(y2, pair_score(P[i], m)) for i in range(len(P))]
    pc_seed = np.array([[roc_auc_score((y == c).astype(int), P[i][:, c]) for c in range(4)]
                        for i in range(len(P))])
    return dict(macro_ens=macro_ens, pc_ens=pc_ens, pc_ci=pc_ci, pw_ens=pw_ens, pw_ci=pw_ci,
                pw_seed=np.array(pw_seed), pc_seed=pc_seed)


S = panel_a_stats()
print("[ensemble] macro-AUC = %.4f" % S["macro_ens"])
for c in range(4):
    print("[ensemble] %-13s = %.4f  CI=[%.4f, %.4f]  | per-seed mean %.4f +/- %.4f"
          % (CLASSES[c], S["pc_ens"][c], S["pc_ci"][c][0], S["pc_ci"][c][1],
             S["pc_seed"][:, c].mean(), S["pc_seed"][:, c].std(ddof=1)))
print("[ensemble] active vs latent (2-class renorm) = %.4f  CI=[%.4f, %.4f]  | per-seed %.4f +/- %.4f"
      % (S["pw_ens"], S["pw_ci"][0], S["pw_ci"][1], S["pw_seed"].mean(), S["pw_seed"].std(ddof=1)))
print("[check] 稿件数字：macroAUC(ens)=0.9893, per-class 0.9999/0.9994/0.9902/0.9675, "
      "a-vs-l=0.6499 CI[0.5620,0.7433], per-seed 0.6209+/-0.0281")

# ---------- (b) 低层参照 ----------
with open(LL_JSON, encoding="utf-8") as f:
    ll = json.load(f)["auc"]
FEATS = [("thumb16_plus_stats5", "16x16 + 5 stats"),
         ("thumb16_only", "16x16 only"),
         ("global_stats5_only", "5 stats only"),
         ("thumb16_border_frame", "border frame"),
         ("thumb16_center_8x8", "center 8x8")]
print("[lowlevel] keys:", list(ll["TBX11K"].keys()))

# ---------- 画 ----------
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.2, 3.5))
ax1 = fig.add_axes([0.07, 0.24, 0.42, 0.68])
ax2 = fig.add_axes([0.60, 0.24, 0.37, 0.68])

# (a) 集合值 + 集合 bootstrap CI（与摘要/表 3 同一口径）
ANNOT = []          # 记录我们自己加的注记，供 QA
labels = CLASSES + ["Active vs\nlatent"]
vals = list(S["pc_ens"]) + [S["pw_ens"]]
err_lo = [S["pc_ens"][c] - S["pc_ci"][c][0] for c in range(4)] + [S["pw_ens"] - S["pw_ci"][0]]
err_hi = [S["pc_ci"][c][1] - S["pc_ens"][c] for c in range(4)] + [S["pw_ci"][1] - S["pw_ens"]]
colors = ["#cfe0f0", "#cfe0f0", "#fbe0c4", "#fbe0c4", "#d94f3d"]
bars = ax1.bar(range(5), vals, color=colors, edgecolor="#333333", linewidth=0.7)
bars[-1].set_hatch("//")
ax1.errorbar(range(5), vals, yerr=[err_lo, err_hi], fmt="none", ecolor="#333333",
             elinewidth=0.8, capsize=2.5)
for i, v in enumerate(vals):
    t = ax1.text(i, v + err_hi[i] + 0.012, "%.3f" % v, ha="center", va="bottom", fontsize=7.0)
    ANNOT.append(t)
ax1.axhline(0.5, color="#888888", lw=0.8, ls=":")
ANNOT.append(ax1.text(4.35, 0.505, "chance", ha="right", va="bottom", fontsize=6.5, color="#666666"))
ax1.set_xticks(range(5))
ax1.set_xticklabels(labels, fontsize=7.5)
ax1.set_ylim(0.45, 1.06)
ax1.set_ylabel("AUC (ensemble of 10 seeds)", fontsize=8.5)
ax1.set_title("(a) Per-class vs the decisive contrast", fontsize=8.5, pad=6)

# (b)
x = np.arange(len(COHORTS))
w = 0.16
shades = ["#2f5d8a", "#6f9dc4", "#b9cfe4", "#d98a6a", "#a5442f"]
for j, (key, name) in enumerate(FEATS):
    ys = [ll[c][key] for c in COHORTS]
    ax2.bar(x + (j - 2) * w, ys, w, label=name, color=shades[j], edgecolor="#333333", linewidth=0.4)
ax2.axhline(0.5, color="#888888", lw=0.8, ls=":")
ax2.set_xticks(x)
ax2.set_xticklabels(COHORTS, fontsize=7.5)
ax2.set_ylim(0.35, 1.12)
ax2.set_ylabel("AUC of logistic probe", fontsize=8.5)
ax2.set_title("(b) Low-level reference, same splits", fontsize=8.5, pad=6)
leg = ax2.legend(fontsize=6.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                 frameon=False, columnspacing=0.8, handlelength=1.1)
for i, c in enumerate(COHORTS):
    v = ll[c]["thumb16_plus_stats5"]
    off = 0.005 if i != 1 else 0.03
    ANNOT.append(ax2.text(i, v + off, "%.3f" % v, ha="center", va="bottom",
                          fontsize=6.8, color="#a5442f"))

# ---------- QA：只看我们主动加的注记，排除刻度/图例等 matplotlib 自建文本 ----------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
boxes = [(t, t.get_window_extent(renderer=rend)) for t in ANNOT]
overlap = 0
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        if boxes[i][1].overlaps(boxes[j][1]):
            overlap += 1
            print("[QA] overlap:", boxes[i][0].get_text()[:22], "|", boxes[j][0].get_text()[:22])
fw, fh = fig.canvas.get_width_height()
outside = sum(1 for _, b in boxes if b.x0 < -1 or b.y0 < -1 or b.x1 > fw + 1 or b.y1 > fh + 1)
# 图例是否压在柱子注记上
legbox = leg.get_window_extent(renderer=rend)
leg_over = sum(1 for _, b in boxes if b.overlaps(legbox))
print("[QA] 注记数 = %d | 注记两两重叠 = %d | 出画布 = %d | 与图例重叠 = %d"
      % (len(boxes), overlap, outside, leg_over))

out = os.path.join(HERE, "fig2_aggregate_and_lowlevel")
fig.savefig(out + ".png", dpi=300)
fig.savefig(out + ".pdf")
print("[OK] wrote %s.png/.pdf" % out)

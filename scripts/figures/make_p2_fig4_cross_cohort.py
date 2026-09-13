# -*- coding: utf-8 -*-
"""Paper 2 Fig 4: cross-cohort transfer matrix (a) and the same-vs-cross drop (b).
(2026-09-13 图号重排：原 fig3 改号为图 4；脚本由 make_p2_fig3.py 更名。)

数据源：tbX11k_proj/output_cross/matrix_summary_qatar.json（4 库 × 4 库 × 5 seed，灰度口径）
脚本内含 fail-fast：把 16 个格子与稿件锁定值逐一核对（|Δ| <= 0.0015），不一致就报错。
输出：fig3_cross_cohort.{png,pdf}
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(r"G:\Xray\tbx11k_proj", "output_cross", "matrix_summary_qatar.json")

COH = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]
# 稿件锁定值（train 行 × test 列），用于 fail-fast 核对
EXPECT = {
    "TBX11K":    [1.000, 0.548, 0.703, 0.807],
    "Shenzhen":  [0.883, 0.916, 0.844, 0.881],
    "Montgomery":[0.889, 0.814, 0.861, 0.720],
    "Qatar":     [0.408, 0.511, 0.566, 1.000],
}

with open(SUMMARY, encoding="utf-8") as f:
    S = json.load(f)

M = np.array([[S[r][c]["mean"] for c in COH] for r in COH])
SD = np.array([[S[r][c]["std"] for c in COH] for r in COH])
for i, r in enumerate(COH):
    for j, c in enumerate(COH):
        if abs(M[i, j] - EXPECT[r][j]) > 0.0015:
            raise SystemExit("❌ 与稿件锁定值不符：%s->%s 实测 %.4f vs 期望 %.3f" % (r, c, M[i, j], EXPECT[r][j]))
print("[fail-fast] 16 格与稿件锁定值全部一致 (<=0.0015)")

# 同库 vs 跨库（跨库=除自身外三库均值）
same = np.diag(M)
cross = np.array([M[i, [j for j in range(4) if j != i]].mean() for i in range(4)])
for i, c in enumerate(COH):
    print("[drop] %-11s same=%.3f cross=%.3f  Δ=%+.3f" % (c, same[i], cross[i], cross[i] - same[i]))

# ---------------- 画 ----------------
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.2, 3.4))
ax1 = fig.add_axes([0.08, 0.22, 0.42, 0.62])
ax2 = fig.add_axes([0.60, 0.22, 0.37, 0.62])
ANNOT = []

# (a) 热图：Qatar 行/列灰显
mask = np.ma.array(M, mask=False)
mask.mask = np.zeros_like(M, dtype=bool)
mask.mask[3, :] = True
mask.mask[:, 3] = True
im = ax1.imshow(mask, cmap="Blues", vmin=0.35, vmax=1.0)
ax1.imshow(np.ma.masked_where(~np.array([[i == 3 or j == 3 for j in range(4)] for i in range(4)]),
                              M), cmap="Greys", vmin=0.35, vmax=1.0, alpha=0.55)
for i in range(4):
    for j in range(4):
        col = "#111111" if (i < 3 and j < 3) else "#333333"
        wt = "bold" if i == j else "normal"
        ANNOT.append(ax1.text(j, i - 0.14, "%.3f" % M[i, j], ha="center", va="center",
                              fontsize=7.2, color=col, weight=wt))
        ANNOT.append(ax1.text(j, i + 0.22, "$\\pm$%.3f" % SD[i, j], ha="center", va="center",
                              fontsize=5.6, color="#555555"))
ax1.set_xticks(range(4)); ax1.set_yticks(range(4))
ax1.set_xticklabels(COH, fontsize=7.0, rotation=20, ha="right")
ax1.set_yticklabels(COH, fontsize=7.0)
ax1.set_xlabel("test cohort", fontsize=8.0)
ax1.set_ylabel("training cohort", fontsize=8.0)
ax1.set_title("(a) Transfer AUC (5 seeds, grayscale)", fontsize=8.5, pad=6)
cb = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.03)
cb.ax.tick_params(labelsize=6.5)
ANNOT.append(ax1.text(3.0, -0.95, "Qatar: excluded from the\nprimary matrix (confound)", ha="center", va="center",
                      fontsize=6.0, color="#a5442f"))

# (b) 同库 vs 跨库
x = np.arange(4); w = 0.36
ax2.bar(x - w / 2, same, w, label="same cohort", color="#2f5d8a", edgecolor="#333333", linewidth=0.5)
ax2.bar(x + w / 2, cross, w, label="other cohorts (mean)", color="#b9cfe4", edgecolor="#333333", linewidth=0.5)
for i, c in enumerate(COH):
    ANNOT.append(ax2.text(i - w / 2, same[i] + 0.015, "%.3f" % same[i], ha="center", va="bottom", fontsize=6.2))
    ANNOT.append(ax2.text(i + w / 2, cross[i] - 0.02, "%.3f" % cross[i], ha="center", va="top",
                          fontsize=6.2, color="#1b3550"))
ax2.set_xticks(x); ax2.set_xticklabels(COH, fontsize=7.0, rotation=20, ha="right")
ax2.set_ylim(0, 1.18)
ax2.set_ylabel("AUC", fontsize=8.0)
ax2.set_title("(b) Same-cohort vs cross-cohort", fontsize=8.5, pad=6)
ax2.legend(fontsize=6.4, loc="upper center", ncol=2, frameon=False, columnspacing=0.8)

# ---------------- QA ----------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
boxes = [(t, t.get_window_extent(renderer=rend)) for t in ANNOT]
overlap = 0
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        if boxes[i][1].overlaps(boxes[j][1]):
            overlap += 1
            print("[QA] overlap:", repr(boxes[i][0].get_text()[:20]), "|", repr(boxes[j][0].get_text()[:20]))
fw, fh = fig.canvas.get_width_height()
outside = sum(1 for _, b in boxes if b.x0 < -1 or b.y0 < -1 or b.x1 > fw + 1 or b.y1 > fh + 1)
print("[QA] 注记数 = %d | 两两重叠 = %d | 出画布 = %d" % (len(boxes), overlap, outside))

out = os.path.join(HERE, "fig4_cross_cohort")
fig.savefig(out + ".png", dpi=300)
fig.savefig(out + ".pdf")
print("[OK] wrote %s.png/.pdf" % out)

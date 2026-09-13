# -*- coding: utf-8 -*-
"""Paper 2 Fig 1: cohorts, protocols and controls (schematic).

三类面板：(a) 四个公共队列及规模；(b) 三套协议；(c) 两个对照（低层参照 / 重复图敏感性）。
数字来源：paper2-data.md（= output_cross/lowlevel_same_split_20260911.json 的 splits 字段 + TBX11K 官方划分）。
输出：fig1_design.{png,pdf}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})

COL_FROZEN = "#cfe0f0"
COL_TRAIN = "#fbe0c4"
COL_CTRL = "#e6dcf2"
COL_WARN = "#f6d3cd"

fig = plt.figure(figsize=(7.2, 5.0))
ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
ax.set_xlim(0, 100)
H = 100 * 5.0 / 7.2
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")
ANNOT = []


def box(x, y, w, h, text, fc, fs=7.0, ec="#333333"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.7))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    ANNOT.append(t)
    return t


def arrow(x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=color, shrinkA=1, shrinkB=1))


def head(x, y, s, fs=8.0):
    ANNOT.append(ax.text(x, y, s, ha="left", va="bottom", fontsize=fs, weight="bold"))


# ---------------- (a) cohorts ----------------
head(1.5, H - 5.0, "(a) Four public cohorts (all as released)")
cw, ch = 22.5, 15.0
y0 = 46.0
COH = [
    ("TBX11K\n(4 labels + boxes)\ntrain 6,600 / val 1,800\ntest 3,302 (no public labels)", COL_FROZEN),
    ("Shenzhen\n(binary, from filename)\n566 images (Mendeley subset)\ntrain 452 / test 114", COL_FROZEN),
    ("Montgomery\n(binary)\n138 images\ntrain 110 / test 28", COL_FROZEN),
    ("Qatar\n(binary)\n4,200 images\ntrain 3,360 / test 840", COL_WARN),
]
xs = [0.5, 24.5, 48.5, 72.5]
for x, (txt, fc) in zip(xs, COH):
    box(x, y0, cw, ch, txt, fc, fs=6.6)
ANNOT.append(ax.text(72.5 + cw / 2, y0 - 1.8, "excluded from the primary matrix\n(class-source confound, case study)",
                     ha="center", va="top", fontsize=6.2, color="#a5442f"))

# ---------------- (b) protocols ----------------
head(1.5, 39.4, "(b) Three protocols, one identical pipeline (ResNet-50, frozen early stages)")
pw, ph = 29.0, 17.5
py = 21.0
PX = [0.5, 32.5, 64.5]
box(PX[0], py, pw, ph,
    "P1  Four-class task (TBX11K)\n10 seeds, 15 epochs\n\nhealthy / sick-non-TB /\nactive TB / latent TB\n\nmacro-AUC 0.9876\nvs a-vs-l 0.6499", COL_TRAIN, fs=6.8)
box(PX[1], py, pw, ph,
    "P2  Dedicated active vs latent\n10 seeds, 15 epochs\n\nonly the two TB classes\n(599 train / 200 val)\n\nAUC 0.6367", COL_TRAIN, fs=6.8)
box(PX[2], py, pw, ph,
    "P3  Cross-cohort transfer\n5 seeds, 15 epochs\n\ntrain on one cohort,\ntest on all four\n(binary normal vs TB)\n\nsame-cohort 1.000 -> 0.55-0.88 cross", COL_TRAIN, fs=6.8)
arrow(PX[0] + pw, py + ph / 2, PX[1], py + ph / 2)
arrow(PX[1] + pw, py + ph / 2, PX[2], py + ph / 2)

# ---------------- (c) controls ----------------
head(1.5, 16.4, "(c) Two controls added to the pipeline")
cy = 0.5
box(0.5, cy, 45.0, 15.0,
    "Low-level reference (same splits)\n16×16 thumbnail + 5 intensity stats\nlogistic: TBX11K 0.977  Qatar 0.976\nbest of 4 learners: 0.997 / 0.996\nShenzhen 0.859   Montgomery 0.875\nbounds how much of a score needs learned pathology",
    COL_CTRL, fs=6.4)
box(50.0, cy, 45.0, 15.0,
    "Duplicate-image sensitivity\ntrain/val overlap: 38 byte-identical + 45 near-duplicate pairs\n(all sick-non-TB); removing all 45:\nmacro-AUC 0.9893 -> 0.9890, accuracy 0.9606 -> 0.9595",
    COL_CTRL, fs=6.6)

# ---------------- QA ----------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
boxes = [(t, t.get_window_extent(renderer=rend)) for t in ANNOT]
overlap = 0
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        if boxes[i][1].overlaps(boxes[j][1]):
            overlap += 1
            print("[QA] overlap:", repr(boxes[i][0].get_text()[:26]), "|", repr(boxes[j][0].get_text()[:26]))
fw, fh = fig.canvas.get_width_height()
outside = sum(1 for _, b in boxes if b.x0 < -1 or b.y0 < -1 or b.x1 > fw + 1 or b.y1 > fh + 1)
print("[QA] 注记数 = %d | 两两重叠 = %d | 出画布 = %d" % (len(boxes), overlap, outside))
for t, b in boxes:
    if b.x0 < -1 or b.y0 < -1 or b.x1 > fw + 1 or b.y1 > fh + 1:
        print("   [out]", repr(t.get_text()[:46]), "x=[%.0f,%.0f] y=[%.0f,%.0f] canvas=(%d,%d)" % (b.x0,b.x1,b.y0,b.y1,fw,fh))

out = os.path.join(HERE, "fig1_design")
fig.savefig(out + ".png", dpi=300)
fig.savefig(out + ".pdf")
print("[OK] wrote %s.png/.pdf" % out)

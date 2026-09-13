# -*- coding: utf-8 -*-
"""Paper 2 Fig 3: dedicated active-vs-latent task — ROC (a) and confusion matrix (b).
(2026-09-13 图号重排：本节 §5.2 先于 §5.3，故原 fig4 改号为图 3；脚本由 make_p2_fig4.py 更名。)

数据源（现算，禁止抄表）：tbX11k_proj/output_a_vs_l/predictions/*.npz（10 seed，200 张 val = 164 active / 36 latent）
输出：fig4_active_vs_latent.{png,pdf}
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(r"G:\Xray\tbx11k_proj", "output_a_vs_l", "predictions")

files = sorted(glob.glob(os.path.join(NPZ, "*.npz")))
assert len(files) == 10, f"expected 10 npz, got {len(files)}"
P, ref = [], None
for f in files:
    d = np.load(f)
    if ref is None:
        ref = d["labels"]
    else:
        assert np.array_equal(ref, d["labels"]), "labels differ across seeds"
    P.append(d["probs"])
P = np.stack(P)                      # (10, 200, 2)
y = ref.astype(int)                  # 0 = active, 1 = latent
n_act, n_lat = int((y == 0).sum()), int((y == 1).sum())

per_seed = np.array([roc_auc_score(y, P[i][:, 1]) for i in range(len(P))])
ens = P.mean(0)
auc_ens = roc_auc_score(y, ens[:, 1])
print("[recompute] per-seed AUC = %.4f +/- %.4f (range %.3f-%.3f)"
      % (per_seed.mean(), per_seed.std(ddof=1), per_seed.min(), per_seed.max()))
print("[recompute] ensemble AUC = %.4f   (n_active=%d, n_latent=%d)" % (auc_ens, n_act, n_lat))
print("[check] 稿件：专项二分类 AUC = 0.6367 +/- 0.0247（逐 seed 0.59-0.67）")

pred = (ens[:, 1] >= 0.5).astype(int)
cm = confusion_matrix(y, pred, labels=[0, 1])
rec = cm[1, 1] / cm[1].sum()
print("[recompute] confusion (rows=true active/latent):\n", cm)
print("[recompute] latent recall = %.4f (%d/%d)" % (rec, cm[1, 1], cm[1].sum()))

# ---------------- 画 ----------------
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42})
fig = plt.figure(figsize=(6.6, 3.1))
ax1 = fig.add_axes([0.08, 0.20, 0.38, 0.70])
ax2 = fig.add_axes([0.58, 0.20, 0.34, 0.70])
ANNOT = []

# (a) ROC
for i in range(len(P)):
    fpr, tpr, _ = roc_curve(y, P[i][:, 1])
    ax1.plot(fpr, tpr, color="#b9cfe4", lw=0.7, zorder=1)
fpr, tpr, _ = roc_curve(y, ens[:, 1])
ax1.plot(fpr, tpr, color="#2f5d8a", lw=1.6, zorder=3,
         label="ensemble  AUC = %.3f" % auc_ens)
ax1.plot([0, 1], [0, 1], color="#888888", lw=0.8, ls=":", zorder=2)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_xlabel("False positive rate", fontsize=8.5)
ax1.set_ylabel("True positive rate (latent TB)", fontsize=8.5)
ax1.set_title("(a) Active vs latent, dedicated model", fontsize=8.5, pad=6)
ax1.legend(fontsize=6.8, loc="lower right", frameon=False)
ANNOT.append(ax1.text(0.03, 0.93, "per-seed AUC = %.3f $\\pm$ %.3f" % (per_seed.mean(), per_seed.std(ddof=1)),
                      fontsize=6.8, color="#333333"))

# (b) 混淆矩阵
im = ax2.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
for r in range(2):
    for c in range(2):
        ax2.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=10,
                 color="white" if cm[r, c] > cm.max() * 0.55 else "#222222")
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["pred\nactive", "pred\nlatent"], fontsize=7.5)
ax2.set_yticks([0, 1]); ax2.set_yticklabels(["true\nactive", "true\nlatent"], fontsize=7.5)
ax2.set_title("(b) Confusion matrix (ensemble, p = 0.5)", fontsize=8.5, pad=6)
ax2.set_xlabel("n = %d active / %d latent" % (n_act, n_lat), fontsize=7.0)
ANNOT.append(ax2.text(0.98, 0.06, "latent recall %.3f (%d/%d)" % (rec, cm[1, 1], cm[1].sum()),
                      transform=ax2.transAxes, ha="right", va="bottom", fontsize=6.8, color="#a5442f"))

# ---------------- QA ----------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
boxes = [(t, t.get_window_extent(renderer=rend)) for t in ANNOT]
overlap = 0
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        if boxes[i][1].overlaps(boxes[j][1]):
            overlap += 1
            print("[QA] overlap:", boxes[i][0].get_text()[:24], "|", boxes[j][0].get_text()[:24])
fw, fh = fig.canvas.get_width_height()
outside = sum(1 for _, b in boxes if b.x0 < -1 or b.y0 < -1 or b.x1 > fw + 1 or b.y1 > fh + 1)
print("[QA] 注记数 = %d | 两两重叠 = %d | 出画布 = %d" % (len(boxes), overlap, outside))

out = os.path.join(HERE, "fig3_active_vs_latent")
fig.savefig(out + ".png", dpi=300)
fig.savefig(out + ".pdf")
print("[OK] wrote %s.png/.pdf" % out)

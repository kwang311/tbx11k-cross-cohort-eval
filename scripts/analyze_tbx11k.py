# -*- coding: utf-8 -*-
"""TBX11K 4 类 baseline 结果分析：per-class AUC + 集成 + Bootstrap CI + 混淆矩阵。"""
import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

PRED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "predictions")
CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
NUM = len(CLASS_NAMES)


def macro_auc(probs, labels):
    return roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                         labels=list(range(NUM)))


def per_class_auc(probs, labels):
    out = {}
    for c in range(NUM):
        y = (labels == c).astype(int)
        out[CLASS_NAMES[c]] = roc_auc_score(y, probs[:, c])
    return out


def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "baseline_s*.npz")))
    print(f"找到 {len(files)} 个 seed 预测文件")
    data = [np.load(f) for f in files]
    labels = data[0]["labels"]
    for d in data:
        assert np.array_equal(d["labels"], labels), "val 标签不一致"

    # 逐 seed
    aucs = [macro_auc(d["probs"], d["labels"]) for d in data]
    accs = [accuracy_score(d["labels"], d["probs"].argmax(1)) for d in data]
    print(f"\n逐 seed macroAUC: {[round(a,4) for a in aucs]}")
    print(f"  mean±std = {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
    print(f"  逐 seed acc: mean={np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")

    # 集成
    probs_mean = np.mean(np.stack([d["probs"] for d in data]), axis=0)
    ens_auc = macro_auc(probs_mean, labels)
    ens_acc = accuracy_score(labels, probs_mean.argmax(1))
    print(f"\n集成 macroAUC={ens_auc:.4f}  acc={ens_acc:.4f}")

    # Bootstrap CI（集成）
    rng = np.random.RandomState(0)
    vals = []
    n = len(labels)
    for _ in range(2000):
        idx = rng.choice(n, n, replace=True)
        try:
            vals.append(macro_auc(probs_mean[idx], labels[idx]))
        except ValueError:
            pass
    lo, hi = np.percentile(vals, [2.5, 97.5])
    print(f"  集成 macroAUC 95% CI = [{lo:.4f}, {hi:.4f}]")

    # per-class（集成）
    print("\nper-class AUC（集成）:")
    pc = per_class_auc(probs_mean, labels)
    for c, v in pc.items():
        print(f"  {c:16s} {v:.4f}")
    # per-seed mean of per-class
    print("\nper-class AUC（逐 seed 均值±std）:")
    for c in range(NUM):
        vs = [per_class_auc(d["probs"], d["labels"])[CLASS_NAMES[c]] for d in data]
        print(f"  {CLASS_NAMES[c]:16s} {np.mean(vs):.4f} ± {np.std(vs, ddof=1):.4f}")

    # 混淆矩阵（集成）
    print("\n混淆矩阵（集成 argmax，行=真值，列=预测）:")
    cm = confusion_matrix(labels, probs_mean.argmax(1), labels=list(range(NUM)))
    print("            " + "".join(f"{c[:9]:>10s}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:12s}" + "".join(f"{v:>10d}" for v in row))

    # active vs latent 二分类子任务（仅这两类的样本）
    mask = np.isin(labels, [2, 3])
    if mask.sum() > 0:
        y2 = (labels[mask] == 3).astype(int)  # latent=1
        p2 = probs_mean[mask][:, 3] / (probs_mean[mask][:, 2] + probs_mean[mask][:, 3] + 1e-9)
        print(f"\n[子任务] active vs latent（n={mask.sum()}）AUC(latent) = {roc_auc_score(y2, p2):.4f}")


if __name__ == "__main__":
    main()

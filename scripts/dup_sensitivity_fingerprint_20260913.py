# -*- coding: utf-8 -*-
"""TBX11K 重复图敏感性 —— 扩展到"指纹级近重复"（45 张），不只是"逐字节相同"（38 张）

背景：`fingerprint_verify_20260913.json` 判定，train×val 有 **45 组**近重复对：
38 对逐字节相同（MD5），另 7 对非字节级但**视觉等同**（256×256 灰度 MAE 0.01–0.42，
Pearson r ≥ 0.9966；随机对照 MAE ≥ 26.1、r ≤ 0.86），且 45 对在 64×64 指纹下全部相同。
稿件 §5.6 此前只报 38 张字节级重复，需给出"剔除全部 45 张"的指标。

口径与 `dup_sensitivity_20260913.py` 完全一致（manifest 行序 → val 顺序；
npz 标签与 val 顺序逐元素断言；macro-AUC = ovr macro；集成 = 10 seed 概率均值）。

产出：output/dup_sensitivity_fingerprint_20260913.{json,txt}
用法：python3 dup_sensitivity_fingerprint_20260913.py
"""
import csv
import glob
import json
import hashlib
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

TBX_ROOT = os.environ.get("TBX_ROOT", "/mnt/f/datasets/TBX11K")
PROJ = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
TAG2CLS = {"healthy": "healthy", "sick_but_non-tb": "sick_but_non-tb",
           "sick_but_non_tb": "sick_but_non-tb", "active_tb": "active_tb",
           "latent_tb": "latent_tb", "active&latent_tb": "active_tb"}
CLS2IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_split(split):
    paths, labels = [], []
    with open(os.path.join(TBX_ROOT, f"manifest_{split}.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cls = TAG2CLS.get(r["tag"])
            if cls is None:
                continue
            paths.append(os.path.join(TBX_ROOT, split, "img", r["image"]))
            labels.append(CLS2IDX[cls])
    return paths, np.array(labels, dtype=np.int64)


def md5(p, buf=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_auc(y, p):
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(4))))


def per_class(y, p):
    return {CLASS_NAMES[c]: float(roc_auc_score((y == c).astype(int), p[:, c])) for c in range(4)}


def main():
    va_paths, va_y = load_split("val")
    tr_paths, tr_y = load_split("train")
    rep = {"script": os.path.basename(__file__), "root": TBX_ROOT,
           "protocol": {"macro_auc": 'roc_auc_score(multi_class="ovr", average="macro")',
                        "ensemble": "10 seeds, elementwise mean of probs",
                        "set_byte_identical": "MD5 of the raw file identical (train vs val)",
                        "set_fingerprint": "64x64 fingerprint identical AND 256x256 MAE < 1 grey level "
                                           "(45 pairs; 38 of them byte-identical, 7 visually identical)"},
           "val_n": len(va_paths), "train_n": len(tr_paths)}

    npz_files = sorted(glob.glob(os.path.join(PROJ, "output", "predictions", "baseline_s*.npz")))
    assert len(npz_files) == 10, f"expected 10 npz, got {len(npz_files)}"
    probs = [np.load(f)["probs"] for f in npz_files]
    labels = np.load(npz_files[0])["labels"]
    assert np.array_equal(labels, va_y), "npz val labels != manifest order (index mapping invalid)"
    ens = np.mean(probs, axis=0)

    # 字节级重复：val 中每个有 train 孪生的图
    tr_md5 = {}
    for p in tr_paths:
        tr_md5.setdefault(md5(p), []).append(os.path.basename(p))
    idx_md5 = [i for i, p in enumerate(va_paths) if md5(p) in tr_md5]

    # 指纹级（45 对）：直接用 fingerprint_verify 的判定结果
    fv = json.load(open(os.path.join(PROJ, "output", "fingerprint_verify_20260913.json"),
                        encoding="utf-8"))
    fp_pairs = [(g["train"], g["val"]) for g in fv["pairs"]]
    val_fp = {v for _, v in fp_pairs}
    idx_fp = [i for i, p in enumerate(va_paths) if os.path.basename(p) in val_fp]
    assert len(idx_fp) == len(val_fp), "some fingerprint val images not found in manifest order"

    rep["sets"] = {
        "n_byte_identical_val": len(idx_md5),
        "n_fingerprint_val": len(idx_fp),
        "extra_beyond_byte": sorted(val_fp - {os.path.basename(va_paths[i]) for i in idx_md5}),
        "fingerprint_val_labels": {c: 0 for c in CLASS_NAMES},
    }
    for i in idx_fp:
        rep["sets"]["fingerprint_val_labels"][CLASS_NAMES[int(va_y[i])]] += 1

    def metrics(keep_idx=None):
        y = va_y if keep_idx is None else va_y[keep_idx]
        p = ens if keep_idx is None else ens[keep_idx]
        d = {"n": int(len(y)), "macro_auc": macro_auc(y, p),
             "accuracy": float(accuracy_score(y, p.argmax(axis=1))), "per_class": per_class(y, p)}
        return d

    keep_md5 = [i for i in range(len(va_y)) if i not in set(idx_md5)]
    keep_fp = [i for i in range(len(va_y)) if i not in set(idx_fp)]
    rep["all_val"] = metrics()
    rep["drop_byte_identical_38"] = metrics(keep_md5)
    rep["drop_fingerprint_45"] = metrics(keep_fp)

    print("=== 结果（10-seed 集成）===")
    for k in ("all_val", "drop_byte_identical_38", "drop_fingerprint_45"):
        d = rep[k]
        print(f"  {k:26s} n={d['n']:5d} macro-AUC={d['macro_auc']:.4f} acc={d['accuracy']:.4f}")
    print("  per-class 45-drop:", {k: round(v, 4) for k, v in rep["drop_fingerprint_45"]["per_class"].items()})

    with open(os.path.join(PROJ, "output", "dup_sensitivity_fingerprint_20260913.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(PROJ, "output", "dup_sensitivity_fingerprint_20260913.txt"), "w",
              encoding="utf-8") as f:
        f.write("TBX11K duplicate sensitivity — byte-identical (38) vs fingerprint-level (45)\n")
        for k in ("all_val", "drop_byte_identical_38", "drop_fingerprint_45"):
            d = rep[k]
            f.write(f"{k}: n={d['n']} macro-AUC={d['macro_auc']:.4f} acc={d['accuracy']:.4f}\n")
        f.write("fingerprint set class breakdown: %s\n" % rep["sets"]["fingerprint_val_labels"])
        f.write("additional (non-byte-identical) val images: %s\n" % rep["sets"]["extra_beyond_byte"])
    print("[OK] wrote dup_sensitivity_fingerprint_20260913.{json,txt}")


if __name__ == "__main__":
    main()

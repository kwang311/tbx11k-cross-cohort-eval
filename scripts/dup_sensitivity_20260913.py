# -*- coding: utf-8 -*-
"""TBX11K 重复图（train∩val 逐字节相同）敏感度定量 —— 为正文 §5.6 生成可追溯实物。

背景：正文 §5.6 称 "train/val 有 38 张逐字节相同的图（全部 sick-but-non-TB），
剔除后集成 macro-AUC 0.9893→0.9890、accuracy 0.9606→0.9597"。此前该数字没有落盘产物。
本脚本按 data_tbx11k.load_data 的同一构建方式复现 val 顺序，用 MD5 找重复，
再在 output/predictions/*.npz 上剔除对应行重算，产出 json/txt。

口径（与 analyze_tbx11k.py / stats.py 一致）：
  macro-AUC = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=range(4))
  集成      = 10 seed 概率逐元素平均后再算指标
不依赖 torch / PIL；只用 hashlib + numpy + sklearn。

用法（WSL）：  python3 dup_sensitivity_20260913.py
       （Windows venv）把 TBX_ROOT 换成 r"F:\\datasets\\TBX11K"
"""
import csv
import glob
import hashlib
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

TBX_ROOT = os.environ.get("TBX_ROOT", "/mnt/f/datasets/TBX11K")
PROJ = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJ, "output", "dup_sensitivity_20260913")

CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
TAG2CLS = {
    "healthy": "healthy", "sick_but_non-tb": "sick_but_non-tb",
    "sick_but_non_tb": "sick_but_non-tb", "active_tb": "active_tb",
    "latent_tb": "latent_tb", "active&latent_tb": "active_tb",
}
CLS2IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_split(split):
    """与 data_tbx11k.load_data 完全同构：manifest 行序 → 路径 + 标签。"""
    paths, labels, dropped = [], [], 0
    with open(os.path.join(TBX_ROOT, f"manifest_{split}.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cls = TAG2CLS.get(r["tag"])
            if cls is None:
                dropped += 1
                continue
            paths.append(os.path.join(TBX_ROOT, split, "img", r["image"]))
            labels.append(CLS2IDX[cls])
    return paths, np.array(labels, dtype=np.int64), dropped


def md5(p, buf=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_auc(y, p):
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro",
                               labels=list(range(4))))


def main():
    res = {"script": os.path.basename(__file__), "root": TBX_ROOT,
           "protocol": {"macro_auc": 'roc_auc_score(multi_class="ovr", average="macro", labels=range(4))',
                        "ensemble": "10 seeds, elementwise mean of probs, then metric",
                        "dup_rule": "MD5 of the raw image file identical between val and train"}}

    va_paths, va_y, va_dropped = load_split("val")
    tr_paths, tr_y, tr_dropped = load_split("train")
    res["splits"] = {"val_n": len(va_paths), "train_n": len(tr_paths),
                     "val_dropped_no_label": va_dropped, "train_dropped_no_label": tr_dropped}
    res["val_class_dist"] = {CLASS_NAMES[i]: int((va_y == i).sum()) for i in range(4)}

    # ---- fail-fast 0：val 顺序/标签必须与 npz 逐元素一致，否则剔除的下标无意义 ----
    npz_files = sorted(glob.glob(os.path.join(PROJ, "output", "predictions", "baseline_s*.npz")))
    assert len(npz_files) == 10, f"期望 10 个 npz，实际 {len(npz_files)}"
    npz_labels = np.load(npz_files[0])["labels"]
    assert np.array_equal(npz_labels, va_y), "npz 的 labels 与 manifest_val.csv 重导结果不一致"
    for f in npz_files:
        assert np.array_equal(np.load(f)["labels"], va_y), f"labels 不一致: {f}"
    res["failfast_npz_labels_match_manifest"] = True

    # ---- MD5 ----
    print("hashing val...", flush=True)
    va_md5 = [md5(p) for p in va_paths]
    print("hashing train...", flush=True)
    tr_md5 = set(md5(p) for p in tr_paths)

    dup_idx = np.array([i for i, h in enumerate(va_md5) if h in tr_md5], dtype=int)
    res["n_duplicates"] = int(len(dup_idx))
    res["dup_val_indices"] = dup_idx.tolist()
    res["dup_class_dist"] = {CLASS_NAMES[i]: int((va_y[dup_idx] == i).sum()) for i in range(4)}
    res["dup_fraction_of_val"] = round(len(dup_idx) / len(va_y), 4)

    # ---- 有/无重复两套口径 ----
    keep = np.ones(len(va_y), dtype=bool)
    keep[dup_idx] = False
    all_probs = np.stack([np.load(f)["probs"] for f in npz_files])   # (10, N, 4)
    ens_all = all_probs.mean(axis=0)
    ens_keep = ens_all[keep]

    res["with_duplicates"] = {
        "n": int(len(va_y)),
        "ensemble_macro_auc": round(macro_auc(va_y, ens_all), 4),
        "ensemble_accuracy": round(float(accuracy_score(va_y, ens_all.argmax(1))), 4),
        "per_class_auc": {CLASS_NAMES[i]: round(float(roc_auc_score(va_y == i, ens_all[:, i])), 4)
                          for i in range(4)},
    }
    res["without_duplicates"] = {
        "n": int(keep.sum()),
        "ensemble_macro_auc": round(macro_auc(va_y[keep], ens_keep), 4),
        "ensemble_accuracy": round(float(accuracy_score(va_y[keep], ens_keep.argmax(1))), 4),
        "per_class_auc": {CLASS_NAMES[i]: round(float(roc_auc_score(va_y[keep] == i, ens_keep[:, i])), 4)
                          for i in range(4)},
    }

    # ---- fail-fast 1：剔除前必须复现正文的 0.9893 / 0.9606，否则本脚本口径与正文不一致 ----
    exp_auc_all, exp_acc_all = 0.9893, 0.9606
    got_auc_all = res["with_duplicates"]["ensemble_macro_auc"]
    got_acc_all = res["with_duplicates"]["ensemble_accuracy"]
    res["failfast_reproduces_manuscript_ensemble"] = bool(
        abs(got_auc_all - exp_auc_all) <= 5e-5 and abs(got_acc_all - exp_acc_all) <= 5e-5)
    assert res["failfast_reproduces_manuscript_ensemble"], \
        f"剔除前集成 {got_auc_all}/{got_acc_all} != 正文 {exp_auc_all}/{exp_acc_all}"

    # ---- 正文 §5.6 断言的自动核对（记录，不抛错）----
    res["manuscript_section_5_6_claim"] = {
        "n_duplicates": 38,
        "all_sick_but_non_tb": True,
        "macro_auc_0.9893_to_0.9890": True,
        "accuracy_0.9606_to_0.9597": True,
        "sick_but_non_tb_auc_unchanged_to_4dp": True,
    }
    res["manuscript_section_5_6_verdict"] = {
        "n_duplicates": res["n_duplicates"] == 38,
        "all_sick_but_non_tb": res["dup_class_dist"]["sick_but_non-tb"] == res["n_duplicates"],
        "macro_auc_to_0.9890": res["without_duplicates"]["ensemble_macro_auc"] == 0.9890,
        "accuracy_to_0.9597": res["without_duplicates"]["ensemble_accuracy"] == 0.9597,
        "sick_auc_unchanged_to_4dp":
            res["without_duplicates"]["per_class_auc"]["sick_but_non-tb"]
            == res["with_duplicates"]["per_class_auc"]["sick_but_non-tb"],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    with open(OUT + ".txt", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(res, ensure_ascii=False, indent=2))

    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n=> {OUT}.json / .txt")
    bad = [k for k, v in res["manuscript_section_5_6_verdict"].items() if not v]
    print("§5.6 断言核对：" + ("全部符合 ✓" if not bad else f"不符项 {bad}"))


if __name__ == "__main__":
    main()

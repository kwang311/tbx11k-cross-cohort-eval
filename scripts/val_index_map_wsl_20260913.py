# -*- coding: utf-8 -*-
"""导出 val 行号索引映射 —— 回应 rocky30 咨询件 Q2（45 对那一档的去重指标要可独立复算）

它的问题：包内只给了 7 张「额外近重复」的文件名，没给它们在 val 中的**行号**；
而 val 的文件名编号与 manifest 行序不是一回事（它验证过：38 张里文件名编号 109/135/…/4862
对不上工件给的 38 个行号 818/828/…/1580）。

本脚本把三套索引显式落盘，任何人可用 `manifest_val.csv` + 四类 npz 复算：
  * `byte_identical_38`: val 中 MD5 与某张 train 图相同的行（388 张里 38 张）
  * `extra_7`           : 指纹级额外多出的 7 张（非字节级）
  * `fingerprint_45`    : 38 ∪ 7 = 45
每项给 `{row, file, md5, md5_identical_train}`。

fail-fast（三条，任一不过就中断）：
  1) npz 的 labels 与 manifest val 顺序逐元素相等（索引映射有效）；
  2) 三套集合的规模 == 38 / 7 / 45；
  3) 用这三套索引剔除后重算的 macro-AUC / accuracy == 工件
     `dup_sensitivity_fingerprint_20260913.json` 的对应值（1e-9 内）。

用法：TBX_ROOT=/mnt/f/datasets/TBX11K python3 val_index_map_wsl_20260913.py
产物：output/val_row_index_map_wsl_20260913.{json,txt}
"""
import os
import re
import csv
import glob
import json
import hashlib
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

PROJ = os.path.dirname(os.path.abspath(__file__))
TBX_ROOT = os.environ.get("TBX_ROOT", "/mnt/f/datasets/TBX11K")
OUT = os.path.join(PROJ, "output", "val_row_index_map_wsl_20260913.json")
CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
TAG2CLS = {"healthy": "healthy", "sick_but_non-tb": "sick_but_non-tb", "sick_but_non_tb": "sick_but_non-tb",
           "active_tb": "active_tb", "latent_tb": "latent_tb", "active&latent_tb": "active_tb"}
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


def main():
    va_paths, va_y = load_split("val")
    tr_paths, tr_y = load_split("train")
    npz_files = sorted(glob.glob(os.path.join(PROJ, "output", "predictions", "baseline_s*.npz")))
    ens = np.mean([np.load(f)["probs"] for f in npz_files], axis=0)
    labels = np.load(npz_files[0])["labels"]
    assert np.array_equal(labels, va_y), "npz val labels != manifest order (index mapping invalid)"

    tr_md5 = {}
    for p in tr_paths:
        tr_md5.setdefault(md5(p), []).append(os.path.basename(p))
    idx_md5 = [i for i, p in enumerate(va_paths) if md5(p) in tr_md5]

    fv = json.load(open(os.path.join(PROJ, "output", "fingerprint_verify_20260913.json"), encoding="utf-8"))
    val_fp = {g["val"] for g in fv["pairs"]}
    idx_fp = [i for i, p in enumerate(va_paths) if os.path.basename(p) in val_fp]
    assert len(idx_fp) == len(val_fp), "some fingerprint val images not found in manifest order"

    rec38 = [{"row": i, "file": os.path.basename(va_paths[i]), "md5": md5(va_paths[i])} for i in idx_md5]
    extra = sorted(val_fp - {os.path.basename(va_paths[i]) for i in idx_md5})
    rec7 = []
    for f in extra:
        i = [j for j, p in enumerate(va_paths) if os.path.basename(p) == f][0]
        rec7.append({"row": i, "file": f, "md5": md5(va_paths[i]),
                     "train_twin": sorted({g["train"] for g in fv["pairs"] if g["val"] == f})})
    rec45 = [{"row": i, "file": os.path.basename(va_paths[i]), "md5_identical_train": bool(os.path.basename(va_paths[i]) in {r["file"] for r in rec38})}
             for i in idx_fp]

    assert len(rec38) == 38 and len(rec7) == 7 and len(rec45) == 45, (len(rec38), len(rec7), len(rec45))

    keep38 = [i for i in range(len(va_y)) if i not in set(idx_md5)]
    keep45 = [i for i in range(len(va_y)) if i not in set(idx_fp)]
    art = json.load(open(os.path.join(PROJ, "output", "dup_sensitivity_fingerprint_20260913.json"), encoding="utf-8"))
    got = {
        "all": {"n": int(len(va_y)), "macro_auc": macro_auc(va_y, ens),
                "accuracy": float(accuracy_score(va_y, ens.argmax(1)))},
        "drop38": {"n": int(len(keep38)), "macro_auc": macro_auc(va_y[keep38], ens[keep38]),
                   "accuracy": float(accuracy_score(va_y[keep38], ens[keep38].argmax(1)))},
        "drop45": {"n": int(len(keep45)), "macro_auc": macro_auc(va_y[keep45], ens[keep45]),
                   "accuracy": float(accuracy_score(va_y[keep45], ens[keep45].argmax(1)))},
    }
    for key, ref in (("all", "all_val"), ("drop38", "drop_byte_identical_38"), ("drop45", "drop_fingerprint_45")):
        for m in ("n", "macro_auc", "accuracy"):
            exp = art[ref][m]
            v = got[key][m]
            assert abs(float(exp) - float(v)) < 1e-9 or int(exp) == int(v), f"{key}.{m}: {v} vs {exp}"

    payload = {
        "note": ("explicit val row indices for the duplicate-image sensitivity; row order = manifest_val.csv order "
                 "(the same order as output/predictions/*.npz labels; asserted elementwise)"),
        "scripts": {"this": "val_index_map_wsl_20260913.py",
                    "sensitivity": "dup_sensitivity_fingerprint_20260913.py",
                    "fingerprint_pairs": "fingerprint_verify_20260913.json"},
        "val_n": int(len(va_y)),
        "byte_identical_38": rec38,
        "extra_7": rec7,
        "fingerprint_45": rec45,
        "self_check": {"recomputed": got,
                       "artifact": {k: {m: art[k][m] for m in ("n", "macro_auc", "accuracy")}
                                    for k in ("all_val", "drop_byte_identical_38", "drop_fingerprint_45")}},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(OUT.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {OUT} (+ .txt)")
    print(f"  38 行号: {[r['row'] for r in rec38][:12]} …")
    print(f"  7  行号: {[r['row'] for r in rec7]}")
    print(f"  自检: drop38 macro-AUC={got['drop38']['macro_auc']:.10f} / drop45={got['drop45']['macro_auc']:.10f}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""逐文件 MD5 清单 + 重复图清单 —— 回应 rocky30 材料需求单【需求 3b】

它的原话：要“38 张 train∩val 重复图的文件名清单；三库（TBX11K 的 train 与 val、Qatar 的
Normal 与 Tuberculosis、Shenzhen、Montgomery）的逐文件 MD5 清单；以及生成这些清单的脚本。
只要清单，不要原图。”

本脚本用与管线同源的路径来源（`data_tbcohort.load_cohort` 与 TBX11K 的 manifest_*.csv），
对六个集合逐文件算 MD5，落盘清单；并顺带给出：
  * TBX11K train ∩ val 的 MD5 重复清单（即 §5.6 那 38 张，逐张给文件名 + md5）
  * 每个集合内部的重复组（Qatar TB 类内 3 组）
  * 关键断言：TBX11K train/val 的重复计数、Qatar TB 内重复组数（与稿件口径一致才继续）

用法：python make_md5_lists_wsl_20260913.py
产物：output_cross/md5_lists_20260913/{TBX11K_train,TBX11K_val,Qatar_Normal,Qatar_Tuberculosis,
      Shenzhen,Montgomery}.md5.txt + duplicates_summary_wsl_20260913.json + _INDEX.json
"""
import os
import csv
import json
import glob
import hashlib
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
from data_tbcohort import load_cohort  # noqa: E402

TBX_ROOT = r"F:\datasets\TBX11K"
DST = os.path.join(BASE, "output_cross", "md5_lists_20260913")
os.makedirs(DST, exist_ok=True)


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def tbx_paths(split):
    man = os.path.join(TBX_ROOT, f"manifest_{split}.csv")
    out = []
    with open(man, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(os.path.join(TBX_ROOT, split, "img", r["image"]))
    return out


def main():
    sets = {
        "TBX11K_train": tbx_paths("train"),
        "TBX11K_val": tbx_paths("val"),
    }
    for name, key in [("Shenzhen", "Shenzhen"), ("Montgomery", "Montgomery"), ("Qatar", "Qatar")]:
        p, y = load_cohort(key)
        if name == "Qatar":
            sets["Qatar_Normal"] = [x for x, yy in zip(p, y) if yy == 0]
            sets["Qatar_Tuberculosis"] = [x for x, yy in zip(p, y) if yy == 1]
        else:
            sets[name] = list(p)

    index, digests = {}, {}
    for name, paths in sets.items():
        paths = sorted(paths)
        rec = {}
        fn = os.path.join(DST, f"{name}.md5.txt")
        with open(fn, "w", encoding="utf-8") as fh:
            for p in paths:
                try:
                    d = md5_of(p)
                except Exception as e:
                    print(f"  [warn] unreadable {p}: {e}", flush=True)
                    continue
                rec[os.path.basename(p)] = d
                fh.write(f"{d}  {os.path.basename(p)}\n")
        digests[name] = rec
        n_unique = len(set(rec.values()))
        print(f"[OK] {fn}  files={len(rec)} unique_md5={n_unique}", flush=True)
        index[os.path.basename(fn)] = {"n_files": len(rec), "n_unique_md5": n_unique}

    # TBX11K train ∩ val（§5.6 的 38 张）
    tr, va = digests["TBX11K_train"], digests["TBX11K_val"]
    tr_by_md5 = defaultdict(list)
    for f, d in tr.items():
        tr_by_md5[d].append(f)
    cross = [(f, d, tr_by_md5[d]) for f, d in va.items() if d in tr_by_md5]
    cross.sort()
    with open(os.path.join(DST, "TBX11K_train_val_duplicates.md5.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"# TBX11K val images whose MD5 also occurs in train: {len(cross)}\n")
        fh.write("# val_file  md5  train_file(s)\n")
        for f, d, t in cross:
            fh.write(f"{f}  {d}  {','.join(t)}\n")

    # 各集合内部重复组
    internal = {}
    for name, rec in digests.items():
        byd = defaultdict(list)
        for f, d in rec.items():
            byd[d].append(f)
        groups = {d: sorted(v) for d, v in byd.items() if len(v) > 1}
        internal[name] = {"n_groups": len(groups),
                          "groups": {d: sorted(v) for d, v in sorted(groups.items())}}
        print(f"  {name}: internal duplicate groups = {len(groups)}", flush=True)

    summary = {
        "note": ("per-file MD5 lists for the six image sets used in Paper 2; "
                 "paths come from the same sources as the pipeline "
                 "(data_tbcohort.load_cohort; TBX11K manifest_{train,val}.csv); "
                 "no images shipped, lists only"),
        "scripts": {"this": "make_md5_lists_wsl_20260913.py"},
        "sets": index,
        "tbx11k_train_val_duplicates": {
            "n": len(cross),
            "file": "TBX11K_train_val_duplicates.md5.txt",
            "pairs": [{"val": f, "md5": d, "train": t} for f, d, t in cross],
        },
        "internal_duplicates": internal,
    }
    with open(os.path.join(DST, "duplicates_summary_wsl_20260913.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DST, "_INDEX.json"), "w", encoding="utf-8") as f:
        json.dump({"files": index, "tbx11k_train_val_duplicates_n": len(cross),
                   "internal_groups": {k: v["n_groups"] for k, v in internal.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {DST}/duplicates_summary_wsl_20260913.json "
          f"(train∩val duplicates = {len(cross)})", flush=True)


if __name__ == "__main__":
    main()

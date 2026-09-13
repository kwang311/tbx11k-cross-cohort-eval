# -*- coding: utf-8 -*-
"""导出跨库逐图预测的“可回图”索引 —— 回应 rocky30 材料需求单【需求 1a】

它的原话：跨库各格要给“至少含文件标识（stem）、真实标签、正类概率”的逐图数据，
且“口径需与 matrix_summary_qatar.json 的同一批运行对应”，并要生成脚本与确切调用命令。

本脚本把 output_cross/pred_<prefix>/*.npz 展开成 CSV：
  <prefix>_test-<Target>.csv   列：stem, label, s42_<Src1>, s42_<Src2>, ... s46_<Src4>
每行一图（该目标库的 test 集，顺序 = train_cross_cohort_ext.build_splits → build_ram_cache）。

fail-fast（三条，任一不过就中断）：
  1) 行数 == build_ram_cache 后的长度；
  2) 每个 npz 的 labels 与 split 标签逐元素相等（保证 stem 与概率对得上）；
  3) 5 个 seed 的 labels 互相一致。
另出 _twins.json：Qatar 两张“孪生图”（Tuberculosis-244/509）在 Qatar→Qatar 各 seed 的
得分与名次（量化“记忆效应”），供需求单里“被删图在本评测里的实际得分”一项。

用法：set TBX_GRAY=1 && python export_cross_pred_index_wsl_20260913.py
产物：output_cross/cross_pred_index_20260913/*.csv + _twins.json + _INDEX.json
"""
import os
import json
import csv
import numpy as np
from PIL import Image

assert os.environ.get("TBX_GRAY", "") == "1", "set TBX_GRAY=1 (main matrix preprocessing)"

import train_cross_cohort_ext as X                      # noqa: E402
from data_tbx11k import build_ram_cache, CACHE_SIZE, GRAYSCALE  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output_cross")
DST = os.path.join(OUT_DIR, "cross_pred_index_20260913")
os.makedirs(DST, exist_ok=True)

COHORTS = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]
SEEDS = [42, 43, 44, 45, 46]
PREFIXES = [("resnet50_full", "full"), ("resnet18_full", "full"), ("effnetb0_full", "full"),
            ("resnet50_active", "active"), ("resnet50_latent", "latent")]

TWINS = ["Tuberculosis-244", "Tuberculosis-509"]


def readable_mask(paths):
    """与 data_tbx11k.build_ram_cache 完全相同的可读性判定（open→convert→resize）。"""
    ok = np.ones(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        try:
            im = Image.open(p)
            im = im.convert("L").convert("RGB") if GRAYSCALE else im.convert("RGB")
            im.resize((CACHE_SIZE, CACHE_SIZE), Image.BILINEAR)
        except Exception:
            ok[i] = False
    return ok


def main():
    index = {"note": ("per-image probabilities expanded from output_cross/pred_<prefix>/*.npz; "
                      "probabilities are written with %.9g (float32 round-trip), so AUCs recomputed "
                      "from this CSV match matrix_summary_<prefix>.json; "
                      "row order = train_cross_cohort_ext.build_splits(tbset) -> build_ram_cache; "
                      "TBX_GRAY=1; columns s<seed>_<source> = P(positive class)"),
             "generate": "export_cross_pred_index_wsl_20260913.py",
             "usage": ("set TBX_GRAY=1 && python train_cross_cohort_ext.py --backbone {resnet50|resnet18|"
                       "efficientnet_b0} --tbset {full|active|latent} --npz 1 --prefix <prefix>"),
             "files": {}}
    twins = {}
    for prefix, tbset in PREFIXES:
        splits = X.build_splits(tbset, quick=False)
        for tgt in COHORTS:
            paths, labels = splits[tgt]["test"]
            imgs, y = build_ram_cache(paths, labels)
            ok = readable_mask(paths)
            assert int(ok.sum()) == len(imgs), f"{prefix}/{tgt}: mask {int(ok.sum())} != cache {len(imgs)}"
            assert len(y) == len(imgs), f"{prefix}/{tgt}: labels/cache length mismatch"
            stems = [os.path.splitext(os.path.basename(p))[0] for p, k in zip(paths, ok) if k]

            cols, ref = {}, None
            for src in COHORTS:
                for seed in SEEDS:
                    f = os.path.join(OUT_DIR, f"pred_{prefix}", f"s{seed}_{src}_to_{tgt}.npz")
                    if not os.path.exists(f):
                        print(f"  [skip] missing {os.path.basename(f)}", flush=True)
                        continue
                    d = np.load(f)
                    lab = d["labels"]
                    assert len(lab) == len(y), f"{os.path.basename(f)}: n {len(lab)} != {len(y)}"
                    assert (lab == y).all(), f"{os.path.basename(f)}: labels mismatch vs split"
                    if ref is None:
                        ref = lab
                    else:
                        assert (lab == ref).all(), f"{os.path.basename(f)}: seed labels differ"
                    cols[f"s{seed}_{src}"] = d["probs"][:, 1]

            if not cols:
                continue
            fn = os.path.join(DST, f"{prefix}_test-{tgt}.csv")
            with open(fn, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["stem", "label"] + list(cols.keys()))
                for i, s in enumerate(stems):
                    w.writerow([s, int(y[i])] + [f"{float(v[i]):.9g}" for v in cols.values()])
            index["files"][os.path.basename(fn)] = {
                "prefix": prefix, "tbset": tbset, "target": tgt, "n": int(len(y)),
                "pos": int((np.asarray(y) == 1).sum()), "n_prob_cols": len(cols)}
            print(f"[OK] {fn}  n={len(y)} pos={int((np.asarray(y)==1).sum())} cols={len(cols)}", flush=True)

            # 孪生图（Qatar 目标）得分与名次
            if tgt == "Qatar" and f"s42_{'Qatar'}" in cols:
                for t in TWINS:
                    if t not in stems:
                        continue
                    i = stems.index(t)
                    entry = {"row": int(i), "label": int(y[i])}
                    per = {c: float(v[i]) for c, v in cols.items() if c.endswith("_Qatar")}
                    entry["p_pos_qatar_to_qatar_per_seed"] = per
                    pos = np.where(np.asarray(y) == 1)[0]
                    neg = np.where(np.asarray(y) == 0)[0]
                    entry["rank_within_positives"] = int((cols["s42_Qatar"][pos] > cols["s42_Qatar"][i]).sum() + 1)
                    entry["n_positives"] = int(len(pos))
                    entry["n_negatives"] = int(len(neg))
                    entry["min_negative_prob"] = float(cols["s42_Qatar"][neg].min())
                    entry["max_negative_prob"] = float(cols["s42_Qatar"][neg].max())
                    twins.setdefault(prefix, {})[t] = entry

    with open(os.path.join(DST, "_INDEX.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DST, "_twins.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "Qatar twins: per-seed P(positive) in the Qatar->Qatar cell, with rank; "
                           "scores from output_cross/pred_<prefix>/s<seed>_Qatar_to_Qatar.npz",
                   "twins": twins}, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {DST}/_INDEX.json and _twins.json", flush=True)


if __name__ == "__main__":
    main()

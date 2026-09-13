# -*- coding: utf-8 -*-
"""复核 32x32 指纹找到的跨划分重叠：真近重复 还是 粗哈希碰撞？

背景：`fingerprint_overlap_full_20260913.py` 用 32x32 二值指纹（按图内均值二值化）
在全量 train(6,600) × val(1,800) 上找到 **45 组跨划分重叠**，而字节级 MD5 只找到
38 张（§5.6 报的就是这 38 张）。两者差 7 组，必须判定这 45 组里哪些是真近重复。

本脚本对每一组跨划分候选做三级判定：
  L1 字节级：文件 MD5 是否相同                      → 真重复（可复现的泄漏）
  L2 细指纹：64x64 二值指纹是否相同                  → 强近重复
  L3 像素级：256x256 灰度图的 MAE 与 Pearson r       → 量化相似度
并给出随机对照（从 train/val 各抽同样数量的**非**碰撞对）作为基线。

产出：output/fingerprint_verify_20260913.{json,txt}
"""
import os
import glob
import json
import hashlib
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = r"F:\datasets\TBX11K"
PROJ = os.path.dirname(os.path.abspath(__file__))


def fp(path, n=32):
    im = Image.open(path).convert("L").resize((n, n), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return hashlib.md5((a > a.mean()).astype(np.uint8).tobytes()).hexdigest()


def md5f(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gray256(path):
    im = Image.open(path).convert("L").resize((256, 256), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def listdir(split):
    return sorted(glob.glob(os.path.join(ROOT, split, "img", "*.png")))


def main():
    tr = listdir("train")
    va = listdir("val")
    print(f"train {len(tr)} / val {len(va)}", flush=True)

    fp32_tr, fp32_va = {}, {}
    for i, p in enumerate(tr):
        fp32_tr.setdefault(fp(p, 32), []).append(p)
        if (i + 1) % 2000 == 0:
            print(f"  fp32 train {i+1}/{len(tr)}", flush=True)
    for i, p in enumerate(va):
        fp32_va.setdefault(fp(p, 32), []).append(p)
    cross = sorted(set(fp32_tr) & set(fp32_va))
    print(f"cross-split groups (32x32): {len(cross)}", flush=True)

    groups = []
    for gi, h in enumerate(cross):
        for pt in fp32_tr[h]:
            for pv in fp32_va[h]:
                gt, gv = gray256(pt), gray256(pv)
                r = float(np.corrcoef(gt.ravel(), gv.ravel())[0, 1])
                groups.append({
                    "fp32": h[:12],
                    "train": os.path.basename(pt), "val": os.path.basename(pv),
                    "md5_identical": md5f(pt) == md5f(pv),
                    "fp64_identical": fp(pt, 64) == fp(pv, 64),
                    "mae_256": float(np.abs(gt - gv).mean()),
                    "pearson_r_256": r,
                })
        if (gi + 1) % 10 == 0:
            print(f"  verified {gi+1}/{len(cross)} groups", flush=True)

    # 随机对照：非碰撞对
    rng = np.random.RandomState(0)
    ctrl = []
    for _ in range(len(groups)):
        pt = tr[rng.randint(len(tr))]
        pv = va[rng.randint(len(va))]
        gt, gv = gray256(pt), gray256(pv)
        ctrl.append({"mae_256": float(np.abs(gt - gv).mean()),
                     "pearson_r_256": float(np.corrcoef(gt.ravel(), gv.ravel())[0, 1]),
                     "fp64_identical": fp(pt, 64) == fp(pv, 64)})

    rep = {
        "date": "2026-09-13", "script": os.path.basename(__file__),
        "question": "45 组 32x32 跨划分重叠里，哪些是真近重复、哪些是粗哈希碰撞？",
        "protocol": {
            "L1_md5": "raw file md5 identical",
            "L2_fp64": "64x64 grayscale fingerprint (binarised at image mean) identical",
            "L3_pixels": "MAE and Pearson r between 256x256 grayscale versions",
            "control": "same number of random non-colliding train/val pairs",
        },
        "n_cross_groups": len(cross),
        "n_pairs": len(groups),
        "counts": {
            "md5_identical": int(sum(g["md5_identical"] for g in groups)),
            "fp64_identical": int(sum(g["fp64_identical"] for g in groups)),
            "only_fp32": int(sum(1 for g in groups if not g["md5_identical"] and not g["fp64_identical"])),
        },
        "pairs": groups,
        "control_stats": {
            "mae_mean": float(np.mean([c["mae_256"] for c in ctrl])),
            "mae_min": float(np.min([c["mae_256"] for c in ctrl])),
            "r_mean": float(np.mean([c["pearson_r_256"] for c in ctrl])),
            "r_max": float(np.max([c["pearson_r_256"] for c in ctrl])),
            "fp64_identical": int(sum(c["fp64_identical"] for c in ctrl)),
        },
    }
    print("\n=== 判定 ===", flush=True)
    print(json.dumps(rep["counts"], indent=2), flush=True)
    print("对照组:", json.dumps(rep["control_stats"]), flush=True)
    with open(os.path.join(PROJ, "output", "fingerprint_verify_20260913.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(PROJ, "output", "fingerprint_verify_20260913.txt"), "w",
              encoding="utf-8") as f:
        f.write("Verification of 32x32 cross-split fingerprint overlaps (2026-09-13)\n")
        f.write(f"cross groups {len(cross)}, pairs {len(groups)}\n")
        f.write(f"byte-identical (MD5): {rep['counts']['md5_identical']}\n")
        f.write(f"64x64 fingerprint identical: {rep['counts']['fp64_identical']}\n")
        f.write(f"only the coarse 32x32 fingerprint matches: {rep['counts']['only_fp32']}\n")
        f.write(f"control (random pairs): MAE mean {rep['control_stats']['mae_mean']:.2f} "
                f"min {rep['control_stats']['mae_min']:.2f}; r mean {rep['control_stats']['r_mean']:.3f} "
                f"max {rep['control_stats']['r_max']:.3f}\n")
        for g in sorted(groups, key=lambda x: x["mae_256"]):
            f.write(f"  {g['fp32']}  train {g['train']:14s} val {g['val']:14s} "
                    f"md5={g['md5_identical']} fp64={g['fp64_identical']} "
                    f"MAE={g['mae_256']:.2f} r={g['pearson_r_256']:.3f}\n")
    print("[OK] wrote fingerprint_verify_20260913.{json,txt}", flush=True)


if __name__ == "__main__":
    main()

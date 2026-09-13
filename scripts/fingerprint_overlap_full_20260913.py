# -*- coding: utf-8 -*-
"""全量 32x32 灰度指纹近重复排查（train × val，不抽样）

背景：audit_tbx11k.py 的第 4 部分为省时只抽了 train 的一部分；
稿件 §4 声明 "Train/validation overlap was checked by content hashing and
32x32 fingerprints"，要让这句话可核，需对**全部** train(6,600) 与 val(1,800)
逐张算指纹并统计跨划分重叠组。

指纹定义与 audit_tbx11k.py 完全一致：RGB->L、resize 32x32 BILINEAR、
以图内均值为阈值二值化、按字节串取 md5。

产出：output/fingerprint_overlap_full_20260913.{json,txt}
用法：python fingerprint_overlap_full_20260913.py
"""
import os
import glob
import json
import time
import hashlib
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = r"F:\datasets\TBX11K"
PROJ = os.path.dirname(os.path.abspath(__file__))


def fp(path):
    im = Image.open(path).convert("L").resize((32, 32), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    return hashlib.md5((a > a.mean()).astype(np.uint8).tobytes()).hexdigest()


def collect(split):
    d = defaultdict(list)
    fs = sorted(glob.glob(os.path.join(ROOT, split, "img", "*.png")))
    t0 = time.time()
    for i, p in enumerate(fs):
        d[fp(p)].append(os.path.basename(p))
        if (i + 1) % 2000 == 0:
            print(f"  {split}: {i+1}/{len(fs)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  {split}: done {len(fs)} images in {time.time()-t0:.0f}s", flush=True)
    return d, len(fs)


def main():
    rep = {"date": "2026-09-13", "script": os.path.basename(__file__),
           "protocol": "32x32 grayscale fingerprint (resize BILINEAR, binarised at image mean, md5 of bytes); "
                       "identical to audit_tbx11k.py part 4 but with NO sampling",
           "root": ROOT}
    tr, n_tr = collect("train")
    va, n_va = collect("val")
    inter = sorted(set(tr) & set(va))
    rep["n_train"] = n_tr
    rep["n_val"] = n_va
    rep["n_unique_fingerprints_train"] = len(tr)
    rep["n_unique_fingerprints_val"] = len(va)
    rep["n_within_train_groups"] = sum(1 for v in tr.values() if len(v) > 1)
    rep["n_within_val_groups"] = sum(1 for v in va.values() if len(v) > 1)
    rep["n_cross_split_overlap_groups"] = len(inter)
    rep["cross_split_examples"] = [{"fingerprint": h[:12], "train": tr[h][:3], "val": va[h][:3]}
                                   for h in inter[:10]]
    print(f"\n[train] {n_tr} 张 / 唯一指纹 {len(tr)} / 类内重复组 {rep['n_within_train_groups']}")
    print(f"[val]   {n_va} 张 / 唯一指纹 {len(va)} / 类内重复组 {rep['n_within_val_groups']}")
    print(f"[cross] train × val 跨划分重叠组 = {len(inter)}")
    for e in rep["cross_split_examples"]:
        print("   ", e)
    with open(os.path.join(PROJ, "output", "fingerprint_overlap_full_20260913.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(PROJ, "output", "fingerprint_overlap_full_20260913.txt"), "w",
              encoding="utf-8") as f:
        f.write("Full 32x32 fingerprint overlap check (no sampling), 2026-09-13\n")
        f.write(f"train {n_tr} images, {len(tr)} unique fingerprints, "
                f"{rep['n_within_train_groups']} within-train duplicate groups\n")
        f.write(f"val   {n_va} images, {len(va)} unique fingerprints, "
                f"{rep['n_within_val_groups']} within-val duplicate groups\n")
        f.write(f"cross-split (train vs val) overlap groups: {len(inter)}\n")
    print("\n[OK] wrote output/fingerprint_overlap_full_20260913.{json,txt}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Near-duplicate screening with 32x32 grayscale fingerprints over the full sets (train x val, no subsampling)

Background: part 4 of audit_tbx11k.py sampled only a subset of train to save time;
Section 4 of the manuscript states "Train/validation overlap was checked by content hashing and
32x32 fingerprints"; to make that statement checkable, fingerprints must be computed for **all** train (6,600) and val (1,800) images
image by image, and the cross-split overlapping groups counted.

The fingerprint definition is identical to audit_tbx11k.py: RGB->L, resize 32x32 BILINEAR,
binarise at the in-image mean, then take the md5 of the byte string.

Output: output/fingerprint_overlap_full.{json,txt}
Usage: python fingerprint_overlap_full.py
"""
import os
import glob
import json
import time
import hashlib
from collections import defaultdict

import numpy as np
from PIL import Image

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
ROOT = str(P.TBX11K)
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
    print(f"\n[train] {n_tr} images / unique fingerprints {len(tr)} / within-class duplicate groups {rep['n_within_train_groups']}")
    print(f"[val]   {n_va} images / unique fingerprints {len(va)} / within-class duplicate groups {rep['n_within_val_groups']}")
    print(f"[cross] train x val cross-split overlapping groups = {len(inter)}")
    for e in rep["cross_split_examples"]:
        print("   ", e)
    with open(os.path.join(PROJ, "output", "fingerprint_overlap_full.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(PROJ, "output", "fingerprint_overlap_full.txt"), "w",
              encoding="utf-8") as f:
        f.write("Full 32x32 fingerprint overlap check (no sampling), 2026-09-13\n")
        f.write(f"train {n_tr} images, {len(tr)} unique fingerprints, "
                f"{rep['n_within_train_groups']} within-train duplicate groups\n")
        f.write(f"val   {n_va} images, {len(va)} unique fingerprints, "
                f"{rep['n_within_val_groups']} within-val duplicate groups\n")
        f.write(f"cross-split (train vs val) overlap groups: {len(inter)}\n")
    print("\n[OK] wrote output/fingerprint_overlap_full.{json,txt}")


if __name__ == "__main__":
    main()

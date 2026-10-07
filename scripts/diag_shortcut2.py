# -*- coding: utf-8 -*-
"""逐类捷径诊断：每个库的 normal vs TB 分开比图像属性。"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
import os
import csv
import glob
import numpy as np
from PIL import Image
from collections import Counter


def stats(paths, n=200):
    if len(paths) > n:
        rng = np.random.RandomState(0)
        paths = [paths[i] for i in rng.choice(len(paths), n, replace=False)]
    modes, dims, means, fs = Counter(), Counter(), [], []
    for p in paths:
        try:
            im = Image.open(p)
            modes[im.mode] += 1
            dims[im.size] += 1
            means.append(np.asarray(im.convert("L"), dtype=np.float32).mean())
            fs.append(os.path.getsize(p))
        except Exception:
            pass
    return modes, dims, means, fs


def line(tag, paths):
    m, d, mu, fs = stats(paths)
    print(f"  {tag:26s} n={len(mu):4d} mode={dict(m)} "
          f"bright={np.mean(mu):6.1f} KB={np.mean(fs)/1024:5.0f} dims={dict(d)}")


print("=== TBX11K（healthy vs TB，用 manifest）===")
for split in ("train", "val"):
    man = {}
    with open(os.path.join(str(P.TBX11K), f"manifest_{split}.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            man[r["image"]] = r["tag"]
    pos, neg = [], []
    for img, tag in man.items():
        p = os.path.join(str(P.TBX11K), split, "img", img)
        if tag == "healthy":
            neg.append(p)
        elif tag in ("active_tb", "latent_tb", "active&latent_tb"):
            pos.append(p)
    line(f"TBX11K-{split}/healthy", neg)
    line(f"TBX11K-{split}/TB", pos)

print("\n=== Shenzhen（_0 normal / _1 TB）===")
base = os.path.join(str(P.TBPUB), "Shenzhen", "Shenzhen", "img")
n0 = [p for p in glob.glob(os.path.join(base, "*.png")) if p[:-4].endswith("_0")]
n1 = [p for p in glob.glob(os.path.join(base, "*.png")) if p[:-4].endswith("_1")]
line("Shenzhen/_0 normal", n0)
line("Shenzhen/_1 TB", n1)

print("\n=== Montgomery ===")
base = os.path.join(str(P.TBPUB), "Montgomery", "Montgomery", "img")
n0 = [p for p in glob.glob(os.path.join(base, "*.png")) if p[:-4].endswith("_0")]
n1 = [p for p in glob.glob(os.path.join(base, "*.png")) if p[:-4].endswith("_1")]
line("Montgomery/_0 normal", n0)
line("Montgomery/_1 TB", n1)

print("\n=== Qatar（已见差异，全量确认）===")
line("Qatar/Normal", glob.glob(os.path.join(str(P.QATAR), "Normal", "*")))
line("Qatar/TB", glob.glob(os.path.join(str(P.QATAR), "Tuberculosis", "*")))

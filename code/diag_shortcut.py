# -*- coding: utf-8 -*-
"""Diagnostic: whether image properties differ systematically between classes in each cohort (suspected shortcut learning / leakage)."""
import os
import glob
import numpy as np
from PIL import Image

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
OUT = str(P.RESULTS / "cross" / "matrix_summary.json")
SETS = {
    "Qatar": [(os.path.join(str(P.QATAR), "Normal"), "normal"),
              (os.path.join(str(P.QATAR), "Tuberculosis"), "TB")],
    "Shenzhen": [(os.path.join(str(P.TBPUB), "Shenzhen", "Shenzhen", "img"), "mix(_0/_1 by name)")],
    "Montgomery": [(os.path.join(str(P.TBPUB), "Montgomery", "Montgomery", "img"), "mix(_0/_1 by name)")],
    "TBX11K_train": [(os.path.join(str(P.TBX11K), "train", "img"), "mix")],
}


def stats_of(paths, n=150):
    if len(paths) > n:
        paths = paths[:n]
    sizes, means, fsize, modes = [], [], [], {}
    for p in paths:
        try:
            im = Image.open(p)
            sizes.append(im.size)
            modes[im.mode] = modes.get(im.mode, 0) + 1
            a = np.asarray(im.convert("L"), dtype=np.float32)
            means.append(a.mean())
            fsize.append(os.path.getsize(p))
        except Exception as e:
            print("  read failed", p, e)
    return sizes, means, fsize, modes


def summarize(tag, paths):
    sizes, means, fsize, modes = stats_of(paths)
    if not sizes:
        print(f"  {tag}: no images")
        return
    dims = {}
    for s in sizes:
        dims[s] = dims.get(s, 0) + 1
    top_dims = sorted(dims.items(), key=lambda x: -x[1])[:4]
    print(f"  {tag:22s} n={len(sizes):4d}  size distribution {top_dims}  "
          f"brightness mean={np.mean(means):.1f}  file KB={np.mean(fsize)/1024:.0f}  mode={modes}")


print("=== Cross-cohort matrix ===")
import json
print(json.dumps(json.load(open(OUT, encoding="utf-8")), ensure_ascii=False, indent=1))

print("\n=== Qatar class attribute comparison (main focus) ===")
for d, lab in SETS["Qatar"]:
    ps = sorted(glob.glob(os.path.join(d, "*")))
    ps = [p for p in ps if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    summarize(f"Qatar/{lab}", ps)

print("\n=== Other cohorts for reference ===")
for name in ["Shenzhen", "Montgomery", "TBX11K_train"]:
    d, _ = SETS[name][0]
    ps = sorted(glob.glob(os.path.join(d, "*.png")))
    summarize(name, ps)

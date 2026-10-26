# -*- coding: utf-8 -*-
"""Cross-cohort binary TB screening data loading (normal vs TB); four public cohorts share one label space.

Cohorts and labels:
  TBX11K      : healthy=0, TB(active_tb/latent_tb/active&latent)=1; sick_but_non-tb excluded (the other cohorts have no such class)
  Shenzhen    : filename _0=normal(0) / _1=TB(1)
  Montgomery  : filename _0=normal(0) / _1=TB(1)
  Qatar       : Normal/=0 / Tuberculosis/=1

Usage: from data_tbcohort import load_cohort; paths, labels = load_cohort("Shenzhen")
"""
import os
import csv
import glob

COHORTS = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
TBY = str(P.TBX11K)
TBPUB = str(P.TBPUB)
QATAR = str(P.QATAR)

# Filename prefix convention within a split (used for the internal train/test split)
TBX_TAG_POS = {"active_tb", "latent_tb", "active&latent_tb"}
TBX_TAG_NEG = {"healthy"}


def load_cohort(name):
    """Return (paths list, labels list), binary 0=normal/1=TB."""
    paths, labels = [], []
    if name == "TBX11K":
        # Merge the official train/val into the whole cohort and handle the split internally; this returns the full set, load_tbx_split is separate
        for split, tagset in [("train", None), ("val", None)]:
            man = os.path.join(TBY, f"manifest_{split}.csv")
            with open(man, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    t = r["tag"]
                    if t in TBX_TAG_POS:
                        lab = 1
                    elif t in TBX_TAG_NEG:
                        lab = 0
                    else:
                        continue  # sick_but_non-tb / NONE excluded
                    paths.append(os.path.join(TBY, split, "img", r["image"]))
                    labels.append(lab)
    elif name in ("Shenzhen", "Montgomery"):
        base = os.path.join(TBPUB, name, name, "img")
        for p in sorted(glob.glob(os.path.join(base, "*.png"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.endswith("_0"):
                lab = 0
            elif stem.endswith("_1"):
                lab = 1
            else:
                continue
            paths.append(p)
            labels.append(lab)
    elif name == "Qatar":
        for sub, lab in [("Normal", 0), ("Tuberculosis", 1)]:
            for p in sorted(glob.glob(os.path.join(QATAR, sub, "*"))):
                if p.lower().endswith((".png", ".jpg", ".jpeg")):
                    paths.append(p)
                    labels.append(lab)
    else:
        raise ValueError(name)
    return paths, labels


def load_tbx_split():
    """TBX11K official train/val kept separate (used as an in-cohort reference)."""
    out = {}
    for split in ("train", "val"):
        paths, labels = [], []
        man = os.path.join(TBY, f"manifest_{split}.csv")
        with open(man, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                t = r["tag"]
                if t in TBX_TAG_POS:
                    lab = 1
                elif t in TBX_TAG_NEG:
                    lab = 0
                else:
                    continue
                paths.append(os.path.join(TBY, split, "img", r["image"]))
                labels.append(lab)
        out[split] = (paths, labels)
    return out


if __name__ == "__main__":
    import numpy as np
    for c in COHORTS:
        p, y = load_cohort(c)
        y = np.array(y)
        print(f"{c:12s} n={len(y):5d}  normal={int((y==0).sum()):5d}  TB={int((y==1).sum()):5d}")
    s = load_tbx_split()
    for k, (p, y) in s.items():
        y = np.array(y)
        print(f"TBX11K-{k:5s} n={len(y):5d}  normal={int((y==0).sum()):5d}  TB={int((y==1).sum()):5d}")

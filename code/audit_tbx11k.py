# -*- coding: utf-8 -*-
"""TBX11K correctness audit: independent recomputation + label verification + near-duplicate leakage screening.

Four parts, all independent of the training code:
  1. Re-derive labels independently from the ann JSON and compare against the manifest CSV (look for mismatches)
  2. Does the npz validation label distribution match the expectation? Do the probs rows match the validation size?
  3. Recompute macroAUC / per-class independently and compare against results.json
  4. Near-duplicate screening: 32x32 grayscale fingerprint comparison of train/val images (broader than md5)
"""
import os
import csv
import json
import glob
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
ROOT = str(P.TBX11K)
PROJ = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
TAG2CLS = {"healthy": "healthy", "sick_but_non-tb": "sick_but_non-tb",
           "sick_but_non_tb": "sick_but_non-tb", "active_tb": "active_tb",
           "latent_tb": "latent_tb", "active&latent_tb": "active_tb"}
IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def part1_label_audit():
    print("=" * 70)
    print("1) Label verification: ann JSON re-derivation vs manifest CSV")
    print("=" * 70)
    for split in ("train", "val"):
        # Re-derive independently from the ann
        re_tags = {}
        for f in glob.glob(os.path.join(ROOT, split, "ann", "*.json")):
            d = json.load(open(f, encoding="utf-8"))
            tags = [t.get("name") for t in d.get("tags", [])]
            img = os.path.basename(f)[:-5]  # strip .json
            re_tags[img] = tags[0] if tags else "NONE"
        # manifest
        man = {}
        with open(os.path.join(ROOT, f"manifest_{split}.csv"), newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                man[r["image"]] = r["tag"]
        # Compare
        diff = [(k, re_tags.get(k), man.get(k)) for k in set(re_tags) | set(man)
                if re_tags.get(k) != man.get(k)]
        print(f"  {split}: ann count={len(re_tags)} manifest count={len(man)} mismatches={len(diff)}")
        for k, a, b in diff[:5]:
            print(f"    difference {k}: ann={a} manifest={b}")
        # Class distribution (after mapping)
        from collections import Counter
        cc = Counter(TAG2CLS.get(t, "DROP") for t in re_tags.values())
        print(f"    distribution after mapping: {dict(cc)}")


def part2_npz_check():
    print("\n" + "=" * 70)
    print("2) npz check: validation label distribution + row counts")
    print("=" * 70)
    files = sorted(glob.glob(os.path.join(PROJ, "output", "predictions", "baseline_s*.npz")))
    print(f"  prediction files: {len(files)} in total")
    d0 = np.load(files[0])
    labels = d0["labels"]
    probs = d0["probs"]
    print(f"  probs shape={probs.shape} labels shape={labels.shape}")
    dist = np.bincount(labels, minlength=4)
    print(f"  validation label distribution: {dict(zip(CLASS_NAMES, dist))}")
    print(f"  expected: healthy 800 / sick 800 / active 164 / latent 36  -> "
          f"{'✓ match' if list(dist)==[800,800,164,36] else '✗ mismatch!'}")
    # Whether the labels agree across all seeds
    same = all(np.array_equal(np.load(f)["labels"], labels) for f in files)
    print(f"  labels identical across 10 seeds: {same}")


def part3_recompute():
    print("\n" + "=" * 70)
    print("3) Independent recomputation of macroAUC / per-class vs results.json")
    print("=" * 70)

    def macro_auc(p, y):
        return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(4)))

    files = sorted(glob.glob(os.path.join(PROJ, "output", "predictions", "baseline_s*.npz")))
    data = [np.load(f) for f in files]
    recomputed = [macro_auc(d["probs"], d["labels"]) for d in data]
    res = json.load(open(os.path.join(PROJ, "output", "results.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(PROJ, "output", "results.json")) else []
    print(f"  per-seed recomputation: {[round(a,4) for a in recomputed]}")
    print(f"  recomputed mean={np.mean(recomputed):.4f}")
    if res:
        rec = [(r["model"], r["model_seed"], r["best_macro_auc"]) for r in res]
        print(f"  results.json records: {[(m,s,round(a,4)) for m,s,a in rec]}")
        # Compare (results.json holds the best epoch, the npz holds a re-evaluation of the best state -> should agree)
        ok = True
        for (m, s, a), rc in zip(sorted(rec, key=lambda x: x[1]), recomputed):
            if abs(a - rc) > 1e-6:
                ok = False
                print(f"    ✗ seed {s}: results={a:.6f} vs recomputed={rc:.6f}")
        print(f"  {'✓ all consistent' if ok else '✗ inconsistency (best epoch record vs npz re-evaluation)'}")


def part4_near_dup():
    print("\n" + "=" * 70)
    print("4) Near-duplicate screening: 32x32 grayscale fingerprints (train vs val)")
    print("=" * 70)
    def fp(path):
        im = Image.open(path).convert("L").resize((32, 32), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
        return (a > a.mean()).astype(np.uint8).tobytes()  # binary fingerprint
    def collect(split, cap=None):
        fs = sorted(glob.glob(os.path.join(ROOT, split, "img", "*.png")))
        if cap and len(fs) > cap:
            # Note: this function sorts by filename and then truncates, and the first N train images are all h*.png (healthy)
            # -- the sampling window contains no s*.png at all, so early versions falsely reported "0 overlap" (sampling bias, not absence of overlap).
            # Default cap=None means the full set; if subsampling, use random sampling and state it in the output.
            fs = fs[:cap]
        out = {}
        for p in fs:
            out.setdefault(fp(p), []).append(os.path.basename(p))
        return out
    tr = collect("train")           # full 6,600 (the sampling bias from cap=2000 in older versions has been fixed)
    va = collect("val")             # full 1,800
    n_tr = sum(len(v) for v in tr.values())
    n_va = sum(len(v) for v in va.values())
    inter = set(tr) & set(va)
    pref = {}
    for v in tr.values():
        for f in v:
            pref[f[0]] = pref.get(f[0], 0) + 1
    print(f"  train full set {n_tr} images (prefix distribution {pref}) / val full set {n_va} images: overlapping fingerprint groups = {len(inter)}")
    for k in list(inter)[:5]:
        print(f"    near duplicate: train{tr[k][:2]} ~ val{va[k][:2]}")


if __name__ == "__main__":
    part1_label_audit()
    part2_npz_check()
    part3_recompute()
    part4_near_dup()
    print("\n===== Audit complete =====")

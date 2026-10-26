# -*- coding: utf-8 -*-
"""Per-file MD5 lists + duplicate-image lists

Their request, verbatim: "a filename list of the 38 duplicate train/val images; per-file MD5 lists for the three databases (TBX11K train and val, Qatar
Normal and Tuberculosis, Shenzhen and Montgomery); and the script that generates these lists.
Lists only, no original images."

This script uses the same path sources as the pipeline (`data_tbcohort.load_cohort` and the TBX11K manifest_*.csv),
computes an MD5 for every file in the six sets and writes the lists to disk; it also provides:
  * the MD5 duplicate list for TBX11K train intersect val (the 38 images of Section 5.6, each with its filename + md5)
  * duplicate groups inside each set (3 groups within the Qatar TB class)
  * key assertions: the TBX11K train/val duplicate count and the number of duplicate groups inside the Qatar TB class (the run continues only if they match the manuscript)

Usage: python make_md5_lists.py
Output: md5_lists/{TBX11K_train,TBX11K_val,Qatar_Normal,Qatar_Tuberculosis,
      Shenzhen,Montgomery}.md5.txt + duplicates_summary.json + _INDEX.json
"""
import os
import csv
import json
import glob
import hashlib
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
from data_tbcohort import load_cohort  # noqa: E402

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
TBX_ROOT = str(P.TBX11K)
DST = str(P.MD5_LISTS)
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

    # TBX11K train intersect val (the 38 images of Section 5.6)
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

    # Duplicate groups inside each set
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
        "scripts": {"this": "make_md5_lists.py"},
        "sets": index,
        "tbx11k_train_val_duplicates": {
            "n": len(cross),
            "file": "TBX11K_train_val_duplicates.md5.txt",
            "pairs": [{"val": f, "md5": d, "train": t} for f, d, t in cross],
        },
        "internal_duplicates": internal,
    }
    with open(os.path.join(DST, "duplicates_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DST, "_INDEX.json"), "w", encoding="utf-8") as f:
        json.dump({"files": index, "tbx11k_train_val_duplicates_n": len(cross),
                   "internal_groups": {k: v["n_groups"] for k, v in internal.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {DST}/duplicates_summary.json "
          f"(train∩val duplicates = {len(cross)})", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Qatar / public-cohort profiling diagnostic (2026-09-13): re-check the two assertions of manuscript Section 5.5 and test whether within-class duplicates straddle the train/test split.

Output: results/cross/qatar_diagnostics.{json,txt}
Convention (copied from the manuscript and from train_cross_cohort.build_splits):
  . split: Normal first, Tuberculosis second (each sorted by filename) -> train_test_split(test_size=0.2, stratify=y, random_state=42)
  . duplicate test: byte-level MD5 of the file
  . channel test: after convert("RGB"), compare the three channels pixel by pixel (vectorised with numpy)
Read-only on the data; no original file is modified.
"""
import glob, hashlib, json, os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = "/mnt/g/Xray/tawsifurrahman"
TBPUB = "/mnt/f/datasets/TB_public"
EXTS = (".png", ".jpg", ".jpeg")
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
OUT_J = str(P.RESULTS / "cross" / "qatar_diagnostics.json")
OUT_T = str(P.RESULTS / "cross" / "qatar_diagnostics.txt")


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def qatar_paths():
    paths, labels = [], []
    for sub, lab in [("Normal", 0), ("Tuberculosis", 1)]:
        for p in sorted(glob.glob(os.path.join(ROOT, sub, "*"))):
            if p.lower().endswith(EXTS):
                paths.append(p)
                labels.append(lab)
    return np.array(paths), np.array(labels)


def pub_paths(name):
    base = os.path.join(TBPUB, name, name, "img")
    paths, labels = [], []
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
    return np.array(paths), np.array(labels)


def split_of(paths, labels):
    tr_p, te_p, tr_y, te_y = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=42)
    return ({os.path.basename(x) for x in tr_p}, {os.path.basename(x) for x in te_p},
            len(tr_p), len(te_p), int(tr_y.sum()), int(te_y.sum()))


rep = {"date": "2026-09-13", "script": "qatar_diagnostics.py",
       "purpose": "re-check the two assertions of Section 5.5 (0 MD5 cross-class duplicates, R=G=B) + whether within-class duplicates straddle the train/test split",
       "protocol": {"duplicate_rule": "MD5 of the raw file", "split": "train_test_split(test_size=0.2, stratify=y, random_state=42) on the sorted file list used by data_tbcohort.load_cohort",
                    "channel_rule": "np.asarray(im.convert('RGB')); require band0==band1==band2 for every pixel"}}

# ---------- Qatar ----------
q_p, q_y = qatar_paths()
tr, te, n_tr, n_te, pos_tr, pos_te = split_of(q_p, q_y)
h_by_class, modes, non_gray = {}, Counter(), []
for lab, sub in [(0, "Normal"), (1, "Tuberculosis")]:
    fs = [p for p, y in zip(q_p, q_y) if y == lab]
    h_by_class[sub] = [md5(p) for p in fs]
setN, setTB = set(h_by_class["Normal"]), set(h_by_class["Tuberculosis"])
inter = setN & setTB
within = {}
for sub in ("Normal", "Tuberculosis"):
    cnt = Counter(h_by_class[sub])
    dup_hashes = {h: c for h, c in cnt.items() if c > 1}
    groups = []
    for h in dup_hashes:
        names = [os.path.basename(p) for p, hh in zip([p for p, y in zip(q_p, q_y)
                 if y == (0 if sub == "Normal" else 1)], h_by_class[sub]) if hh == h]
        where = ["test" if n in te else ("train" if n in tr else "?") for n in names]
        groups.append({"files": names, "where": where, "straddles_split": len(set(where)) > 1})
    within[sub] = {"n_duplicate_hashes": len(dup_hashes), "groups": groups,
                   "n_groups_straddling_split": sum(g["straddles_split"] for g in groups)}
for p in q_p:
    with Image.open(p) as im:
        modes[im.mode] += 1
        arr = np.asarray(im.convert("RGB"))
    if not (np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2])):
        non_gray.append(os.path.basename(p))
rep["Qatar"] = {
    # 2026-09-13 correction: the key "n_files" used to hold the **unique MD5 count** (TB 700 files -> 697),
    #    which was misread as "there are only 697 files". It is now split into two keys; the file count uses the actual list length.
    "n_files_actual": {"Normal": len(h_by_class["Normal"]), "Tuberculosis": len(h_by_class["Tuberculosis"])},
    "n_unique_md5": {"Normal": len(setN), "Tuberculosis": len(setTB)},
    "n_files_note": "3 pairs among the 700 Tuberculosis files have identical content (3 pairs = 6 files), so the unique MD5 count is 697; total files 4,200",
    "split": {"n_train": n_tr, "n_test": n_te, "n_pos_train": pos_tr, "n_pos_test": pos_te},
    "md5_cross_class_identical": len(inter), "md5_within_class": within,
    "raw_modes": dict(modes), "raw_images_not_R_eq_G_eq_B": len(non_gray),
    "raw_not_gray_examples": non_gray[:5],
    "after_pipeline_conversion": "data_tbx11k/data_tbcohort apply .convert('L').convert('RGB') on load, so after conversion all images have R=G=B (guaranteed by construction)",
}

# ---------- Shenzhen / Montgomery (same checks) ----------
rep["shenzhen_montgomery"] = {}
for name in ("Shenzhen", "Montgomery"):
    p_, y_ = pub_paths(name)
    tr_, te_, *rest = split_of(p_, y_)
    d = defaultdict(list)
    for p in p_: d[md5(p)].append(os.path.basename(p))
    dups = {h: v for h, v in d.items() if len(v) > 1}
    cross = [{"files": v, "where": ["test" if n in te_ else "train" for n in v]}
             for v in dups.values() if len({"test" if n in te_ else "train" for n in v}) > 1]
    rep["shenzhen_montgomery"][name] = {"n": len(p_), "n_duplicate_hashes": len(dups),
                                        "n_groups_straddling_split": len(cross), "groups": cross[:5]}

rep["verdict"] = {
    "cross_class_identical_pairs_Qatar": len(inter),
    "claim_0_md5_overlaps_across_classes": "SUPPORTED" if len(inter) == 0 else "CONTRADICTED",
    "claim_channels_R_eq_G_eq_B_every_image": "TRUE BY CONSTRUCTION AFTER CONVERSION; raw files include non-gray images (see raw_images_not_R_eq_G_eq_B)",
    "new_finding": "Qatar TB class contains %d duplicate-file groups, %d of which straddle the train/test split"
                   % (within["Tuberculosis"]["n_duplicate_hashes"], within["Tuberculosis"]["n_groups_straddling_split"]),
}

os.makedirs(os.path.dirname(OUT_J), exist_ok=True)
with open(OUT_J, "w") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
with open(OUT_T, "w") as f:
    q = rep["Qatar"]
    f.write("DATASET DIAGNOSTICS 2026-09-13 (qatar_diagnostics.py)\n")
    f.write(f"Qatar: Normal {q['n_files_actual']['Normal']} / TB {q['n_files_actual']['Tuberculosis']} (unique MD5 {q['n_unique_md5']['Tuberculosis']})  split {q['split']}\n")
    f.write(f"MD5 cross-class identical: {q['md5_cross_class_identical']}  |  within-class duplicate hashes: Normal {within['Normal']['n_duplicate_hashes']}, TB {within['Tuberculosis']['n_duplicate_hashes']}\n")
    for g in within["Tuberculosis"]["groups"]:
        f.write(f"  TB dup group: {g['files']} -> {g['where']}  straddles={g['straddles_split']}\n")
    f.write(f"raw modes: {q['raw_modes']}  |  raw images with non-equal channels: {q['raw_images_not_R_eq_G_eq_B']}\n")
    for name, v in rep["shenzhen_montgomery"].items():
        f.write(f"{name}: n={v['n']} duplicate hashes {v['n_duplicate_hashes']} straddling split {v['n_groups_straddling_split']}\n")
print("wrote", OUT_J)
print(json.dumps(rep["verdict"], ensure_ascii=False, indent=2))

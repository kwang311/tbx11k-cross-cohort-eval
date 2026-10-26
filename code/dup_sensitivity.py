# -*- coding: utf-8 -*-
"""Quantify the sensitivity of TBX11K duplicate images (byte-identical across train and val) -- produce traceable artefacts for Section 5.6 of the main text.

Background: Section 5.6 of the main text states that "train/val contain 38 byte-identical images (all sick-but-non-TB);
after removal the ensemble macro-AUC goes 0.9893->0.9890 and accuracy 0.9606->0.9597". Until now these numbers had no artefact on disk.
This script reproduces the val order with the same construction as data_tbx11k.load_data, finds duplicates by MD5,
then drops the corresponding rows from output/predictions/*.npz and recomputes them, producing json/txt.

Convention (same as analyze_tbx11k.py / stats.py):
  macro-AUC = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=range(4))
  ensemble      = metrics computed after elementwise averaging of the 10 seed probabilities
No torch / PIL dependency; only hashlib + numpy + sklearn.

Usage:  python3 dup_sensitivity.py
       (Windows venv) point TBX_ROOT at your TBX11K directory, or set TBX_DATA_ROOT
"""
import csv
import glob
import hashlib
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

TBX_ROOT = os.environ.get("TBX_ROOT", "/mnt/f/datasets/TBX11K")
PROJ = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJ, "results", "dup_sensitivity")

CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
TAG2CLS = {
    "healthy": "healthy", "sick_but_non-tb": "sick_but_non-tb",
    "sick_but_non_tb": "sick_but_non-tb", "active_tb": "active_tb",
    "latent_tb": "latent_tb", "active&latent_tb": "active_tb",
}
CLS2IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def load_split(split):
    """Exactly the same construction as data_tbx11k.load_data: manifest row order -> paths + labels."""
    paths, labels, dropped = [], [], 0
    with open(os.path.join(TBX_ROOT, f"manifest_{split}.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cls = TAG2CLS.get(r["tag"])
            if cls is None:
                dropped += 1
                continue
            paths.append(os.path.join(TBX_ROOT, split, "img", r["image"]))
            labels.append(CLS2IDX[cls])
    return paths, np.array(labels, dtype=np.int64), dropped


def md5(p, buf=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def macro_auc(y, p):
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro",
                               labels=list(range(4))))


def main():
    res = {"script": os.path.basename(__file__), "root": TBX_ROOT,
           "protocol": {"macro_auc": 'roc_auc_score(multi_class="ovr", average="macro", labels=range(4))',
                        "ensemble": "10 seeds, elementwise mean of probs, then metric",
                        "dup_rule": "MD5 of the raw image file identical between val and train"}}

    va_paths, va_y, va_dropped = load_split("val")
    tr_paths, tr_y, tr_dropped = load_split("train")
    res["splits"] = {"val_n": len(va_paths), "train_n": len(tr_paths),
                     "val_dropped_no_label": va_dropped, "train_dropped_no_label": tr_dropped}
    res["val_class_dist"] = {CLASS_NAMES[i]: int((va_y == i).sum()) for i in range(4)}

    # ---- fail-fast 0: the val order/labels must match the npz element by element, otherwise the dropped indices mean nothing ----
    npz_files = sorted(glob.glob(os.path.join(PROJ, "output", "predictions", "baseline_s*.npz")))
    assert len(npz_files) == 10, f"expected 10 npz files, found {len(npz_files)}"
    npz_labels = np.load(npz_files[0])["labels"]
    assert np.array_equal(npz_labels, va_y), "the npz labels do not match the re-derivation from manifest_val.csv"
    for f in npz_files:
        assert np.array_equal(np.load(f)["labels"], va_y), f"labels mismatch: {f}"
    res["failfast_npz_labels_match_manifest"] = True

    # ---- MD5 ----
    print("hashing val...", flush=True)
    va_md5 = [md5(p) for p in va_paths]
    print("hashing train...", flush=True)
    tr_md5 = set(md5(p) for p in tr_paths)

    dup_idx = np.array([i for i, h in enumerate(va_md5) if h in tr_md5], dtype=int)
    res["n_duplicates"] = int(len(dup_idx))
    res["dup_val_indices"] = dup_idx.tolist()
    res["dup_class_dist"] = {CLASS_NAMES[i]: int((va_y[dup_idx] == i).sum()) for i in range(4)}
    res["dup_fraction_of_val"] = round(len(dup_idx) / len(va_y), 4)

    # ---- Two conventions: with and without duplicates ----
    keep = np.ones(len(va_y), dtype=bool)
    keep[dup_idx] = False
    all_probs = np.stack([np.load(f)["probs"] for f in npz_files])   # (10, N, 4)
    ens_all = all_probs.mean(axis=0)
    ens_keep = ens_all[keep]

    res["with_duplicates"] = {
        "n": int(len(va_y)),
        "ensemble_macro_auc": round(macro_auc(va_y, ens_all), 4),
        "ensemble_accuracy": round(float(accuracy_score(va_y, ens_all.argmax(1))), 4),
        "per_class_auc": {CLASS_NAMES[i]: round(float(roc_auc_score(va_y == i, ens_all[:, i])), 4)
                          for i in range(4)},
    }
    res["without_duplicates"] = {
        "n": int(keep.sum()),
        "ensemble_macro_auc": round(macro_auc(va_y[keep], ens_keep), 4),
        "ensemble_accuracy": round(float(accuracy_score(va_y[keep], ens_keep.argmax(1))), 4),
        "per_class_auc": {CLASS_NAMES[i]: round(float(roc_auc_score(va_y[keep] == i, ens_keep[:, i])), 4)
                          for i in range(4)},
    }

    # ---- fail-fast 1: the 0.9893 / 0.9606 of the main text must be reproduced before removal, otherwise this script uses a different convention ----
    exp_auc_all, exp_acc_all = 0.9893, 0.9606
    got_auc_all = res["with_duplicates"]["ensemble_macro_auc"]
    got_acc_all = res["with_duplicates"]["ensemble_accuracy"]
    res["failfast_reproduces_manuscript_ensemble"] = bool(
        abs(got_auc_all - exp_auc_all) <= 5e-5 and abs(got_acc_all - exp_acc_all) <= 5e-5)
    assert res["failfast_reproduces_manuscript_ensemble"], \
        f"ensemble before removal {got_auc_all}/{got_acc_all} != main text {exp_auc_all}/{exp_acc_all}"

    # ---- Automatic check of the Section 5.6 assertions (recorded, not raised) ----
    res["manuscript_section_5_6_claim"] = {
        "n_duplicates": 38,
        "all_sick_but_non_tb": True,
        "macro_auc_0.9893_to_0.9890": True,
        "accuracy_0.9606_to_0.9597": True,
        "sick_but_non_tb_auc_unchanged_to_4dp": True,
    }
    res["manuscript_section_5_6_verdict"] = {
        "n_duplicates": res["n_duplicates"] == 38,
        "all_sick_but_non_tb": res["dup_class_dist"]["sick_but_non-tb"] == res["n_duplicates"],
        "macro_auc_to_0.9890": res["without_duplicates"]["ensemble_macro_auc"] == 0.9890,
        "accuracy_to_0.9597": res["without_duplicates"]["ensemble_accuracy"] == 0.9597,
        "sick_auc_unchanged_to_4dp":
            res["without_duplicates"]["per_class_auc"]["sick_but_non-tb"]
            == res["with_duplicates"]["per_class_auc"]["sick_but_non-tb"],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    with open(OUT + ".txt", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(res, ensure_ascii=False, indent=2))

    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n=> {OUT}.json / .txt")
    bad = [k for k, v in res["manuscript_section_5_6_verdict"].items() if not v]
    print("Section 5.6 assertion check: " + ("all match ✓" if not bad else f"mismatches {bad}"))


if __name__ == "__main__":
    main()

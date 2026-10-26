# -*- coding: utf-8 -*-
"""Per-tag manifest counts (including the four-class distribution after mapping)

It is meant to independently verify two things:
 (1) that one unlabelled row out of 6,600 (tb1199.png) is dropped by the pipeline;
 (2) the four-class distribution (healthy 3,000 / sick-but-non-TB 3,000 / active 473 / latent 103 / active&latent 23 / unlabelled 1).

This script does not duplicate the pipeline logic; it **imports the pipeline mapping table directly** (`data_tbx11k.TAG2CLS`),
so the rules "NONE -> dropped, active&latent_tb -> active_tb" can be checked line by line by a third party.
The output provides the raw tag counts, the post-mapping counts, the list of dropped filenames, and the evidence for empty annotation files.

Usage: python count_manifest_tags.py
Output: results/cross/manifest_tag_counts.{json,txt}
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
import os
import csv
import json
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
import data_tbx11k as D  # noqa: E402  (only reads its TAG2CLS / ROOT, no training)

ROOT = D.ROOT
DST = str(P.RESULTS / "cross" / "manifest_tag_counts.json")


def read_manifest(split):
    fn = os.path.join(ROOT, f"manifest_{split}.csv")
    rows = []
    with open(fn, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return fn, rows


def main():
    out = {"note": "per-tag counts read from the released TBX11K manifests; mapping taken by "
                   "importing the pipeline's own table (data_tbx11k.TAG2CLS), so that the "
                   "NONE->drop and active&latent_tb->active_tb rules can be checked line by line",
           "root": ROOT,
           "tag2cls": D.TAG2CLS,
           "class_names": D.CLASS_NAMES,
           "splits": {}}
    for split in ("train", "val", "test", "all"):
        fn, rows = read_manifest(split)
        raw = Counter(r["tag"] for r in rows)
        mapped = Counter()
        dropped = []
        for r in rows:
            c = D.TAG2CLS.get(r["tag"])
            if c is None:
                dropped.append(r["image"])
            else:
                mapped[c] += 1
        out["splits"][split] = {
            "file": os.path.basename(fn),
            "n_rows": len(rows),
            "raw_tag_counts": dict(sorted(raw.items())),
            "mapped_class_counts": dict(sorted(mapped.items())),
            "n_dropped": len(dropped),
            "dropped_files": sorted(dropped)[:20],
        }
        print(f"{split:6s} rows={len(rows):5d} raw={dict(sorted(raw.items()))} "
              f"mapped={dict(sorted(mapped.items()))} dropped={len(dropped)}", flush=True)

    # Direct evidence for the dropped row: the manifest row + the ann JSON content
    ev = []
    for r in out["splits"]["train"]["dropped_files"]:
        stem = os.path.splitext(r)[0]
        ann = os.path.join(ROOT, "train", "ann", f"{r}.json")
        item = {"image": r, "ann_path": ann, "ann_exists": os.path.exists(ann)}
        if os.path.exists(ann):
            with open(ann, encoding="utf-8") as fh:
                item["ann_json"] = json.load(fh)
        ev.append(item)
    out["dropped_evidence"] = ev
    out["derived"] = {
        "train_four_class_total": sum(out["splits"]["train"]["mapped_class_counts"].values()),
        "val_four_class_total": sum(out["splits"]["val"]["mapped_class_counts"].values()),
        "train_manifest_rows": out["splits"]["train"]["n_rows"],
        "note": "train_four_class_total should equal manifest rows minus n_dropped",
    }
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(DST.replace(".json", ".txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[OK] wrote {DST} (+ .txt)", flush=True)
    print("derived:", out["derived"], flush=True)


if __name__ == "__main__":
    main()

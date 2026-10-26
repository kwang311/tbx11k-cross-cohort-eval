# -*- coding: utf-8 -*-
"""Parse TBX11K Supervisely annotations -> classification manifest + detection annotation counts.

Supervisely annotation JSON structure:
  { "tags": [{"name": "healthy|sick_but_non_tb|active_tb|latent_tb|active&latent_tb", ...}],
    "size": {...},
    "objects": [{"classTitle": "ActiveTuberculosis|ObsoletePulmonaryTuberculosis",
                 "points": {"exterior": [[x1,y1],[x2,y2],...], "interior": [...]}}, ...] }

Usage: python3 build_manifest.py
Output: /mnt/f/datasets/TBX11K/manifest_{split}.csv (split, image, n_objects, tag)
     plus the tag distribution and bbox object distribution for each split
"""
import os
import json
import csv
import glob
from collections import Counter

ROOT = "/mnt/f/datasets/TBX11K"
SPLITS = ["train", "val", "test"]


def parse_split(split):
    ann_dir = os.path.join(ROOT, split, "ann")
    files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    rows = []
    tag_counter = Counter()
    obj_counter = Counter()
    n_tag_multi = 0
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        tags = [t.get("name") for t in d.get("tags", [])]
        if len(tags) > 1:
            n_tag_multi += 1
        tag = tags[0] if tags else "NONE"
        tag_counter[tag] += 1
        objs = d.get("objects", [])
        for o in objs:
            obj_counter[o.get("classTitle", "?")] += 1
        img = os.path.basename(f)[:-len(".json")]
        rows.append({"split": split, "image": img, "n_objects": len(objs), "tag": tag})
    return rows, tag_counter, obj_counter, n_tag_multi


def main():
    all_rows = []
    for split in SPLITS:
        rows, tagc, objc, multi = parse_split(split)
        all_rows.extend(rows)
        print(f"\n===== {split}: {len(rows)} images =====")
        print(f"  image-level tag distribution: {dict(tagc)}")
        print(f"  bbox object distribution:   {dict(objc)}")
        print(f"  images with multiple tags: {multi}")
        out = os.path.join(ROOT, f"manifest_{split}.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["split", "image", "n_objects", "tag"])
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote: {out}")

    # Combined manifest (train+val for training/validation; test is unannotated)
    out = os.path.join(ROOT, "manifest_all.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "image", "n_objects", "tag"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote combined manifest: {out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""解析 TBX11K Supervisely 标注 → 分类清单 + 检测标注统计。

Supervisely ann JSON 结构：
  { "tags": [{"name": "healthy|sick_but_non_tb|active_tb|latent_tb|active&latent_tb", ...}],
    "size": {...},
    "objects": [{"classTitle": "ActiveTuberculosis|ObsoletePulmonaryTuberculosis",
                 "points": {"exterior": [[x1,y1],[x2,y2],...], "interior": [...]}}, ...] }

用法：python3 build_manifest.py
输出：/mnt/f/datasets/TBX11K/manifest_{split}.csv （split, image, n_objects, tag）
     以及各 split 的 tag 分布 + bbox 对象分布
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
        print(f"\n===== {split}: {len(rows)} 张 =====")
        print(f"  图片级 tag 分布: {dict(tagc)}")
        print(f"  bbox 对象分布:   {dict(objc)}")
        print(f"  多 tag 图:        {multi}")
        out = os.path.join(ROOT, f"manifest_{split}.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["split", "image", "n_objects", "tag"])
            w.writeheader()
            w.writerows(rows)
        print(f"  已写: {out}")

    # 合并清单（train+val 用于训练/验证；test 无标注）
    out = os.path.join(ROOT, "manifest_all.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "image", "n_objects", "tag"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n已写合并清单: {out}（{len(all_rows)} 行）")


if __name__ == "__main__":
    main()

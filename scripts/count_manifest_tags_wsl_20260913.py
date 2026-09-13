# -*- coding: utf-8 -*-
"""manifest 逐 tag 计数（含映射后四类分布）—— 回应 rocky30 材料需求单【需求 3a】

它要独立验证两件事：
 ① 「6,600 行中 1 行无标签（tb1199.png）被管线丢弃」；
 ② 四类分布（healthy 3,000 / sick-but-non-TB 3,000 / active 473 / latent 103 / active&latent 23 / 无标签 1）。

本脚本不复制管线逻辑，而是**直接 import 管线的映射表**（`data_tbx11k.TAG2CLS`），
这样“NONE → 丢弃、active&latent_tb → active_tb”的规则可被第三方逐行核对。
产物里同时给：原始 tag 计数、映射后计数、被丢弃的文件名清单、空标注文件的判定证据。

用法：python count_manifest_tags_wsl_20260913.py
产物：output_cross/manifest_tag_counts_wsl_20260913.{json,txt}
"""
import os
import csv
import json
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
import data_tbx11k as D  # noqa: E402  （只读它的 TAG2CLS / ROOT，不训练）

ROOT = D.ROOT
DST = os.path.join(BASE, "output_cross", "manifest_tag_counts_wsl_20260913.json")


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

    # 被丢弃那行的直接证据：manifest 行 + ann JSON 内容
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

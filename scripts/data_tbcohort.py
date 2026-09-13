# -*- coding: utf-8 -*-
"""跨库二元 TB 筛查数据加载（normal vs TB），4 个公共库统一标签空间。

库与标签：
  TBX11K      : healthy=0, TB(active_tb/latent_tb/active&latent)=1；sick_but_non-tb 排除（其他库无此类）
  Shenzhen    : 文件名 _0=normal(0) / _1=TB(1)
  Montgomery  : 文件名 _0=normal(0) / _1=TB(1)
  Qatar       : Normal/=0 / Tuberculosis/=1

用法：from data_tbcohort import load_cohort; paths, labels = load_cohort("Shenzhen")
"""
import os
import csv
import glob

COHORTS = ["TBX11K", "Shenzhen", "Montgomery", "Qatar"]
TBY = r"F:\datasets\TBX11K"
TBPUB = r"F:\datasets\TB_public"
QATAR = r"G:\Xray\tawsifurrahman"

# split 内文件前缀约定（用于 train/test 内部划分）
TBX_TAG_POS = {"active_tb", "latent_tb", "active&latent_tb"}
TBX_TAG_NEG = {"healthy"}


def load_cohort(name):
    """返回 (paths list, labels list)，二值 0=normal/1=TB。"""
    paths, labels = [], []
    if name == "TBX11K":
        # 用官方 train/val 合并为整个库，内部再按 split 处理；这里返回 full，另有 load_tbx_split
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
                        continue  # sick_but_non-tb / NONE 排除
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
    """TBX11K 官方 train/val 分开（用于同库参考）。"""
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

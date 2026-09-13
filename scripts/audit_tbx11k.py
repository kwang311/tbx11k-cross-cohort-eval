# -*- coding: utf-8 -*-
"""TBX11K 正确性审计：独立重算 + 标签复核 + 近重复泄漏排查。

分 4 部分，全部独立于训练代码：
  1. 从 ann JSON 独立重导标签，与 manifest CSV 对照（找不一致）
  2. npz 的 val 标签分布 == 期望？probs 行数 == val 大小？
  3. 独立重算 macroAUC / per-class，与 results.json 对照
  4. 近重复排查：train/val 图像 32×32 灰度指纹比对（比 md5 更宽）
"""
import os
import csv
import json
import glob
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = r"F:\datasets\TBX11K"          # Windows 路径（本脚本用 Windows venv python 跑）
PROJ = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
TAG2CLS = {"healthy": "healthy", "sick_but_non-tb": "sick_but_non-tb",
           "sick_but_non_tb": "sick_but_non-tb", "active_tb": "active_tb",
           "latent_tb": "latent_tb", "active&latent_tb": "active_tb"}
IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def part1_label_audit():
    print("=" * 70)
    print("1) 标签复核：ann JSON 独立重导 vs manifest CSV")
    print("=" * 70)
    for split in ("train", "val"):
        # 独立从 ann 重导
        re_tags = {}
        for f in glob.glob(os.path.join(ROOT, split, "ann", "*.json")):
            d = json.load(open(f, encoding="utf-8"))
            tags = [t.get("name") for t in d.get("tags", [])]
            img = os.path.basename(f)[:-5]  # 去 .json
            re_tags[img] = tags[0] if tags else "NONE"
        # manifest
        man = {}
        with open(os.path.join(ROOT, f"manifest_{split}.csv"), newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                man[r["image"]] = r["tag"]
        # 对照
        diff = [(k, re_tags.get(k), man.get(k)) for k in set(re_tags) | set(man)
                if re_tags.get(k) != man.get(k)]
        print(f"  {split}: ann 数={len(re_tags)} manifest 数={len(man)} 不一致={len(diff)}")
        for k, a, b in diff[:5]:
            print(f"    差异 {k}: ann={a} manifest={b}")
        # 类别分布（映射后）
        from collections import Counter
        cc = Counter(TAG2CLS.get(t, "DROP") for t in re_tags.values())
        print(f"    映射后分布: {dict(cc)}")


def part2_npz_check():
    print("\n" + "=" * 70)
    print("2) npz 检查：val 标签分布 + 行数")
    print("=" * 70)
    files = sorted(glob.glob(os.path.join(PROJ, "output", "predictions", "baseline_s*.npz")))
    print(f"  预测文件 {len(files)} 个")
    d0 = np.load(files[0])
    labels = d0["labels"]
    probs = d0["probs"]
    print(f"  probs shape={probs.shape} labels shape={labels.shape}")
    dist = np.bincount(labels, minlength=4)
    print(f"  val 标签分布: {dict(zip(CLASS_NAMES, dist))}")
    print(f"  期望: healthy 800 / sick 800 / active 164 / latent 36  → "
          f"{'✓ 匹配' if list(dist)==[800,800,164,36] else '✗ 不匹配!'}")
    # 所有 seed 的标签是否一致
    same = all(np.array_equal(np.load(f)["labels"], labels) for f in files)
    print(f"  10 个 seed 标签一致: {same}")


def part3_recompute():
    print("\n" + "=" * 70)
    print("3) 独立重算 macroAUC / per-class vs results.json")
    print("=" * 70)

    def macro_auc(p, y):
        return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(4)))

    files = sorted(glob.glob(os.path.join(PROJ, "output", "predictions", "baseline_s*.npz")))
    data = [np.load(f) for f in files]
    recomputed = [macro_auc(d["probs"], d["labels"]) for d in data]
    res = json.load(open(os.path.join(PROJ, "output", "results.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(PROJ, "output", "results.json")) else []
    print(f"  独立重算逐 seed: {[round(a,4) for a in recomputed]}")
    print(f"  重算 mean={np.mean(recomputed):.4f}")
    if res:
        rec = [(r["model"], r["model_seed"], r["best_macro_auc"]) for r in res]
        print(f"  results.json 记录: {[(m,s,round(a,4)) for m,s,a in rec]}")
        # 对照（results.json 存的是 best epoch，npz 存的是 best state 复评 → 应一致）
        ok = True
        for (m, s, a), rc in zip(sorted(rec, key=lambda x: x[1]), recomputed):
            if abs(a - rc) > 1e-6:
                ok = False
                print(f"    ✗ seed {s}: results={a:.6f} vs 重算={rc:.6f}")
        print(f"  {'✓ 全部一致' if ok else '✗ 存在不一致（best epoch 记录 vs npz 复评）'}")


def part4_near_dup():
    print("\n" + "=" * 70)
    print("4) 近重复排查：32×32 灰度指纹（train vs val）")
    print("=" * 70)
    def fp(path):
        im = Image.open(path).convert("L").resize((32, 32), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32)
        return (a > a.mean()).astype(np.uint8).tobytes()  # 二值指纹
    def collect(split, cap=None):
        fs = sorted(glob.glob(os.path.join(ROOT, split, "img", "*.png")))
        if cap and len(fs) > cap:
            # 注意：本函数按文件名排序后截断，而 train 的前 N 张全是 h*.png（healthy）
            # —— 采样区间内不含任何 s*.png，故 early 版本误报"0 重叠"（采样偏置，非无重叠）。
            # 默认 cap=None = 全量；如要抽样请用随机抽样并在输出里注明。
            fs = fs[:cap]
        out = {}
        for p in fs:
            out.setdefault(fp(p), []).append(os.path.basename(p))
        return out
    tr = collect("train")           # 全量 6,600（旧版 cap=2000 造成的采样偏置已修）
    va = collect("val")             # 全量 1,800
    n_tr = sum(len(v) for v in tr.values())
    n_va = sum(len(v) for v in va.values())
    inter = set(tr) & set(va)
    pref = {}
    for v in tr.values():
        for f in v:
            pref[f[0]] = pref.get(f[0], 0) + 1
    print(f"  train 全量 {n_tr} 张（前缀分布 {pref}）/ val 全量 {n_va} 张：指纹重叠组数 = {len(inter)}")
    for k in list(inter)[:5]:
        print(f"    近重复: train{tr[k][:2]} ~ val{va[k][:2]}")


if __name__ == "__main__":
    part1_label_audit()
    part2_npz_check()
    part3_recompute()
    part4_near_dup()
    print("\n===== 审计完成 =====")

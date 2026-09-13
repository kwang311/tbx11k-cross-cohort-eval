# -*- coding: utf-8 -*-
"""Qatar 类内重复图敏感性（2026-09-13，WSL/GPU 侧执行）

背景：Qatar 的 TB 类里有 3 组逐字节相同的图片，其中 2 组跨 train/test 划分
（qatar_diagnostics_20260913.json → md5_within_class.Tuberculosis）：
    组A: Tuberculosis-243(train) ≡ Tuberculosis-244(test)
    组B: Tuberculosis-346(train) ≡ Tuberculosis-348(train)
    组C: Tuberculosis-509(test)  ≡ Tuberculosis-510(train)

问题：这 2 张"孪生"（以及训练集内的重复）是否抬高了 Qatar 的同库 AUC（CNN 1.000）
      与跨库行（0.408 / 0.511 / 0.566）？

本脚本做两件事：
  第 1 部分（不重训，纯复算）：用已存的逐 seed 概率 `output_cross/pred_resnet50_full/s*_Qatar_to_Qatar.npz`，
      在"全测试集(840)"与"去重测试集(838，剔除 244/509)"上分别重算 AUC。
  第 2 部分（重训，GPU）：把 Qatar 训练集里的重复文件剔除（丢掉每组的重复成员，见下），
      保持测试集不变(840)，重训 Qatar→{TBX11K,Shenzhen,Montgomery,Qatar} × 5 seed，
      并在两个测试口径上评估。

剔除规则（只在训练集上动手，测试集不动以保持与已发表数字可比）：
    组A 丢 243（其孪生 244 在测试集 → 直接消除泄漏通道）
    组B 丢 348（组内两张都在训练集，丢一留一，使训练集无重复内容）
    组C 丢 510（其孪生 509 在测试集）
  → 训练集 3,360 → 3,357

产出：output_cross/dedup_qatar_20260913.{json,txt}
      output_cross/pred_dedup_qatar/s*_Qatar_to_*.npz
用法：python dedup_qatar_sensitivity_20260913.py [--epochs 15] [--seeds 42,43,44,45,46]
"""
import os
import sys
import json
import time
import glob
import hashlib
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

import train_cross_cohort_ext as T   # 复用已验证的 split/训练/预测（同一份代码路径）

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output_cross")
COHORTS = T.COHORTS
DUP_DROP = ["Tuberculosis-243.png", "Tuberculosis-348.png", "Tuberculosis-510.png"]
DUP_TEST = ["Tuberculosis-244.png", "Tuberculosis-509.png"]


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def find_duplicate_groups(paths, labels):
    """只在 TB 类内按 MD5 找重复组（口径与 qatar_diagnostics 一致）。"""
    tb = [(p, y) for p, y in zip(paths, labels) if y == 1]
    by = {}
    for p, _ in tb:
        by.setdefault(md5(p), []).append(os.path.basename(p))
    return {h: v for h, v in by.items() if len(v) > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    rep = {"date": "2026-09-13", "script": os.path.basename(__file__),
           "purpose": "Qatar 类内逐字节重复图对同库/跨库 AUC 的影响（第1部分复算 + 第2部分重训）",
           "drop_from_train": DUP_DROP, "drop_from_test": DUP_TEST}

    # ---------- split（与已发表运行同一代码路径） ----------
    print("=== building splits (full) ===", flush=True)
    splits = T.build_splits("full")
    q_tr_p, q_tr_y = list(splits["Qatar"]["train"][0]), list(splits["Qatar"]["train"][1])
    q_te_p, q_te_y = list(splits["Qatar"]["test"][0]), list(splits["Qatar"]["test"][1])
    print(f"Qatar split: train={len(q_tr_p)} (pos={sum(q_tr_y)}) test={len(q_te_p)} (pos={sum(q_te_y)})",
          flush=True)

    # ---------- 重复组复核 ----------
    groups = find_duplicate_groups(q_tr_p + q_te_p, q_tr_y + q_te_y)
    rep["duplicate_groups"] = {h[:12]: v for h, v in groups.items()}
    print("duplicate groups:", rep["duplicate_groups"], flush=True)
    assert len(groups) == 3, "expected 3 duplicate groups (see qatar_diagnostics_20260913.json)"

    te_names = [os.path.basename(p) for p in q_te_p]
    idx_drop_test = [te_names.index(n) for n in DUP_TEST if n in te_names]
    assert len(idx_drop_test) == 2, "both duplicate test twins must be present"
    print("test indices to drop:", idx_drop_test, flush=True)

    # ---------- 第 1 部分：已存概率的复算（不重训） ----------
    part1 = {}
    for seed in seeds:
        f = os.path.join(OUT, "pred_resnet50_full", f"s{seed}_Qatar_to_Qatar.npz")
        d = np.load(f)
        probs, labels = d["probs"], d["labels"]
        # 严格校验：npz 的标签顺序必须与 split 的测试顺序逐位一致
        assert np.array_equal(labels, np.asarray(q_te_y)), f"label order mismatch in {f}"
        keep = [i for i in range(len(labels)) if i not in idx_drop_test]
        part1[seed] = {
            "auc_full_test": float(roc_auc_score(labels, probs[:, 1])),
            "auc_dedup_test": float(roc_auc_score(labels[keep], probs[keep, 1])),
        }
        print(f"  [part1] seed {seed}: full={part1[seed]['auc_full_test']:.6f} "
              f"dedup={part1[seed]['auc_dedup_test']:.6f}", flush=True)
    ens = np.mean([np.load(os.path.join(OUT, "pred_resnet50_full", f"s{s}_Qatar_to_Qatar.npz"))["probs"]
                   for s in seeds], axis=0)
    lab = np.load(os.path.join(OUT, "pred_resnet50_full", f"s{seeds[0]}_Qatar_to_Qatar.npz"))["labels"]
    keep = [i for i in range(len(lab)) if i not in idx_drop_test]
    part1["ensemble"] = {"auc_full_test": float(roc_auc_score(lab, ens[:, 1])),
                         "auc_dedup_test": float(roc_auc_score(lab[keep], ens[keep, 1]))}
    rep["part1_reevaluate_existing"] = part1
    print(f"  [part1] ensemble: full={part1['ensemble']['auc_full_test']:.6f} "
          f"dedup={part1['ensemble']['auc_dedup_test']:.6f}", flush=True)

    # ---------- 第 2 部分：去重重训 ----------
    tr_keep = [(p, y) for p, y in zip(q_tr_p, q_tr_y)
               if os.path.basename(p) not in DUP_DROP]
    dropped = [(os.path.basename(p), int(y)) for p, y in zip(q_tr_p, q_tr_y)
               if os.path.basename(p) in DUP_DROP]
    assert len(dropped) == 3, f"expected 3 files dropped, got {dropped}"
    rep["part2_retrain"] = {"n_train_before": len(q_tr_p), "n_train_after": len(tr_keep),
                            "dropped": dropped,
                            "n_pos_before": int(sum(q_tr_y)),
                            "n_pos_after": int(sum(y for _, y in tr_keep))}
    print(f"\n=== part2 retrain: train {len(q_tr_p)} -> {len(tr_keep)} (dropped {dropped}) ===",
          flush=True)

    t0 = time.time()
    cache = {}
    for c in COHORTS:
        # 只有 Qatar 用去重后的训练集；其余目标库只用到 test
        if c == "Qatar":
            cache[c] = {"train": T.build_ram_cache([p for p, _ in tr_keep], [y for _, y in tr_keep]),
                        "test": T.build_ram_cache(q_te_p, q_te_y)}
        else:
            cache[c] = {"train": T.build_ram_cache(*splits[c]["train"]),
                        "test": T.build_ram_cache(*splits[c]["test"])}
        print(f"  cache {c}: train={len(cache[c]['train'][0])} test={len(cache[c]['test'][0])}",
              flush=True)
    print(f"  cache built in {time.time()-t0:.1f}s", flush=True)

    device = "cuda" if T.torch.cuda.is_available() else "cpu"
    print(f"  device={device}", flush=True)
    pred_dir = os.path.join(OUT, "pred_dedup_qatar")
    os.makedirs(pred_dir, exist_ok=True)
    results = {}
    for seed in seeds:
        model = T.train_one(cache["Qatar"]["train"], seed, args.epochs, device, "resnet50")
        row = {}
        for tgt in COHORTS:
            probs, labels = T.predict(model, cache[tgt]["test"], device)
            row[tgt] = {"auc": float(roc_auc_score(labels, probs[:, 1])), "n_test": int(len(labels))}
            if tgt == "Qatar":
                k = [i for i in range(len(labels)) if i not in idx_drop_test]
                row[tgt]["auc_dedup_test"] = float(roc_auc_score(labels[k], probs[k, 1]))
                row[tgt]["n_test_dedup"] = int(len(k))
            np.savez_compressed(os.path.join(pred_dir, f"s{seed}_Qatar_to_{tgt}.npz"),
                                probs=probs.astype(np.float32), labels=labels.astype(np.int64))
        results[str(seed)] = row
        print(f"  seed {seed}: " + "  ".join(f"{t}={row[t]['auc']:.3f}" for t in COHORTS), flush=True)
    rep["part2_retrain"]["per_seed"] = results

    summary = {}
    for tgt in COHORTS:
        vals = [results[str(s)][tgt]["auc"] for s in seeds]
        summary[tgt] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                        "per_seed": {str(s): results[str(s)][tgt]["auc"] for s in seeds}}
    vals = [results[str(s)]["Qatar"]["auc_dedup_test"] for s in seeds]
    summary["Qatar_dedup_test"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                                   "per_seed": {str(s): results[str(s)]["Qatar"]["auc_dedup_test"]
                                                for s in seeds}}
    rep["part2_retrain"]["summary"] = summary

    # ---------- 对照表 ----------
    base = json.load(open(os.path.join(OUT, "matrix_summary_resnet50_full.json"), encoding="utf-8"))
    rep["comparison"] = {
        "published_qatar_row": {t: base["Qatar"][t]["mean"] for t in COHORTS},
        "retrained_dedup_qatar_row": {t: summary[t]["mean"] for t in COHORTS},
        "delta": {t: summary[t]["mean"] - base["Qatar"][t]["mean"] for t in COHORTS},
        "published_qatar_same_cohort_test_full": base["Qatar"]["Qatar"]["mean"],
        "existing_model_on_dedup_test_ensemble": part1["ensemble"]["auc_dedup_test"],
        "retrained_on_full_test": summary["Qatar"]["mean"],
        "retrained_on_dedup_test": summary["Qatar_dedup_test"]["mean"],
    }
    print("\n=== comparison ===", flush=True)
    print(json.dumps(rep["comparison"], indent=2), flush=True)

    with open(os.path.join(OUT, "dedup_qatar_20260913.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "dedup_qatar_20260913.txt"), "w", encoding="utf-8") as f:
        f.write("Qatar duplicate-image sensitivity (2026-09-13)\n")
        f.write(f"dropped from train: {DUP_DROP}\n")
        f.write("part1 (existing models, no retrain):\n")
        for s in seeds:
            f.write(f"  seed {s}: full={part1[s]['auc_full_test']:.6f} "
                    f"dedup={part1[s]['auc_dedup_test']:.6f}\n")
        f.write(f"  ensemble: full={part1['ensemble']['auc_full_test']:.6f} "
                f"dedup={part1['ensemble']['auc_dedup_test']:.6f}\n")
        f.write("part2 (retrained without duplicate training images):\n")
        for t in COHORTS:
            f.write(f"  Qatar->{t:11s} {summary[t]['mean']:.4f} +/- {summary[t]['std']:.4f}\n")
        f.write(f"  Qatar->Qatar (dedup test): {summary['Qatar_dedup_test']['mean']:.4f} "
                f"+/- {summary['Qatar_dedup_test']['std']:.4f}\n")
        f.write("comparison:\n" + json.dumps(rep["comparison"], indent=2) + "\n")
    print("\n[OK] wrote dedup_qatar_20260913.json/.txt and pred_dedup_qatar/*.npz", flush=True)


if __name__ == "__main__":
    main()

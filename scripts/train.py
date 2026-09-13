"""单标签 9 分类训练脚本（多 seed）。

正确实现：backbone 后插 1 个增强模块 + 冻结前层（conv1/bn1/layer1/layer2）。
CE loss（类逆频率加权）+ macro-AUC 评估，保存 val 预测 npz 供统计检验。
"""
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

from models import Chest9Classifier, MODEL_TYPES
from data_tbx11k import (load_data, build_ram_cache, RamDataset,
                         NUM_CLASSES, CLASS_NAMES)

_TASK = os.environ.get("TBX_TASK", "4class")
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "output" if _TASK == "4class" else f"output_{_TASK}")


def evaluate(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for images, lbl in loader:
            images, lbl = images.to(device), lbl.to(device)
            logits = model(images)
            val_loss += criterion(logits, lbl).item()
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(lbl.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    equal_weight = roc_auc_score(labels, probs, multi_class="ovr",
                                 average="macro", labels=list(range(NUM_CLASSES))) \
        if NUM_CLASSES > 2 else roc_auc_score(labels, probs[:, 1])
    macro_auc = equal_weight
    weighted_auc = roc_auc_score(labels, probs, multi_class="ovr",
                                 average="weighted", labels=list(range(NUM_CLASSES))) \
        if NUM_CLASSES > 2 else equal_weight
    per_class = {}
    for c in range(NUM_CLASSES):
        y_true_c = (labels == c).astype(int)
        if y_true_c.sum() > 0 and (1 - y_true_c).sum() > 0:
            per_class[CLASS_NAMES[c]] = float(roc_auc_score(y_true_c, probs[:, c]))
        else:
            per_class[CLASS_NAMES[c]] = float("nan")
    return macro_auc, weighted_auc, acc, per_class, val_loss / max(1, len(loader)), probs, labels


def train_one_model(model_type, train_cache, val_cache, batch_size=32,
                    max_epochs=15, lr=1e-4, weight_decay=1e-4, model_seed=42,
                    patience=5, device="cuda", save_ckpt=False):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    train_images, train_labels = train_cache
    val_images, val_labels = val_cache
    train_ds = RamDataset(train_images, train_labels, train=True)
    val_ds = RamDataset(val_images, val_labels, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float32)
    total = counts.sum()
    class_weights = torch.tensor(total / (NUM_CLASSES * counts),
                                 dtype=torch.float32, device=device)
    print(f"[{model_type}] class_weights={np.round(class_weights.cpu().numpy(), 2).tolist()}")

    model = Chest9Classifier(model_type, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 冻结前层后，只优化 requires_grad=True 的参数
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_macro, best_state, no_improve = 0.0, None, 0
    best_info = None

    for epoch in range(max_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if torch.isnan(loss):
                print(f"  [E{epoch+1}] NaN loss！跳过 batch")
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))
        scheduler.step()

        macro_auc, weighted_auc, acc, per_class, val_loss, _, _ = evaluate(
            model, val_loader, criterion, device)
        print(f"  [E{epoch+1}/{max_epochs}] loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"macroAUC={macro_auc:.4f} wAUC={weighted_auc:.4f} acc={acc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        if macro_auc > best_macro:
            best_macro = macro_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_info = {"epoch": epoch + 1, "macro_auc": macro_auc,
                         "weighted_auc": weighted_auc, "acc": acc, "per_class": per_class}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stopping @ epoch {epoch+1}")
                break

    os.makedirs(OUT_ROOT, exist_ok=True)
    if save_ckpt and best_state is not None:
        torch.save(best_state, os.path.join(OUT_ROOT, f"{model_type}_s{model_seed}_best.pth"))

    result = {"model": model_type, "model_seed": model_seed,
              "epochs_run": epoch + 1, "best_macro_auc": float(best_macro)}
    if best_info is not None:
        result.update({k: (float(v) if isinstance(v, (int, float)) else v)
                       for k, v in best_info.items()})
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        _, _, _, _, _, probs, labels = evaluate(model, val_loader, criterion, device)
        pred_dir = os.path.join(OUT_ROOT, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        np.savez(os.path.join(pred_dir, f"{model_type}_s{model_seed}.npz"),
                 probs=probs, labels=labels)

    print(f"\n[{model_type}] s{model_seed} 完成。best_macro_auc={best_macro:.4f}\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--quick", action="store_true", help="冒烟：每类 20 张 × 1 epoch")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-per-class-train", type=int, default=1200)
    ap.add_argument("--max-per-class-val", type=int, default=300)
    ap.add_argument("--data-seed", type=int, default=42, help="固定 split+抽样")
    ap.add_argument("--seeds", default="42", help="模型初始化 seed 列表，逗号分隔")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--device", default=None, help="强制 cpu 或 cuda（默认 auto）")
    args = ap.parse_args()

    if args.quick:
        args.max_per_class_train = 20
        args.max_per_class_val = 20
        args.epochs = 1
        args.seeds = "42"

    model_seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs(OUT_ROOT, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  "
          f"{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    tr_paths, tr_y = load_data("train")
    va_paths, va_y = load_data("val")
    if args.quick:  # 冒烟：每类抽 cap（复用 max_per_class 参数）
        def _cap(paths, ys, cap):
            rng = np.random.RandomState(42)
            keep = []
            for c in range(NUM_CLASSES):
                idx = np.where(ys == c)[0]
                if len(idx) > cap:
                    idx = rng.choice(idx, cap, replace=False)
                keep.append(idx)
            return np.sort(np.concatenate(keep))
        ti = _cap(tr_paths, tr_y, args.max_per_class_train)
        vi = _cap(va_paths, va_y, args.max_per_class_val)
        tr_paths, tr_y = tr_paths[ti], tr_y[ti]
        va_paths, va_y = va_paths[vi], va_y[vi]

    t_cache = time.time()
    print("读图缓存 ...", flush=True)
    train_cache = build_ram_cache(tr_paths, tr_y)
    val_cache = build_ram_cache(va_paths, va_y)
    print(f"  缓存就绪: train={len(train_cache[0])}, val={len(val_cache[0])}  "
          f"耗时 {time.time()-t_cache:.1f}s", flush=True)

    models_to_run = [args.model] if args.model else MODEL_TYPES
    # 续训：读取已有 results.json，跳过已完成的 (model, seed) 组合
    results_path = os.path.join(OUT_ROOT, "results.json")
    all_results = []
    done = set()
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                _prev = json.load(f)
            for _d in _prev:
                if isinstance(_d, dict) and "model" in _d and "model_seed" in _d:
                    done.add((_d["model"], int(_d["model_seed"])))
                    all_results.append(_d)
            print(f"  续训：已有 {len(done)} 个 (model,seed) 已完成，将跳过", flush=True)
        except Exception as _e:
            print(f"  警告：读取 results.json 失败，视为从头训练 ({_e})", flush=True)
    for ms in model_seeds:
        for mt in models_to_run:
            if (mt, ms) in done:
                print(f"  [跳过] {mt} s{ms} 已完成", flush=True)
                continue
            r = train_one_model(mt, train_cache, val_cache,
                                batch_size=args.batch_size, max_epochs=args.epochs,
                                model_seed=ms, device=device, save_ckpt=args.save_ckpt)
            all_results.append(r)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n===== 汇总 =====")
    for r in all_results:
        print(f"  {r['model']:8s} s{r['model_seed']}  macroAUC={r['best_macro_auc']:.4f}  "
              f"wAUC={r.get('weighted_auc', float('nan')):.4f}  acc={r.get('acc', float('nan')):.4f}")


if __name__ == "__main__":
    main()

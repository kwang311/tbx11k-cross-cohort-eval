# -*- coding: utf-8 -*-
"""TBX11K 4 类分类数据加载（TB 亚型：healthy / sick_but_non-tb / active_tb / latent_tb）。

数据：F:\\datasets\\TBX11K\\{train,val}\\{img,ann}（Supervisely 格式，512×512）
标签：来自 ann JSON 的 tags[].name（已解析为 manifest_{split}.csv）
类别映射：active&latent_tb → active_tb（含活动 TB，并入）；NONE → 丢弃
划分：用 TBX11K **官方固定** train/val（不改动，保证与 benchmark 可比）
"""
import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = r"F:\datasets\TBX11K"

# 任务选择：TBX_TASK=4class（默认，4 类亚型）或 a_vs_l（仅 active vs latent 二分类）
TASK = os.environ.get("TBX_TASK", "4class")

if TASK == "4class":
    CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
    TAG2CLS = {
        "healthy": "healthy",
        "sick_but_non-tb": "sick_but_non-tb",
        "sick_but_non_tb": "sick_but_non-tb",   # 兼容两种写法
        "active_tb": "active_tb",
        "latent_tb": "latent_tb",
        "active&latent_tb": "active_tb",
    }
elif TASK == "a_vs_l":
    CLASS_NAMES = ["active_tb", "latent_tb"]
    TAG2CLS = {
        "active_tb": "active_tb",
        "active&latent_tb": "active_tb",
        "latent_tb": "latent_tb",
    }
else:
    raise ValueError(f"未知 TBX_TASK: {TASK}")

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CACHE_SIZE = 256   # 读图缓存边长（原图 512）；训练随机裁 224
# 灰度模式（TBX_GRAY=1）：所有图转 L 再复制成 3 通道 → 消除"RGB vs 灰度"色度捷径
GRAYSCALE = os.environ.get("TBX_GRAY", "0") == "1"


def load_data(split="train"):
    """读 manifest_{split}.csv → 路径 + 标签（4 类）。"""
    man = os.path.join(ROOT, f"manifest_{split}.csv")
    paths, labels, dropped = [], [], 0
    with open(man, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cls = TAG2CLS.get(r["tag"])
            if cls is None:
                dropped += 1
                continue
            paths.append(os.path.join(ROOT, split, "img", r["image"]))
            labels.append(CLASS_TO_IDX[cls])
    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int64)
    dist = dict(zip(CLASS_NAMES, np.bincount(labels, minlength=NUM_CLASSES)))
    print(f"[{split}] {len(labels)} 张（丢弃无标签 {dropped}）类分布: {dist}")
    return paths, labels


def build_ram_cache(paths, labels, size=CACHE_SIZE):
    """读图 resize(size) → uint8 (N,size,size,3)；损坏图跳过。"""
    labels = np.asarray(labels)
    paths = list(paths)
    images = np.zeros((len(paths), size, size, 3), dtype=np.uint8)
    ok = np.ones(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        try:
            img = Image.open(p)
            if GRAYSCALE:
                img = img.convert("L").convert("RGB")  # 灰度→3通道(R=G=B)，去色度捷径
            else:
                img = img.convert("RGB")
            img = img.resize((size, size), Image.BILINEAR)
            images[i] = np.asarray(img, dtype=np.uint8)
        except Exception:
            ok[i] = False
    return images[ok], labels[ok]


class RamDataset(Dataset):
    def __init__(self, images, labels, train=True):
        self.images = images
        self.labels = labels
        self.train = train
        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = self.transform(self.images[idx])
        return t, torch.tensor(self.labels[idx], dtype=torch.long)


def get_loaders(batch_size=16, num_workers=0, size=CACHE_SIZE):
    tr_paths, tr_y = load_data("train")
    va_paths, va_y = load_data("val")
    tr_imgs, tr_lbls = build_ram_cache(tr_paths, tr_y, size)
    va_imgs, va_lbls = build_ram_cache(va_paths, va_y, size)
    train_ds = RamDataset(tr_imgs, tr_lbls, train=True)
    val_ds = RamDataset(va_imgs, va_lbls, train=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    # 自检：只读 3 条，验证通路
    paths, y = load_data("val")
    imgs, lbls = build_ram_cache(paths[:3], y[:3])
    print(f"缓存 3 张: imgs={imgs.shape} labels={lbls} 类别名={[CLASS_NAMES[i] for i in lbls]}")
    ds = RamDataset(imgs, lbls, train=True)
    t, l = ds[0]
    print(f"单条: tensor={tuple(t.shape)} label={l.item()}")
    print("TBX11K 数据自检通过")

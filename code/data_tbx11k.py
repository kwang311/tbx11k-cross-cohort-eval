# -*- coding: utf-8 -*-
"""TBX11K four-class classification data loading (TB subtypes: healthy / sick_but_non-tb / active_tb / latent_tb).

Data: $TBX_DATA_ROOT/TBX11K/{train,val}/{img,ann} (Supervisely format, 512x512)
Labels: tags[].name from the ann JSON (already parsed into manifest_{split}.csv)
Class mapping: active&latent_tb -> active_tb (contains active TB, merged in); NONE -> dropped
Split: the **official fixed** TBX11K train/val (left untouched so results stay comparable with the benchmark)
"""
import os
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as P
ROOT = str(P.TBX11K)

# Task selection: TBX_TASK=4class (default, four subtypes) or a_vs_l (active vs latent binary only)
TASK = os.environ.get("TBX_TASK", "4class")

if TASK == "4class":
    CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
    TAG2CLS = {
        "healthy": "healthy",
        "sick_but_non-tb": "sick_but_non-tb",
        "sick_but_non_tb": "sick_but_non-tb",   # both spellings accepted
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
    raise ValueError(f"unknown TBX_TASK: {TASK}")

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CACHE_SIZE = 256   # cache edge length for loaded images (originals are 512); training crops randomly to 224
# Grayscale mode (TBX_GRAY=1): convert every image to L then replicate to 3 channels -> removes the "RGB vs grayscale" chroma shortcut
GRAYSCALE = os.environ.get("TBX_GRAY", "0") == "1"


def load_data(split="train"):
    """Read manifest_{split}.csv -> paths + labels (4 classes)."""
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
    print(f"[{split}] {len(labels)} images (dropped unlabelled {dropped}) class distribution: {dist}")
    return paths, labels


def build_ram_cache(paths, labels, size=CACHE_SIZE):
    """Read images resize(size) -> uint8 (N,size,size,3); corrupt images are skipped."""
    labels = np.asarray(labels)
    paths = list(paths)
    images = np.zeros((len(paths), size, size, 3), dtype=np.uint8)
    ok = np.ones(len(paths), dtype=bool)
    for i, p in enumerate(paths):
        try:
            img = Image.open(p)
            if GRAYSCALE:
                img = img.convert("L").convert("RGB")  # grayscale -> 3 channels (R=G=B), removes the chroma shortcut
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
    # Self-check: read only 3 items to verify the pathway
    paths, y = load_data("val")
    imgs, lbls = build_ram_cache(paths[:3], y[:3])
    print(f"cached 3 images: imgs={imgs.shape} labels={lbls} class names={[CLASS_NAMES[i] for i in lbls]}")
    ds = RamDataset(imgs, lbls, train=True)
    t, l = ds[0]
    print(f"single item: tensor={tuple(t.shape)} label={l.item()}")
    print("TBX11K data self-check passed")

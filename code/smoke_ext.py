# -*- coding: utf-8 -*-
"""Smoke test for the add-on run script: backbone forward passes + TBX11K label counts."""
import numpy as np
import torch
import train_cross_cohort_ext as X

print("--- backbone forward check (CPU, 2x3x224x224) ---", flush=True)
for bb in ["resnet50", "resnet18", "efficientnet_b0"]:
    m = X.make_model(bb, 2)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    frozen = all(not p.requires_grad for p in m.stem.parameters()) if hasattr(m, "stem") \
        else all(not p.requires_grad for p in m.layer1.parameters())
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    assert out.shape == (2, 2), (bb, out.shape)
    assert frozen, f"{bb}: early stages not frozen"
    print(f"  {bb:17s} out={tuple(out.shape)} early_frozen={frozen} "
          f"trainable={n_tr:,}/{n_all:,}", flush=True)

print("--- TBX11K label counts per --tbset ---", flush=True)
for tbset in ["full", "active", "latent"]:
    s = X.load_tbx_split_tbset(tbset)
    for split in ("train", "val"):
        y = np.array(s[split][1])
        assert set(np.unique(y)) == {0, 1}, (tbset, split, np.unique(y))
        print(f"  tbset={tbset:7s} {split:5s} n={len(y):5d} normal={int((y==0).sum()):5d} "
              f"TB={int((y==1).sum()):4d}", flush=True)

print("--- cohort splits (quick=False, counts only) ---", flush=True)
sp = X.build_splits("full", quick=False)
print("[OK] smoke test passed", flush=True)

# -*- coding: utf-8 -*-
"""Diagnose the resnet18 cross-cohort crash (exit code 0xC0000409 on Windows).

Stages: (1) resnet18 forward+backward on random CUDA tensors with cuDNN on;
        (2) the same with cuDNN disabled;
        (3) one real training step on the TBX11K quick split;
        (4) resnet50 control, identical settings.
Each stage prints before and after, so the log shows exactly which call kills the process.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

import train_cross_cohort_ext as X
from data_tbx11k import build_ram_cache, RamDataset
from torch.utils.data import DataLoader

print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}",
      flush=True)


def step(model, x, y, opt, crit):
    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    opt.step()
    return float(loss)


def stage_random(bb, cudnn):
    torch.backends.cudnn.enabled = cudnn
    print(f"\n[stage] {bb} random-tensor fwd+bwd (cudnn={cudnn})", flush=True)
    torch.manual_seed(0)
    dev = "cuda"
    m = X.make_model(bb, 2).to(dev)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    for it in range(3):
        x = torch.randn(16, 3, 224, 224, device=dev)
        y = torch.randint(0, 2, (16,), device=dev)
        loss = step(m, x, y, opt, crit)
        print(f"  iter {it}: loss={loss:.4f}", flush=True)
    torch.cuda.synchronize()
    print(f"  [OK] {bb} cudnn={cudnn}", flush=True)


def stage_real(bb):
    print(f"\n[stage] {bb} real data (TBX11K quick split, 1 epoch)", flush=True)
    splits = X.build_splits("full", quick=True)
    tr = build_ram_cache(*splits["TBX11K"]["train"])
    print(f"  cache: {tr[0].shape}", flush=True)
    dev = "cuda"
    m = X.make_model(bb, 2).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)
    dl = DataLoader(RamDataset(tr[0], tr[1], train=True), batch_size=16, shuffle=True)
    for i, (im, lb) in enumerate(dl):
        loss = step(m, im.to(dev), lb.to(dev), opt, crit)
        print(f"  batch {i}: loss={loss:.4f}", flush=True)
    torch.cuda.synchronize()
    print(f"  [OK] {bb} real data", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "rand"):
        stage_random("resnet18", True)
    if which in ("all", "nocudnn"):
        stage_random("resnet18", False)
    if which in ("all", "real"):
        stage_real("resnet18")
    if which in ("all", "r50"):
        stage_random("resnet50", True)
    print("\n[ALL STAGES COMPLETED]", flush=True)

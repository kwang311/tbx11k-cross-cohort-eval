# -*- coding: utf-8 -*-
"""resnet18 崩溃最小复现/隔离（诊断用，不参与正式流程）。

用法（Windows venv python）：
  python diag_resnet18.py cpu
  python diag_resnet18.py cuda
"""
import sys
import traceback

import torch
import torchvision
from torchvision import models as tvm  # 与 train_cross_cohort_ext.py 一致
from torch import nn

print(f"[env] torch={torch.__version__} torchvision={torchvision.__version__} "
      f"cuda={torch.cuda.is_available()}", flush=True)

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"


class BackboneClassifier(nn.Module):
    """与 train_cross_cohort_ext.BackboneClassifier 的 resnet18 分支逐行相同。"""

    def __init__(self, backbone="resnet18", num_classes=2, feature_depth=512):
        super().__init__()
        print("[step] building backbone...", flush=True)
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        print("[step] weights loaded", flush=True)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3, self.layer4 = net.layer2, net.layer3, net.layer4
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(nn.Linear(512, feature_depth),
                                        nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.classifier = nn.Linear(feature_depth, num_classes)
        for mod in (self.stem, self.layer2):
            for p in mod.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.stem(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(self.projection(x))


try:
    m = BackboneClassifier().to(dev)
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    print(f"[step] model built on {dev}; trainable={n_tr:,} / total={n_all:,}", flush=True)

    x = torch.randn(4, 3, 224, 224).to(dev)
    y = torch.tensor([0, 1, 0, 1]).to(dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)

    m.train()
    out = m(x)
    print("[step] forward ok", tuple(out.shape), flush=True)
    loss = crit(out, y)
    loss.backward()
    opt.step()
    print(f"[step] backward+step ok; loss={float(loss):.4f}; "
          f"grad on classifier={m.classifier.weight.grad is not None}; "
          f"grad on frozen stem={any(p.grad is not None for p in m.stem.parameters())}",
          flush=True)
    print("RESULT=OK", flush=True)
except Exception:
    traceback.print_exc()
    print("RESULT=PY_EXCEPTION", flush=True)

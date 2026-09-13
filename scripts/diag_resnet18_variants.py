# -*- coding: utf-8 -*-
"""resnet18 CUDA backward 崩溃的逐因素隔离（每个变体在独立进程里跑）。

用法（Windows venv python）： python diag_resnet18_variants.py <variant>
  base      : 与 train_cross_cohort_ext.BackboneClassifier 完全一致（冻结 stem+layer2），cudnn 默认
  nocudnn   : 同上，但 torch.backends.cudnn.enabled = False
  unfrozen  : 不冻结任何层
  noinplace : projection 的 ReLU(inplace=False)
  batch16   : 同 base，但 batch=16（贴近真实训练）
"""
import sys
import traceback

import torch
import torch.nn as nn
from torchvision import models as tvm

variant = sys.argv[1] if len(sys.argv) > 1 else "base"
if variant == "nocudnn":
    torch.backends.cudnn.enabled = False
bs = 16 if variant == "batch16" else 4
print(f"[env] torch={torch.__version__} cudnn={torch.backends.cudnn.version()} "
      f"enabled={torch.backends.cudnn.enabled} variant={variant} bs={bs}", flush=True)


class BC(nn.Module):
    def __init__(self):
        super().__init__()
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3, self.layer4 = net.layer2, net.layer3, net.layer4
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        inplace = (variant != "noinplace")
        self.projection = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=inplace), nn.Dropout(0.5))
        self.classifier = nn.Linear(512, 2)
        if variant != "unfrozen":
            for mod in (self.stem, self.layer2):
                for p in mod.parameters():
                    p.requires_grad = False

    def forward(self, x):
        x = self.layer4(self.layer3(self.layer2(self.stem(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(self.projection(x))


try:
    torch.manual_seed(42)
    m = BC().cuda()
    print("[step] built", flush=True)
    x = torch.randn(bs, 3, 224, 224, device="cuda")
    y = torch.randint(0, 2, (bs,), device="cuda")
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    m.train()
    out = m(x)
    torch.cuda.synchronize()
    print("[step] forward ok", flush=True)
    loss = crit(out, y)
    loss.backward()
    torch.cuda.synchronize()
    opt.step()
    torch.cuda.synchronize()
    print(f"[step] backward ok loss={float(loss):.4f}", flush=True)
    print(f"RESULT=OK variant={variant}", flush=True)
except Exception:
    traceback.print_exc()
    print(f"RESULT=PY_EXCEPTION variant={variant}", flush=True)

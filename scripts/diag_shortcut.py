# -*- coding: utf-8 -*-
"""诊断：各库类间图像属性是否有系统差异（捷径学习/泄漏嫌疑）。"""
import os
import glob
import numpy as np
from PIL import Image

OUT = r"G:\Xray\tbx11k_proj\output_cross\matrix_summary.json"
SETS = {
    "Qatar": [(r"G:\Xray\tawsifurrahman\Normal", "normal"),
              (r"G:\Xray\tawsifurrahman\Tuberculosis", "TB")],
    "Shenzhen": [(r"F:\datasets\TB_public\Shenzhen\Shenzhen\img", "mix(_0/_1按名)")],
    "Montgomery": [(r"F:\datasets\TB_public\Montgomery\Montgomery\img", "mix(_0/_1按名)")],
    "TBX11K_train": [(r"F:\datasets\TBX11K\train\img", "mix")],
}


def stats_of(paths, n=150):
    if len(paths) > n:
        paths = paths[:n]
    sizes, means, fsize, modes = [], [], [], {}
    for p in paths:
        try:
            im = Image.open(p)
            sizes.append(im.size)
            modes[im.mode] = modes.get(im.mode, 0) + 1
            a = np.asarray(im.convert("L"), dtype=np.float32)
            means.append(a.mean())
            fsize.append(os.path.getsize(p))
        except Exception as e:
            print("  读取失败", p, e)
    return sizes, means, fsize, modes


def summarize(tag, paths):
    sizes, means, fsize, modes = stats_of(paths)
    if not sizes:
        print(f"  {tag}: 无图")
        return
    dims = {}
    for s in sizes:
        dims[s] = dims.get(s, 0) + 1
    top_dims = sorted(dims.items(), key=lambda x: -x[1])[:4]
    print(f"  {tag:22s} n={len(sizes):4d}  尺寸分布{top_dims}  "
          f"亮度mean={np.mean(means):.1f}  文件KB={np.mean(fsize)/1024:.0f}  mode={modes}")


print("=== 跨库矩阵 ===")
import json
print(json.dumps(json.load(open(OUT, encoding="utf-8")), ensure_ascii=False, indent=1))

print("\n=== Qatar 类间属性对比（重点）===")
for d, lab in SETS["Qatar"]:
    ps = sorted(glob.glob(os.path.join(d, "*")))
    ps = [p for p in ps if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    summarize(f"Qatar/{lab}", ps)

print("\n=== 其他库参考 ===")
for name in ["Shenzhen", "Montgomery", "TBX11K_train"]:
    d, _ = SETS[name][0]
    ps = sorted(glob.glob(os.path.join(d, "*.png")))
    summarize(name, ps)

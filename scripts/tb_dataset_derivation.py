# -*- coding: utf-8 -*-
"""核查 9 类单标签子集 86,531 的真实推导（回应 R3-M1）。"""
import pandas as pd
import numpy as np

F = r"F:\archive"
DIS8 = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
NINECLASS = set(DIS8 + ["No Finding"])

df = pd.read_csv(rf"{F}\Data_Entry_2017.csv")
print(f"Data_Entry_2017 全量: {len(df)} 行，列={list(df.columns)}")
df["labels"] = df["Finding Labels"].apply(lambda s: [x.strip() for x in str(s).split("|")])
df["nlab"] = df["labels"].apply(len)
df["all_in_9"] = df["labels"].apply(lambda L: all(x in NINECLASS for x in L))

print(f"\n[1] 全量              : {len(df)}")
print(f"[2] 标签全 ∈ 9 类     : {int(df['all_in_9'].sum())}   <- 期望 = 107,010？")
d9 = df[df["all_in_9"]]
print(f"[3] 其中 单标签       : {int((d9['nlab'] == 1).sum())}   <- 期望 = 86,531？")
print(f"[4] 其中 多标签       : {int((d9['nlab'] > 1).sum())}   <- 期望 = 15,497？")

# 与两个 filtered csv 对齐
for nm, p in [("OriginalPreserved", rf"{F}\Filtered_Data_Entry_2017_OriginalPreserved.csv"),
              ("Filtered(86531)", rf"{F}\Filtered_Data_Entry_2017.csv")]:
    d = pd.read_csv(p)
    print(f"\n{nm}: {len(d)} 行，列={list(d.columns)[:8]}")
    idx = set(d.iloc[:, 0].astype(str))
    print(f"  首列样例: {list(idx)[:2]}")

# No Finding 类构成
nf = d9[d9["labels"].apply(lambda L: L == ["No Finding"])]
print(f"\n[5] No Finding 单标签: {len(nf)}")
# 单标签按类
from collections import Counter
c = Counter(L[0] for L in d9[d9["nlab"] == 1]["labels"])
print("[6] 单标签各类计数:")
for k, v in c.most_common():
    print(f"      {k:16s} {v}")

# -*- coding: utf-8 -*-
"""精确推导验证：107,010 / 15,497 / 86,531 / 4,982 的关系。"""
import pandas as pd

F = r"F:\archive"
DIS8 = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
NINE = set(DIS8 + ["No Finding"])

df = pd.read_csv(rf"{F}\Data_Entry_2017.csv")
df["L"] = df["Finding Labels"].apply(lambda s: [x.strip() for x in str(s).split("|")])
df["n9"] = df["L"].apply(lambda L: sum(x in NINE for x in L))      # 九类标签数
df["n_extra"] = df["L"].apply(lambda L: sum(x not in NINE for x in L))  # 额外6类标签数

tot = len(df)
ge1 = int((df["n9"] >= 1).sum())                      # 含≥1 个九类
exactly1 = int((df["n9"] == 1).sum())                 # 恰好 1 个九类
ge2 = int((df["n9"] >= 2).sum())                      # ≥2 个九类（多标签）
single_clean = int(((df["n9"] == 1) & (df["n_extra"] == 0)).sum())  # 恰1个九类 且 无额外

print(f"全量                     : {tot}")
print(f"含 >=1 个九类标签        : {ge1}   (应=107,010)")
print(f"恰好 1 个九类标签        : {exactly1}")
print(f">=2 个九类标签(多标签)   : {ge2}   (应=15,497)")
print(f"恰 1 个九类 且 无额外6类 : {single_clean}   (应=86,531)")
print(f"兜底校验: ge1 - ge2 = {ge1-ge2} (应=exactly1={exactly1})")
print(f"          exactly1 - single_clean = {exactly1 - single_clean}  (应=4,982 被排除)")

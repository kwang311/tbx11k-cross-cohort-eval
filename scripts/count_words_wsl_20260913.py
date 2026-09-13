#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正文词数计数脚本（口径写死）—— 回应 rocky30 材料需求单【需求 4a】

它的原话：要贵方「正文词数」的计数命令或脚本，以便统一口径。

口径（与我们对 CMPB 的报数一致）：
  * 只数正文第 1–7 节（1. Introduction … 7. Conclusion）；
  * **不含**各级标题、摘要、参考文献、表格（`|` 行）、图注（`![` 与 `**Figure`/`**Table` 行）、
    补充材料清单（`- **Supplementary`）与引用块（`>`）；
  * 分词规则：`[A-Za-z0-9][A-Za-z0-9'’×\\-\\./±]*` ——
    **连字符复合词算 1 个词**（`sick-but-non-TB`、`active-versus-latent`、`cross-cohort`…），
    `16×16` 算 1 个，`0.9893` 算 1 个。这就是与 Word/其他口径产生差异的地方。

用法：
  python3 count_words_wsl_20260913.py <manuscript.md> [more.md ...]
  python3 count_words_wsl_20260913.py --per-section <manuscript.md>
"""
import sys
import re

TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’×\-\./±]*")
MAIN = ["1. Introduction", "2. Related work", "3. Data", "4. Methods",
        "5. Results", "6. Discussion", "7. Conclusion"]
SKIP_PREFIX = ("|", "![", "**Table", "**Figure", ">", "- **Supplementary", "**【")


def split_sections(text):
    lines = text.split("\n")
    heads = [i for i, l in enumerate(lines) if l.startswith("## ")]
    names = {i: lines[i].strip("# ").strip() for i in heads}
    pos = {names[i]: k for k, i in enumerate(heads)}
    return lines, heads, names, pos


def count_region(lines, a, b, with_headings=False):
    n = 0
    for l in lines[a + 1:b]:
        s = l.strip()
        if not s or l.startswith("#") or s.startswith(SKIP_PREFIX):
            continue
        n += len(TOK.findall(s))
    if with_headings:
        n += len(TOK.findall(lines[a].strip("# ").strip()))
    return n


def main_count(text):
    lines, heads, names, pos = split_sections(text)
    tot = 0
    for m in MAIN:
        if m not in pos:
            continue
        k = pos[m]
        a = heads[k]
        b = heads[k + 1] if k + 1 < len(heads) else len(lines)
        tot += count_region(lines, a, b)
    return tot


def do_file(path, per_section=False):
    text = open(path, encoding="utf-8").read()
    lines, heads, names, pos = split_sections(text)
    print(f"=== {path}")
    if per_section:
        for m in MAIN:
            if m not in pos:
                continue
            k = pos[m]
            a, b = heads[k], (heads[k + 1] if k + 1 < len(heads) else len(lines))
            print(f"  {m:20s} {count_region(lines, a, b):5d}")
    tot = main_count(text)
    tot_h = sum(count_region(lines, heads[pos[m]],
                             heads[pos[m] + 1] if pos[m] + 1 < len(heads) else len(lines),
                             with_headings=True) for m in MAIN if m in pos)
    print(f"  {'sections 1-7 (no headings)':26s} {tot:5d}")
    print(f"  {'sections 1-7 + headings':26s} {tot_h:5d}")
    if "Abstract" in pos:
        k = pos["Abstract"]
        print(f"  {'abstract':26s} "
              f"{count_region(lines, heads[k], heads[k + 1]):5d}")
    return tot


if __name__ == "__main__":
    args = sys.argv[1:]
    per = "--per-section" in args
    args = [a for a in args if a != "--per-section"]
    if not args:
        print(__doc__)
        sys.exit(0)
    for p in args:
        do_file(p, per_section=per)

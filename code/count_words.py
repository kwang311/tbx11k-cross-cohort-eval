#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Word-count script for the main text (counting convention is fixed)

Their request, verbatim: they want the command or script used to count "main-text words", so that the convention can be unified.

Convention (matches the numbers reported in the manuscript):
  * Count only sections 1-7 of the main text (1. Introduction ... 7. Conclusion);
  * **Exclude** headings at every level, the abstract, references, tables (`|` lines), figure captions (`![` and `**Figure`/`**Table` lines),
    the supplementary item list (`- **Supplementary`) and block quotes (`>`);
  * Tokenisation rule: `[A-Za-z0-9][A-Za-z0-9'’×\\-\\./±]*`, where
    **a hyphenated compound counts as one word** (`sick-but-non-TB`, `active-versus-latent`, `cross-cohort` ...),
    `16×16` counts as one word and `0.9893` counts as one word. This is where differences from Word and other conventions arise.

Usage:
  python3 count_words.py <manuscript.md> [more.md ...]
  python3 count_words.py --per-section <manuscript.md>
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

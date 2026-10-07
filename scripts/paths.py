# -*- coding: utf-8 -*-
"""Central path resolution. Everything is relative to the repository root.

Layout (see README.md):

    predictions/      per-seed predicted probabilities (four_class/, a_vs_l/, cross_cohort/<prefix>/)
    results/          derived tables every number in the paper is read from
    features/         low-level feature matrices
    cross_pred_index/ per-image CSV exports
    md5_lists/        per-file checksums
    figures/, si/     exported figures and supplementary tables

Raw images are not redistributed with this repository. Point the scripts at your local
copies with environment variables:

    TBX_DATA_ROOT    directory containing TBX11K/ and TB_public/   (default: <repo>/data)
    TBX_QATAR_ROOT   Qatar TB chest X-ray set (tawsifurrahman)     (default: <TBX_DATA_ROOT>/tawsifurrahman)
    TBX_ROOT         repository root override                      (default: parent of scripts/)

Paper 1 leftovers (scripts/leakage_quant.py, tb_*derivation*.py, mde_tost_power.py,
perblock_sd.py, stats_equivalence.py, models.py) read data from <TBX_DATA_ROOT> as well.

Nothing outside these directories is hard-coded, so a clone plus the raw images is enough.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("TBX_ROOT") or Path(__file__).resolve().parent.parent).resolve()
DATA = Path(os.environ.get("TBX_DATA_ROOT") or ROOT / "data").resolve()
QATAR = Path(os.environ.get("TBX_QATAR_ROOT") or DATA / "tawsifurrahman").resolve()
TBX11K = DATA / "TBX11K"
TBPUB = DATA / "TB_public"
TB9 = Path(os.environ.get("TBX_TB9_ROOT") or DATA / "tb9_single_label").resolve()
TB9_PERBLOCK = Path(os.environ.get("TBX_TB9_PERBLOCK_ROOT") or DATA / "tb9_single_label_perblock").resolve()

PREDICTIONS = ROOT / "predictions"
RESULTS = ROOT / "results"
FEATURES = ROOT / "features"
FIGURES = ROOT / "figures"
CROSS_INDEX = ROOT / "cross_pred_index"
MD5_LISTS = ROOT / "md5_lists"

# Cross-cohort evaluation of chest-radiograph tuberculosis models

Code, per-seed predictions and derived result tables for the manuscript:

> **Beyond aggregate accuracy: cross-cohort generalization and active-versus-latent discrimination in chest-radiograph tuberculosis models** (submitted to *Computer Methods and Programs in Biomedicine*)

> **Status.** The manuscript is currently under peer review. The code, per-seed predictions and
> derived result tables in this repository are complete and stable; if you use them, please cite the
> associated article once it is published.

The study evaluates one fixed pipeline (ImageNet-pretrained ResNet-50, frozen early stages) on four
public chest-radiograph cohorts, and asks how much of a reported AUC (i) survives a change of cohort
and (ii) can be obtained **without resolving pulmonary pathology**, using a deliberately low-level
reference classifier (16×16 grayscale thumbnail + five global intensity statistics).

## What is in here

| Path | Contents |
|:--|:--|
| `scripts/` | Training / evaluation code. `train_cross_cohort_ext.py` produces the cross-cohort matrix; `train_cross_cohort.py` is the original 3-cohort version; `train_a_vs_l.bat` / `run_a_vs_l.bat` the active-versus-latent task. |
| `scripts/lowlevel_same_split.py` | The **low-level reference control** (thumbnail + intensity statistics, same splits as the CNN). |
| `scripts/lowlevel_nonlinear_control.py` | Non-parametric learners on the same features (random forest, gradient boosting, k-NN). |
| `scripts/audit_tbx11k.py` | Correctness audit: re-derives labels from the annotation files, recomputes all metrics from the stored predictions, and screens train/val image overlap. |
| `scripts/fingerprint_overlap_full_20260913.py`, `fingerprint_verify_20260913.py`, `dup_sensitivity_fingerprint_20260913.py` | Train/validation near-duplicate screening (32×32 fingerprint screen, 64×64 verification, pixel-level confirmation) and the duplicate-removal sensitivity analysis. |
| `scripts/stats_paper2.py`, `stats_cross_cells.py` | Bootstrap confidence intervals (2,000 resamples, image level, `RandomState(0)`) and per-cell CIs for the cross-cohort matrix. |
| `scripts/val_index_map_wsl_20260913.py` | Explicit validation row indices of the duplicate images (for independent recomputation). |
| `scripts/count_words_wsl_20260913.py` | The word-count script used for the journal's 3,500-word limit (see `README` note below). |
| `scripts/figures/` | Figure-generating scripts (matplotlib + PIL; QA checks for label overlap / clipping). |
| `predictions/` | **Per-seed predicted probabilities** (`probs`, `labels`) for every run: `four_class/` (10 seeds), `a_vs_l/` (10 seeds), `cross_cohort/<backbone>/<seed>_<source>_to_<target>.npz` (5 seeds). |
| `features/` | The low-level feature matrices (`thumb16` 256-d, `stats5`, `labels`, `filenames`) for both splits of all four cohorts — enough to reproduce the low-level control **without the images**. |
| `results/` | Every number cited in the manuscript, as JSON: `matrix_summary_*.json` / `matrix_raw_*.json`, `cells_ci_*.json`, `lowlevel_*.json`, `results.json`, `dup_sensitivity_*.json`, `fingerprint_*.json`, `paper2_stats_*.json`, … |
| `cross_pred_index/` | Per-image CSV exports of the cross-cohort predictions (file stem, true label, P(positive)), used for third-party verification. |
| `md5_lists/` | Per-file MD5 lists for all six image sets, and the train∩val duplicate list. |
| `figures/`, `si/` | Publication figures (PNG + PDF) and the Supplementary tables (Markdown). |

## Data

All four cohorts are **public** and were used as released. They are not redistributed here; see
`DATA.md` for sources, expected on-disk layout and the checksums of our copies.

| Cohort | Labels used | n (used) |
|:--|:--|:--|
| TBX11K | 4-class (healthy / sick-but-non-TB / active TB / latent TB) and binary | 6,599 train / 1,800 val |
| Shenzhen | binary normal vs TB | 566 (Mendeley lung-segmentation subset) |
| Montgomery | binary normal vs TB | 138 |
| Qatar | binary normal vs TB | 4,200 (3,500 / 700) |

## Environment

Python 3.11 (Windows) with CUDA 12.1:

```
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the results in the manuscript
(torch 2.3.1+cu121, torchvision 0.18.1+cu121, scikit-learn 1.7.2, numpy 1.26.2, Pillow 9.5.0).

All runs use `TBX_GRAY=1`, i.e. every image is converted to 8-bit grayscale before training, so that
color/encoding cannot act as a shortcut cue.

## Reproducing the main numbers

```bash
# (1) four-class TBX11K, 10 seeds
python train_tbx11k.py --seeds 42-51            # see run_*.bat for the exact invocations used

# (2) dedicated active-vs-latent model, 10 seeds
python train_a_vs_l.py --seeds 42-51

# (3) cross-cohort matrix, 5 seeds, all four backbones in the paper
set TBX_GRAY=1
python train_cross_cohort_ext.py --backbone resnet50        --tbset full   --npz 1 --prefix resnet50_full
python train_cross_cohort_ext.py --backbone resnet18        --tbset full   --npz 1 --prefix resnet18_full
python train_cross_cohort_ext.py --backbone efficientnet_b0 --tbset full   --npz 1 --prefix effnetb0_full
python train_cross_cohort_ext.py --backbone resnet50        --tbset active --npz 1 --prefix resnet50_active
python train_cross_cohort_ext.py --backbone resnet50        --tbset latent --npz 1 --prefix resnet50_latent

# (4) low-level reference control (same splits, no pathology-resolving information)
python lowlevel_same_split.py
python lowlevel_nonlinear_control.py

# (5) statistics, audit, duplicate screening
python stats_paper2.py
python stats_cross_cells.py --prefix resnet50_full
python audit_tbx11k.py
python fingerprint_overlap_full_20260913.py
python fingerprint_verify_20260913.py
python dup_sensitivity_fingerprint_20260913.py
```

Every script reads/writes inside `results/`-style directories; the data paths are collected at the
top of `data_tbx11k.py` and `data_tbcohort.py` and must be pointed at your local copies.

## Headline results (for orientation)

- Four-class TBX11K: macro-AUC **0.9876 ± 0.0012** (ten seeds), ensemble 0.9893 (95% CI 0.9859–0.9922);
  per class 0.9999 / 0.9994 / 0.9902 / 0.9675 (healthy / sick-but-non-TB / active / latent).
- Active-versus-latent restricted AUC: **0.6499** (95% CI 0.5620–0.7433); a dedicated two-class model
  reaches 0.6367 ± 0.0247.
- Same-cohort AUCs 1.000 (TBX11K, Qatar), 0.916 (Shenzhen), 0.861 (Montgomery); cross-cohort transfer
  drops to 0.548–0.703 (from TBX11K) and 0.408–0.566 (from Qatar).
- Low-level reference (thumbnail + five statistics): 0.977 / 0.859 / 0.672 / 0.976 for
  TBX11K / Shenzhen / Montgomery / Qatar; best of four non-parametric learners 0.997 / 0.859 / 0.875 / 0.996,
  i.e. at most 0.0031 (TBX11K) and 0.0044 (Qatar) AUC requires finer-than-thumbnail structure.

## Word-count convention

`scripts/count_words_wsl_20260913.py` implements the convention we used for the journal limit
(sections 1–7 only; headings, abstract, tables, figure legends and references excluded; hyphenated
compounds count as one word). It reports 3,339 words for the submitted main text.

## License / citation

- **MIT License** (see `LICENSE`).
- If you use this code or the derived tables, please cite the manuscript above.

## Contact

Kui Wang — School of Public Health, Shihezi University, Shihezi, Xinjiang, China.

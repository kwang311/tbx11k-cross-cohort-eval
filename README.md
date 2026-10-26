# Cross-cohort evaluation of chest-radiograph tuberculosis models

Code, per-seed predictions and derived result tables for the manuscript:

> **Cross-cohort generalizability of chest radiograph tuberculosis models: high same-cohort accuracy from coarse image statistics and poor transfer**

> **Status.** This repository accompanies a manuscript submitted for publication. The code, per-seed
> predictions and derived result tables are complete and stable; if you use them, please cite the
> associated article once it is published.

The study evaluates one fixed pipeline (ImageNet-pretrained ResNet-50, frozen early stages) on four
public chest-radiograph cohorts, and asks how much of a reported AUC (i) survives a change of cohort
and (ii) can be obtained **without resolving pulmonary pathology**, using a deliberately low-level
reference classifier (16×16 grayscale thumbnail + five global intensity statistics).

## What is in here

| Path | Contents |
|:--|:--|
| `code/` | Training / evaluation code. `train_cross_cohort_ext.py` produces the cross-cohort matrix; `train_cross_cohort.py` is the original 3-cohort version; `run_tbx11k.bat` / `run_a_vs_l.bat` give the exact command lines used for the four-class and active-versus-latent tasks. |
| `code/lowlevel_same_split.py` | The **low-level reference control** (thumbnail + intensity statistics, same splits as the CNN). |
| `code/lowlevel_nonlinear_control.py` | Non-parametric learners on the same features (random forest, gradient boosting, k-NN). |
| `code/audit_tbx11k.py` | Correctness audit: re-derives labels from the annotation files, recomputes all metrics from the stored predictions, and screens train/val image overlap. |
| `code/fingerprint_overlap_full.py`, `fingerprint_verify.py`, `dup_sensitivity_fingerprint.py` | Train/validation near-duplicate screening (32×32 fingerprint screen, 64×64 verification, pixel-level confirmation) and the duplicate-removal sensitivity analysis. |
| `code/stats_paper2.py`, `stats_cross_cells.py` | Entry points for the statistics artefacts (see *Function layer* below). |
| `code/p2_lib.py` | Shared numeric layer used by both statistics entry points: prediction loading, AUC variants, percentile bootstrap, per-seed dispersion and artefact writers. |
| `code/val_index_map.py` | Explicit validation row indices of the duplicate images (for independent recomputation). |
| `code/count_words.py` | The word-count script used for the journal's 3,500-word limit (see `README` note below). |
| `code/figures/` | Figure-generating scripts (matplotlib + PIL; QA checks for label overlap / clipping). |
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

All cross-cohort runs use `TBX_GRAY=1`, i.e. every image is converted to 8-bit grayscale before those
runs, so that color/encoding cannot act as a shortcut cue. The four-class and active-versus-latent
runs use the images as released, as stated in the manuscript.

## Reproducing the main numbers

All commands are run from the repository root (`cd <repo>`).

```bash
# (1) four-class TBX11K, 10 seeds   (see run_tbx11k.bat for the exact invocation)
python code/train.py --model baseline --epochs 15 --batch-size 16 --seeds 42,43,44,45,46,47,48,49,50,51

# (2) dedicated active-vs-latent model, 10 seeds   (see run_a_vs_l.bat)
TBX_TASK=a_vs_l python code/train.py --model baseline --epochs 15 --batch-size 16 --seeds 42,43,44,45,46,47,48,49,50,51

# (3) cross-cohort matrix, 5 seeds, all four backbones in the paper   (see run_cross_gray.bat)
TBX_GRAY=1 python code/train_cross_cohort_ext.py --backbone resnet50        --tbset full   --npz 1 --prefix resnet50_full
TBX_GRAY=1 python code/train_cross_cohort_ext.py --backbone resnet18        --tbset full   --npz 1 --prefix resnet18_full
TBX_GRAY=1 python code/train_cross_cohort_ext.py --backbone efficientnet_b0 --tbset full   --npz 1 --prefix effnetb0_full
TBX_GRAY=1 python code/train_cross_cohort_ext.py --backbone resnet50        --tbset active --npz 1 --prefix resnet50_active
TBX_GRAY=1 python code/train_cross_cohort_ext.py --backbone resnet50        --tbset latent --npz 1 --prefix resnet50_latent

# (4) low-level reference control (same splits, no pathology-resolving information)
TBX_GRAY=1 python code/lowlevel_same_split.py
TBX_GRAY=1 python code/lowlevel_nonlinear_control.py

# (5) statistics, audit, duplicate screening (these read the shipped predictions/ and results/)
python code/stats_paper2.py
python code/stats_cross_cells.py --prefix resnet50_full
python code/audit_tbx11k.py
python code/fingerprint_overlap_full.py
python code/fingerprint_verify.py
python code/dup_sensitivity_fingerprint.py
```

On Windows the `run_*.bat` wrappers do the same thing: they `cd` to the repository root and use
`%PYTHON%` (an existing `.venv`/`venv` next to `code/` is picked up automatically).

Every script resolves its inputs and outputs relative to the repository root (`code/paths.py`):
per-seed predictions in `predictions/`, derived tables in `results/`, low-level features in
`features/`, per-image CSV exports in `cross_pred_index/`, checksums in `md5_lists/`. Raw images are
the only inputs you must supply; see `DATA.md` for the environment variables.

## Function layer

Since v2.0.0 the statistics are split into one numeric module and two thin entry points. All
measurement logic lives in `code/p2_lib.py`; `code/stats_paper2.py` and
`code/stats_cross_cells.py` only fix their parameter block (inputs, class order, 2,000 resamples,
`RandomState(0)`, 95% percentile CI, `ddof=1` dispersion), call the library and assemble their own
artefact, so a caliber change is made once and cannot drift between the two outputs.

| Function | Responsibility |
|:--|:--|
| `load_run_files`, `load_binary_seed_scores` | Per-seed `.npz` discovery, cross-seed label-identity asserts; stored dtype preserved (the four-class mean is cast after averaging, as before). |
| `ensemble_probabilities` | Seed-ensemble probability matrix (mean over the stored stack). |
| `macro_ovr_auc`, `weighted_ovr_auc`, `per_class_ovr_auc`, `binary_auc` | The four AUC read-outs used in the manuscript. |
| `restricted_pair_auc` | Active-vs-latent score on the active/latent subset: `p_lat/(p_lat+p_act)` (primary) plus the two reported alternatives. |
| `confusion_argmax`, `accuracy` | Argmax decision rule, confusion counts and per-class recall (no tuned threshold). |
| `boot_percentile_ci` | Core percentile bootstrap: image-level resampling, a fresh `RandomState(seed)` per call, degenerate resamples discarded without shifting the random stream and counted in `kept`. |
| `boot_auc_ci`, `boot_auc_minus_chance_ci`, `boot_ci_seedmean`, `boot_ci_ensemble` | The four interval shapes reported: binary AUC, AUC−0.5, mean-over-seeds AUC (Table 3 estimand), ensemble reference. |
| `mean_sd`, `min_max` | Per-seed dispersion (mean ± SD, `ddof=1`) and range. |
| `write_json`, `write_lines` | Artefact writing (UTF-8, `indent=2`, identical formatting). |

| Entry point | Reads | Writes |
|:--|:--|:--|
| `code/stats_paper2.py` | `predictions/four_class/`, `predictions/a_vs_l/` | `results/four_class/paper2_stats.json` / `.txt` |
| `code/stats_cross_cells.py --prefix <name>` | `predictions/cross_cohort/<name>/` | `results/cross/cells_ci_<name>.json` / `.txt` |

`make stats` runs both entries; `make figures` reads only `predictions/` and `results/` and does not
import the library. The refactor is numeric-neutral: v2.0.0 reproduces the v1.0.0 statistics files
and all eight figure files byte for byte (SHA-256 checked).

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

`code/count_words.py` implements the convention we used for the journal limit
(sections 1–7 only; headings, abstract, tables, figure legends and references excluded; hyphenated
compounds count as one word). It reports 3,339 words for the submitted main text.

## License / citation

- **MIT License** (see `LICENSE`).
- If you use this code or the derived tables, please cite the manuscript above.

## Contact

Kui Wang — School of Public Health, Shihezi University, Shihezi, Xinjiang, China.

## Reference audit

`audit.py` implements the shape-free reference audit reported in the accompanying
manuscript: it compares a model's same-cohort AUC with a shape-free reference
(`Delta = AUC_model - AUC_reference`, paired bootstrap) and returns a per-cohort
reading. See `AUDIT_README.md` for usage.

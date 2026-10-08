# Shape-free reference audit

`audit.py` quantifies how much of a same-cohort chest-radiograph classification score is
attainable **without resolving pulmonary pathology**, by comparing a model's AUC with a
shape-free reference classifier trained on the identical splits.

This is the implementation of the audit reported in the accompanying manuscript
(*Beyond aggregate accuracy: a shape-free reference audit of cross-cohort generalization in
chest-radiograph tuberculosis models*, manuscript under review). It is independent of the
training code: it needs only the released feature files and per-sample model scores, and it
requires nothing beyond numpy and scikit-learn. Running it on the released artifacts
reproduces **Supplementary Table S4 of the current manuscript version (2026-10-08)**, whose
supplementary material contains Tables S1–S4.

## Definition and reading rule

```
Delta = AUC_model - AUC_reference
```

computed on the **same test split**, with a **paired bootstrap** interval: the test set is
resampled with replacement `B` times (default 10,000) and both AUCs are recomputed on the
*same* resampled indices, which preserves the pairing between model and reference.
Replicates in which a class is absent are discarded.

| Paired-bootstrap interval of Delta | Reading |
|:--|:--|
| upper end below 0.05 | `appearance-dominated` (the score needs no pulmonary pathology) |
| interval straddles zero (and upper end ≥ 0.05) | `inconclusive` (cohort too small to decide) |
| otherwise (interval above 0.05) | `signal beyond appearance` |

The three cases are checked **in that order**: the magnitude test comes first, so a cohort
whose interval excludes zero but stays tiny (for example TBX11K, where the interval is
0.001–0.005) is still labelled `appearance-dominated`. Excluding zero shows that the model
beats the reference by a detectable amount; it does not show that the amount matters.
`signal beyond appearance` therefore means the whole interval sits above 0.05.

**Failure mode.** If the reference exceeds the model (`Delta < 0`), the reading is *not* that
the reference is a ceiling; it means that this model configuration has not exploited even the
appearance-level signal on that cohort.

The 0.05 cut-off is an empirical reading aid, not a test: report the interval, and treat the
label as a summary. On the four released cohorts the reading is stable for every candidate
threshold between 0.005 and 0.040.

## Inputs

The script expects the released layout of this repository:

```
features/<cohort>_train.npz     keys: thumb16 (n,256), stats5 (n,5), labels (n,), filenames (n,)
features/<cohort>_test.npz      same keys
predictions/cross_cohort/<dir>/s<seed>_<cohort>_to_<cohort>.npz    keys: probs (n,2), labels (n,)
```

`<cohort>` is one of `TBX11K`, `Shenzhen`, `Montgomery`, `Qatar`. Model scores may come from
any source: pass your own `.npz` files with the same two keys (`probs`, `labels`) and the tool
will pair them with the reference on the shared test split. The tool verifies that the model
labels and the reference labels are identical position by position before computing anything.

**Which scores the default uses.** The main-matrix same-cohort runs all live in
`predictions/cross_cohort/resnet50_full/`, and that is what the default (`--cohort`/`--all`)
reads for every cohort, matching the diagonal of Table 3 of the manuscript. The directory
`predictions/cross_cohort/dedup_qatar/` is a different artifact: it holds the Qatar
deduplication-sensitivity runs of manuscript Section 5.6 (trained with duplicate files removed,
evaluated on the same 840-image test split; same-cohort AUC 0.999992 against 1.000000 for the
main-matrix runs, a difference of 8e-6). Use it only when auditing that sensitivity analysis,
and pass it explicitly:

Reference family (as reported in the paper):

| Item | Values |
|:--|:--|
| Feature sets | `thumb16_only` (256 dims), `thumb16_plus_stats5` (261 dims) |
| Learners | `logreg` (class_weight=balanced, max_iter=2000), `randomforest` (500 trees, balanced_subsample, random_state=0), `gradientboosting` (max_iter=200, random_state=0), `knn15` (StandardScaler + 15-NN, distance weights) |
| Primary reference | `logreg` on `thumb16_plus_stats5` |
| Strongest reference | the best of the eight feature-set x learner cells on that cohort |

## Usage

```bash
# the four released cohorts, released layout
python audit.py --all --out audit_all.json

# one cohort
python audit.py --cohort TBX11K --resamples 10000 --out audit_TBX11K.json

# explicit paths, e.g. auditing a published model's scores from another source
python audit.py \
    --cohort Shenzhen \
    --features features/Shenzhen_train.npz features/Shenzhen_test.npz \
    --model-preds 'predictions/cross_cohort/resnet50_full/*_Shenzhen_to_Shenzhen.npz' \
    --resamples 10000 --out audit_Shenzhen.json
```

The script self-checks before doing anything: its rank-based AUC must agree with
`sklearn.metrics.roc_auc_score` to within 1e-12 on synthetic scores.

## Output

JSON with one entry per cohort:

```json
{
  "definition": "Delta = AUC_model - AUC_reference, paired bootstrap",
  "resamples": 10000, "seed": 20261008, "threshold": 0.05,
  "results": {
    "TBX11K": {
      "n_test": 1000, "n_positive": 200,
      "model": {"n_seeds": 5, "per_seed_auc": [...], "mean_auc": 1.0},
      "reference_auc": {"thumb16_plus_stats5": {"logreg": ..., "randomforest": ...,
                                                "gradientboosting": ..., "knn15": ...},
                        "thumb16_only": {...}},
      "primary_reference":   {"learner": "logreg", "auc": ..., "delta": ...,
                              "lo": ..., "hi": ..., "p_lt_0": ..., "reading": "..."},
      "strongest_reference": {"learner": ..., "auc": ..., "delta": ...,
                              "lo": ..., "hi": ..., "reading": "..."},
      "delta_range_over_cells": [...]
    }
  }
}
```

## Reproduction

Run from the repository root. On the four released cohorts the audit reproduces the values in
Supplementary Table S4 of the manuscript: with 10,000 resamples and seed 20261008, TBX11K and
Qatar come out `appearance-dominated` (Delta comfortably below 0.05 in both the primary and the
strongest-reference reading), Shenzhen comes out `signal beyond appearance`, and Montgomery
comes out `inconclusive` (its interval straddles zero: n = 28, where a single discordant pair
moves the AUC by 0.0052).

Small numeric differences between implementations are expected for the stochastic learners:
against the released Table S1 values, an independent re-implementation agreed within 0.0006 for
logistic regression, 0.0000 for 15-NN, 0.0052 in one cell for histogram gradient boosting, and
up to 0.0182 for random forest on the 28-image Montgomery split. Point estimates of the model
AUC (the CNN side) reproduce exactly.

**Cross-environment tolerance.** Reference AUCs move by roughly 1e-4 when the scikit-learn
version changes (verified across 1.8.0 and 1.9.1). That is enough to **flip which cell is the
"strongest reference" in near-ties**: for Qatar the envelope cell is
`gradientboosting @ thumb16_plus_stats5` under one version and
`gradientboosting @ thumb16_only` under the other (0.9966 against 0.9959). Read the
`reference_auc` block of the JSON, which lists all eight cells, before treating one learner as
the reference; the reading of every cohort is unaffected, because near-ties differ by far less
than the 0.05 band.

## Requirements

```
python >= 3.9
numpy == 2.4.6
scikit-learn == 1.8.0
```

Versions are pinned because the reference AUCs depend on them: the tool sets single-threaded
BLAS/OpenMP at import, so that these two versions reproduce the reference AUCs bit-for-bit
between machines. Other versions shift reference AUCs in the fourth decimal and can flip a
near-tied strongest cell, so the tool records its own versions in the output JSON and prints a
warning when they differ from the reference environment.

## Files

| File | Purpose |
|:--|:--|
| `audit.py` | the audit (single file, no local imports) |

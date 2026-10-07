# Shape-free reference audit

`audit.py` quantifies how much of a same-cohort chest-radiograph classification score is
attainable **without resolving pulmonary pathology**, by comparing a model's AUC with a
shape-free reference classifier trained on the identical splits.

This is the implementation of the audit reported in the accompanying manuscript
(*Beyond aggregate accuracy: a shape-free reference audit of cross-cohort generalization in
chest-radiograph tuberculosis models*, manuscript under review). It is independent of the
training code: it needs only the released feature files and per-sample model scores, and it
requires nothing beyond numpy and scikit-learn.

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
| wholly below 0.05 | `appearance-dominated` (the score needs no pulmonary pathology) |
| straddles zero | `inconclusive` (cohort too small to decide) |
| otherwise | `signal beyond appearance` |

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

## Requirements

```
python >= 3.9
numpy
scikit-learn
```

## Files

| File | Purpose |
|:--|:--|
| `audit.py` | the audit (single file, no local imports) |

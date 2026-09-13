# Supplementary Table S1 — non-parametric shape-free control (E1)

Learners fitted on the identical features and splits used by the CNNs (`train_cross_cohort.build_splits`),
greyscale (`TBX_GRAY=1`), `random_state=0`, no hyper-parameter tuning. k-NN uses 15 neighbours with features
standardised inside a pipeline fitted on the training split only. Source: `output_cross/lowlevel_nonlinear_control_20260913.json`.

| Cohort | Features | Logistic regression | Random forest | Hist. gradient boosting | 15-NN |
|:--|:--|--:|--:|--:|--:|
| TBX11K | 16x16 + 5 stats | 0.9766 | 0.9913 | 0.9969 | 0.9312 |
| TBX11K | 16x16 only | 0.9760 | 0.9907 | 0.9966 | 0.9300 |
| Shenzhen | 16x16 + 5 stats | 0.8587 | 0.8391 | 0.8381 | 0.8428 |
| Shenzhen | 16x16 only | 0.8587 | 0.8387 | 0.8531 | 0.8444 |
| Montgomery | 16x16 + 5 stats | 0.6719 | 0.7917 | 0.7031 | 0.8750 |
| Montgomery | 16x16 only | 0.6771 | 0.7917 | 0.6927 | 0.8594 |
| Qatar | 16x16 + 5 stats | 0.9765 | 0.9942 | 0.9956 | 0.9762 |
| Qatar | 16x16 only | 0.9765 | 0.9944 | 0.9959 | 0.9758 |

Best learner per cohort (16x16 + 5 stats): TBX11K gradient boosting 0.9969; Shenzhen logistic regression 0.8587;
Montgomery 15-NN 0.8750; Qatar gradient boosting 0.9956. CNN within-cohort AUC for comparison: 1.000 / 0.916 / 0.861 / 1.000.

# Supplementary Table S2 — label-convention sensitivity (E2)

Cross-cohort AUC (5 seeds, greyscale) with TBX11K's TB positive class defined as (A) active + latent (primary),
(B) strictly active, (C) latent only. The other three cohorts are unchanged. Sources:
`output_cross/matrix_summary_resnet50_{full,active,latent}.json`; per-cell 95% CIs from `cells_ci_resnet50_{full,active}.json`.

**A. TB := active + latent (primary)**

| train \ test | TBX11K | Shenzhen | Montgomery | Qatar |
|:--|--:|--:|--:|--:|
| TBX11K | 1.000 | 0.548 | 0.703 | 0.807 |
| Shenzhen | 0.883 | 0.916 | 0.844 | 0.881 |
| Montgomery | 0.889 | 0.814 | 0.861 | 0.720 |
| Qatar | 0.408 | 0.511 | 0.566 | 1.000 |

**B. TB := active only**

| train \ test | TBX11K | Shenzhen | Montgomery | Qatar |
|:--|--:|--:|--:|--:|
| TBX11K | 1.000 | 0.489 | 0.691 | 0.823 |
| Shenzhen | 0.899 | 0.916 | 0.844 | 0.881 |
| Montgomery | 0.900 | 0.814 | 0.861 | 0.720 |
| Qatar | 0.416 | 0.511 | 0.566 | 1.000 |

**C. TB := latent only**

| train \ test | TBX11K | Shenzhen | Montgomery | Qatar |
|:--|--:|--:|--:|--:|
| TBX11K | 1.000 | 0.799 | 0.715 | 0.811 |
| Shenzhen | 0.813 | 0.916 | 0.844 | 0.881 |
| Montgomery | 0.836 | 0.814 | 0.861 | 0.720 |
| Qatar | 0.371 | 0.511 | 0.566 | 1.000 |

TBX11K as source, with the 95% bootstrap CI of the seed-averaged probabilities (2,000 resamples of the target set):

| Target | A. active+latent | B. active only | C. latent only (no CI) |
|:--|:--|:--|:--|
| Shenzhen (n=114→114) | 0.548 (0.451–0.666) | 0.489 (0.394–0.612) | 0.799 |
| Montgomery (n=28→28) | 0.703 (0.515–0.896) | 0.691 (0.497–0.885) | 0.715 |
| Qatar (n=840→840) | 0.807 (0.767–0.838) | 0.823 (0.780–0.850) | 0.811 |
| TBX11K (n=1000→964) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 |

Under B the deficit does not shrink (TBX11K→Shenzhen 0.548 → 0.489; TBX11K→Montgomery 0.703 → 0.691;
same-cohort TBX11K remains 1.000, n_test 1,000 → 964). Column C is a degenerate control: the positive class is
36 latent images in TBX11K's validation split, so it is reported for completeness only and supports no conclusion on its own.

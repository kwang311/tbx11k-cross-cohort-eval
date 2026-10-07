# Supplementary Table S3 — cross-cohort matrix for three backbones (E3)

Mean ± SD over 5 seeds, greyscale inputs, with the 95% bootstrap CI of the mean-over-seeds AUC in parentheses
(2,000 resamples of the target set, `RandomState(0)`). Sources: `output_cross/matrix_summary_<prefix>.json` and
`cells_ci_<prefix>.json` for prefix in `{resnet50_full, resnet18_full, effnetb0_full}`. All three runs use the
shared pipeline described in Methods (ImageNet-pretrained backbone, early stages frozen, 224×224 inputs).

## ResNet-50

| train \ test | TBX11K | Shenzhen | Montgomery | Qatar |
|:--|--:|--:|--:|--:|
| TBX11K | 1.000 ± 0.000 (1.000–1.000) | 0.548 ± 0.065 (0.445–0.644) | 0.703 ± 0.033 (0.514–0.864) | 0.807 ± 0.026 (0.773–0.838) |
| Shenzhen | 0.883 ± 0.029 (0.857–0.908) | 0.916 ± 0.004 (0.863–0.960) | 0.844 ± 0.055 (0.704–0.952) | 0.881 ± 0.016 (0.851–0.909) |
| Montgomery | 0.889 ± 0.014 (0.860–0.913) | 0.814 ± 0.016 (0.733–0.885) | 0.861 ± 0.041 (0.720–0.961) | 0.720 ± 0.033 (0.674–0.760) |
| Qatar | 0.408 ± 0.068 (0.359–0.457) | 0.511 ± 0.100 (0.416–0.609) | 0.566 ± 0.070 (0.338–0.765) | 1.000 ± 0.000 (1.000–1.000) |

Transfer loss (cross mean − same cohort): TBX11K -0.314; Shenzhen -0.047; Montgomery -0.054; Qatar -0.505

## ResNet-18

| train \ test | TBX11K | Shenzhen | Montgomery | Qatar |
|:--|--:|--:|--:|--:|
| TBX11K | 1.000 ± 0.000 (1.000–1.000) | 0.613 ± 0.069 (0.520–0.699) | 0.724 ± 0.042 (0.538–0.886) | 0.714 ± 0.036 (0.673–0.753) |
| Shenzhen | 0.912 ± 0.018 (0.887–0.934) | 0.930 ± 0.015 (0.879–0.968) | 0.859 ± 0.042 (0.713–0.958) | 0.902 ± 0.019 (0.873–0.927) |
| Montgomery | 0.881 ± 0.008 (0.853–0.907) | 0.811 ± 0.009 (0.725–0.888) | 0.846 ± 0.025 (0.681–0.964) | 0.669 ± 0.038 (0.625–0.709) |
| Qatar | 0.451 ± 0.094 (0.403–0.498) | 0.588 ± 0.026 (0.487–0.678) | 0.713 ± 0.039 (0.494–0.889) | 1.000 ± 0.000 (1.000–1.000) |

Transfer loss (cross mean − same cohort): TBX11K -0.316; Shenzhen -0.039; Montgomery -0.059; Qatar -0.416

## EfficientNet-B0

| train \ test | TBX11K | Shenzhen | Montgomery | Qatar |
|:--|--:|--:|--:|--:|
| TBX11K | 1.000 ± 0.000 (1.000–1.000) | 0.683 ± 0.096 (0.594–0.763) | 0.778 ± 0.048 (0.599–0.920) | 0.780 ± 0.035 (0.746–0.812) |
| Shenzhen | 0.911 ± 0.007 (0.887–0.934) | 0.922 ± 0.007 (0.871–0.964) | 0.578 ± 0.055 (0.367–0.775) | 0.828 ± 0.026 (0.795–0.861) |
| Montgomery | 0.894 ± 0.007 (0.872–0.916) | 0.827 ± 0.013 (0.745–0.901) | 0.846 ± 0.019 (0.689–0.961) | 0.774 ± 0.024 (0.741–0.804) |
| Qatar | 0.563 ± 0.085 (0.518–0.605) | 0.619 ± 0.037 (0.532–0.708) | 0.576 ± 0.046 (0.348–0.785) | 1.000 ± 0.000 (1.000–1.000) |

Transfer loss (cross mean − same cohort): TBX11K -0.253; Shenzhen -0.150; Montgomery -0.014; Qatar -0.414

## Reading

The pattern is reproduced by all three backbones: within-cohort AUC at or near 1.000 in TBX11K and Qatar, the
largest transfer losses from exactly those two sources, and the smallest losses from Montgomery and Shenzhen.
Individual cell values are not stable across backbones (e.g. TBX11K→Shenzhen 0.548 / 0.613 / 0.683;
Shenzhen→Montgomery 0.844 / 0.859 / 0.578), so the claim is about the pattern, not about any single cell.

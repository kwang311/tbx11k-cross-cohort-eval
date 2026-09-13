# Supplementary Table S3 — cross-cohort matrix for three backbones (E3)

Mean ± SD over 5 seeds, greyscale inputs, with the 95% bootstrap CI of the seed-averaged probabilities in parentheses
(2,000 resamples of the target set, `RandomState(0)`). Sources: `output_cross/matrix_summary_<prefix>.json` and
`cells_ci_<prefix>.json` for prefix in `{resnet50_full, resnet18_full, effnetb0_full}`. All three runs use the
shared pipeline described in Methods (ImageNet-pretrained backbone, early stages frozen, 224×224 inputs).

## ResNet-50

| train \ test | TBX11K | Shenzhen | Montgomery | Qatar |
|:--|--:|--:|--:|--:|
| TBX11K | 1.000 ± 0.000 (1.000–1.000) | 0.548 ± 0.065 (0.451–0.666) | 0.703 ± 0.033 (0.515–0.896) | 0.807 ± 0.026 (0.767–0.838) |
| Shenzhen | 0.883 ± 0.029 (0.879–0.933) | 0.916 ± 0.004 (0.870–0.967) | 0.844 ± 0.055 (0.743–1.000) | 0.881 ± 0.016 (0.873–0.931) |
| Montgomery | 0.889 ± 0.014 (0.868–0.923) | 0.814 ± 0.016 (0.730–0.891) | 0.861 ± 0.041 (0.740–0.985) | 0.720 ± 0.033 (0.682–0.772) |
| Qatar | 0.408 ± 0.068 (0.337–0.446) | 0.511 ± 0.100 (0.383–0.602) | 0.566 ± 0.070 (0.318–0.805) | 1.000 ± 0.000 (1.000–1.000) |

Transfer loss (cross mean − same cohort): TBX11K -0.314; Shenzhen -0.047; Montgomery -0.054; Qatar -0.505

## ResNet-18

| train \ test | TBX11K | Shenzhen | Montgomery | Qatar |
|:--|--:|--:|--:|--:|
| TBX11K | 1.000 ± 0.000 (1.000–1.000) | 0.613 ± 0.069 (0.548–0.753) | 0.724 ± 0.042 (0.519–0.901) | 0.714 ± 0.036 (0.677–0.758) |
| Shenzhen | 0.912 ± 0.018 (0.900–0.948) | 0.930 ± 0.015 (0.886–0.973) | 0.859 ± 0.042 (0.725–0.980) | 0.902 ± 0.019 (0.887–0.940) |
| Montgomery | 0.881 ± 0.008 (0.863–0.917) | 0.811 ± 0.009 (0.722–0.895) | 0.846 ± 0.025 (0.687–1.000) | 0.669 ± 0.038 (0.630–0.717) |
| Qatar | 0.451 ± 0.094 (0.401–0.509) | 0.588 ± 0.026 (0.481–0.696) | 0.713 ± 0.039 (0.442–0.900) | 1.000 ± 0.000 (1.000–1.000) |

Transfer loss (cross mean − same cohort): TBX11K -0.316; Shenzhen -0.039; Montgomery -0.059; Qatar -0.416

## EfficientNet-B0

| train \ test | TBX11K | Shenzhen | Montgomery | Qatar |
|:--|--:|--:|--:|--:|
| TBX11K | 1.000 ± 0.000 (1.000–1.000) | 0.683 ± 0.096 (0.581–0.779) | 0.778 ± 0.048 (0.599–0.950) | 0.780 ± 0.035 (0.749–0.818) |
| Shenzhen | 0.911 ± 0.007 (0.900–0.947) | 0.922 ± 0.007 (0.875–0.967) | 0.578 ± 0.055 (0.353–0.810) | 0.828 ± 0.026 (0.833–0.896) |
| Montgomery | 0.894 ± 0.007 (0.882–0.926) | 0.827 ± 0.013 (0.752–0.913) | 0.846 ± 0.019 (0.694–0.974) | 0.774 ± 0.024 (0.747–0.812) |
| Qatar | 0.563 ± 0.085 (0.521–0.622) | 0.619 ± 0.037 (0.500–0.711) | 0.576 ± 0.046 (0.350–0.807) | 1.000 ± 0.000 (1.000–1.000) |

Transfer loss (cross mean − same cohort): TBX11K -0.253; Shenzhen -0.150; Montgomery -0.014; Qatar -0.414

## Reading

The pattern is reproduced by all three backbones: within-cohort AUC at or near 1.000 in TBX11K and Qatar, the
largest transfer losses from exactly those two sources, and the smallest losses from Montgomery and Shenzhen.
Individual cell values are not stable across backbones (e.g. TBX11K→Shenzhen 0.548 / 0.613 / 0.683;
Shenzhen→Montgomery 0.844 / 0.859 / 0.578), so the claim is about the pattern, not about any single cell.

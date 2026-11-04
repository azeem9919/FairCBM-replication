# Replication notes: FairCBM vs. Standard CBM (Seeds 42–46, Swin)

## Environment
WSL2 Ubuntu, conda CBM-env (Python 3.10, torch 2.11.0+cu128), RTX 5060 laptop GPU 8 GB.

## Data (NOT included)
dataloader.py has a 2-line edit to allow for image path changes on local device.
Rows kept: 2,848 of 4,000 images used (all from the fitzpatrick17k source subset with valid Fitzpatrick I–VI labels; 397 malignant, 2,451 non-malignant).
Split: 2,278 train / 285 val / 285 test across seeds 42–46 (40 test positives in seeds 42, 44, 45, 46; 39 test positives in seed 43).

## Commands
Fair CBM (seeds 42–46):
python scripts/train_all_models.py --model_type fair_curriculum_cbm --backbone swin \
  --exp_name replic_fair_curriculum_seed42 --epochs 100 --batch_size 16 \
  --fairness_lambda 0.1 --adversarial_lambda 0.01 --eval_every 5 --seed 42 --num_workers 4

Standard CBM baseline (seeds 42–46):
python scripts/train_all_models.py --model_type standard_cbm --backbone swin \
  --exp_name replic_standard_cbm_seed42 --epochs 100 --batch_size 16 \
  --eval_every 5 --seed 42 --num_workers 4

## Five-seed test results: Diagnostic F1 and Accuracy (n = 285 per seed)

| Seed | Std F1 | Fair F1 | Std Acc | Fair Acc |
| :--- | :--- | :--- | :--- | :--- |
| 42 | 0.486 | 0.603 | 0.874 | 0.898 |
| 43 | 0.622 | 0.554 | 0.902 | 0.898 |
| 44 | 0.560 | 0.590 | 0.884 | 0.912 |
| 45 | 0.554 | 0.533 | 0.898 | 0.902 |
| 46 | 0.387 | 0.571 | 0.867 | 0.884 |
| Mean | 0.522 | 0.570 | 0.885 | 0.899 |
| SD | 0.089 | 0.028 | 0.015 | 0.010 |
| Median | 0.554 | 0.571 | 0.884 | 0.898 |
| Min | 0.387 | 0.533 | 0.867 | 0.884 |
| Max | 0.622 | 0.603 | 0.902 | 0.912 |

Across the five seeds, FairCBM raises mean test F1 by +0.049 (SD 0.102) and reduces seed-to-seed F1 standard deviation more than threefold compared to Standard CBM, avoiding the sharp drops seen in Standard CBM on seeds 42 and 46.

## Five-seed test results: Precision and Recall

| Seed | Std Prec | Fair Prec | Std Rec | Fair Rec |
| :--- | :--- | :--- | :--- | :--- |
| 42 | 0.567 | 0.667 | 0.425 | 0.550 |
| 43 | 0.657 | 0.692 | 0.590 | 0.462 |
| 44 | 0.600 | 0.857 | 0.525 | 0.450 |
| 45 | 0.720 | 0.800 | 0.450 | 0.400 |
| 46 | 0.545 | 0.595 | 0.300 | 0.550 |
| Mean | 0.618 | 0.722 | 0.458 | 0.482 |
| SD | 0.071 | 0.106 | 0.110 | 0.066 |
| Median | 0.600 | 0.692 | 0.450 | 0.462 |
| Min | 0.545 | 0.595 | 0.300 | 0.400 |
| Max | 0.720 | 0.857 | 0.590 | 0.550 |

FairCBM achieves higher precision on all five seeds (+0.104 mean gain) while maintaining a slightly higher mean recall and lower recall variance across splits. Mean test AUROC is comparable between Standard CBM (mean 0.781, SD 0.068, median 0.766, min 0.717, max 0.896) and FairCBM (mean 0.773, SD 0.082, median 0.808, min 0.661, max 0.842).

## Five-seed test results: Fairness Disparities (Fitzpatrick I–VI)

| Seed | Std DP | Fair DP | Std EO | Fair EO |
| :--- | :--- | :--- | :--- | :--- |
| 42 | 0.143 | 0.164 | 0.600 | 0.750 |
| 43 | 0.185 | 0.091 | 0.857 | 0.571 |
| 44 | 0.078 | 0.127 | 0.875 | 0.800 |
| 45 | 0.121 | 0.067 | 0.625 | 0.750 |
| 46 | 0.130 | 0.204 | 1.000 | 0.857 |
| Mean | 0.131 | 0.131 | 0.791 | 0.746 |
| SD | 0.039 | 0.055 | 0.173 | 0.107 |
| Median | 0.130 | 0.127 | 0.857 | 0.750 |
| Min | 0.078 | 0.067 | 0.600 | 0.571 |
| Max | 0.185 | 0.204 | 1.000 | 0.857 |

Demographic Parity (DP) disparity is identical on average between both models, while Equalized Odds (EO) disparity is slightly lower on average for FairCBM (-0.046 mean difference, winning on seeds 43, 44, and 46). However, EO disparity is dominated by extreme small-sample noise in dark skin tones: across every test split, Fitzpatrick Type VI contains exactly 1 malignant image (out of 10–11 Type VI test images) and Type V contains only 2–3 malignant images, forcing Type VI true positive rate to binary extremes (0.000 or 1.000).

## Five-seed test results: Concept Bottleneck Performance (23 Concepts)

| Seed | Std Micro F1 | Fair Micro F1 | Std Macro F1 | Fair Macro F1 |
| :--- | :--- | :--- | :--- | :--- |
| 42 | 0.640 | 0.620 | 0.436 | 0.436 |
| 43 | 0.577 | 0.638 | 0.351 | 0.434 |
| 44 | 0.612 | 0.645 | 0.352 | 0.355 |
| 45 | 0.664 | 0.670 | 0.442 | 0.466 |
| 46 | 0.639 | 0.600 | 0.406 | 0.398 |
| Mean | 0.627 | 0.635 | 0.398 | 0.418 |
| SD | 0.033 | 0.026 | 0.044 | 0.043 |
| Median | 0.639 | 0.638 | 0.406 | 0.434 |
| Min | 0.577 | 0.600 | 0.351 | 0.355 |
| Max | 0.664 | 0.670 | 0.442 | 0.466 |

Mean concept prediction accuracy across the 23 clinical concepts is 0.900 (SD 0.013, median 0.901, min 0.878, max 0.910) for Standard CBM and 0.902 (SD 0.012, median 0.904, min 0.890, max 0.915) for FairCBM.

<img width="1950" height="600" alt="faircbm_5seed_comparison" src="https://github.com/user-attachments/assets/5e5b4749-bdfa-4b41-9431-9872b8089716" />

## Caveats
- Five seeds (42–46) and eval_every 5 were used, unlike the FairCBM paper's 100 seeds and eval_every 1.
- Fitzpatrick Type VI has 11 malignant images in the entire kept dataset (and exactly 1 malignant image in each 285-sample test split), causing single-sample prediction flips to swing per-group TPR between 0.000 and 1.000.
- Final log line "Global best model saved (F1: ...)" prints the composite score, not F1.

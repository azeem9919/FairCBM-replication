# Replication notes: FairCBM, seed 42, Swin

## Environment
WSL2 Ubuntu, conda CBM-env (Python 3.10, torch 2.11.0+cu128), RTX 5060 laptop GPU 8 GB.

## Data (NOT included)
dataloader.py has a 2-line edit to allow for image path changes on local device.
Rows kept: 2,848 of 4,000 images used. Split 2,278/285/285 (seed 42).

## Commands
Fair: python scripts/train_all_models.py --model_type fair_curriculum_cbm --backbone swin
  --exp_name replic_fair_curriculum_seed42 --epochs 100 --batch_size 16
  --fairness_lambda 0.1 --adversarial_lambda 0.01 --eval_every 5 --seed 42 --num_workers 4

## Fair run, test metrics (epoch-80 checkpoint; single seed, pipeline check only)
F1 0.603, accuracy 0.898, precision 0.667, recall 0.550, EO disparity 0.750, DP disparity 0.164.

## Caveats
- One seed, eval_every 5 was used unlike the FairCBM Paper's 100 seeds, eval_every 1.
- Type VI has approximately 11 malignant images in total thus affecting the per-group metric.
- Final log line "Global best model saved (F1: ...)" prints the composite score, not F1.
- Baseline run hung twice at one epoch on this laptop (CUDA/driver suspected).

## Baseline run (standard_cbm, same settings, BS=16, seed 42)
Test (epoch-40 checkpoint, selected by val F1 0.677): F1 0.486, accuracy 0.874, precision 0.567, recall 0.425, DP disparity 0.143, EO disparity 0.600.
First launch stalled at epoch 86 (main process spinning, GPU idle); a rerun completed all 100 epochs without a stall, so the stall was not deterministic. Cause unconfirmed (GPU/driver suspected).
Single seed, ~40 test positives: the fair-vs-baseline difference is not statistically supportable. Fair model is worse on DP and EO disparity in this pair.

## How to reproduce
git clone https://github.com/matthew-cockayne/FairCBM && cd FairCBM
git checkout 478eae6
git apply /path/to/FairCBM-replication/patches/dataloader_fallback.patch   # path to this repo's patch file
Then obtain SkinCAP yourself and run the commands above.

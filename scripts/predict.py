# predict.py - per-image predictions from a saved checkpoint, CPU only (safe while the GPU trains)
import sys, os, argparse, torch, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from src.data.dataloader import SkinCapDataset
from train_all_models import create_model   # reuses the exact model-building code

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--model_type", required=True)     # standard_cbm or fair_curriculum_cbm
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--expected_test_f1", type=float, required=True)
a = p.parse_args()

DATA = "/home/ak_hyperx/FairCBM/data"
CSV = f"{DATA}/annotation/skincap_v240623.csv"
concepts = [l.strip() for l in open(f"{DATA}/skincap_concepts.txt") if l.strip()]
assert len(concepts) == 23, len(concepts)

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
print("checkpoint epoch (0-indexed):", ck["epoch"], "| config:", ck.get("config"))

# lambdas match the runs (0.1 / 0.01); they do not affect inference
model = create_model(a.model_type, 23, "swin", 0.1, 0.01, concept_names=concepts)
model.load_state_dict(ck["model_state_dict"])   # strict: a mismatch here means the build differs
model.eval()

rows, test_true = [], None
for split in ("train", "val", "test"):
    ds = SkinCapDataset(root_dir=DATA, raw_csv=CSV, split=split, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    probs, cprobs = [], []
    with torch.no_grad():
        for batch in dl:
            out = model(batch[0])
            c_logit, b_logit = out[0], out[1]          # first two outputs, as in the baseline loop
            probs.append(torch.sigmoid(b_logit).reshape(-1).numpy())
            cprobs.append(torch.sigmoid(c_logit).numpy())
    probs, cprobs = np.concatenate(probs), np.concatenate(cprobs)
    df = pd.DataFrame({"file": ds.df["skincap_file_path"].values, "split": split,
                       "yhat_prob": probs, "yhat_label": (probs >= 0.5).astype(int)})
    for i, n in enumerate(concepts):
        df[f"concept_{n}"] = cprobs[:, i]
    rows.append(df)
    if split == "test":
        y = ds.df["malignant"].values       # used ONLY for the check, never saved
        pred = (probs >= 0.5).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        f1 = 2 * tp / (2 * tp + ((pred == 1) & (y == 0)).sum() + ((pred == 0) & (y == 1)).sum())
        print(f"recomputed test F1 = {f1:.4f} (log: {a.expected_test_f1})")
    print(split, len(df))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
pd.concat(rows).to_csv(a.out, index=False)   # no labels, no tone: safe to share
print("saved", a.out)

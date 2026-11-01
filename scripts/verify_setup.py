import sys
import os

sys.path.append(os.path.abspath("src"))
from data.dataloader import get_dataloader

# Point directly to your annotation CSV and root data directory
ROOT_DIR = "data"
RAW_CSV = "data/annotation/skincap_v240623.csv"

print("Initializing training dataloader...")
train_loader = get_dataloader(
    root_dir=ROOT_DIR,
    split='train',
    batch_size=4,
    label_type='concept',
    raw_csv=RAW_CSV
)

images, concepts, labels, fitzpatrick = next(iter(train_loader))
print("\nBatch loaded successfully!")
print(f"  Images shape: {images.shape}")
print(f"  Concepts shape: {concepts.shape}")
print(f"  Binary labels shape: {labels.shape}")
print(f"  Fitzpatrick scale shape: {fitzpatrick.shape}")
print("\n✓ Dataloader is fully connected and ready for training on your RTX 5060!")

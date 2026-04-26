"""Project paths and runtime config.

IMAGE_DIR is environment-overridable: set IMAGE_DIR to the folder where
polyvore images live. Expected layout: IMAGE_DIR/<set_id>/<index>.jpg
The Flask app falls back gracefully when an image isn't found.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(exist_ok=True)

IMAGE_DIR = Path("C:\\Users\\shrey\\OneDrive\\Desktop\\Outfit-recommendation\\dataset\\images").expanduser().resolve()

NGNN_CKPT = WEIGHTS / "ngnn.pt"
TEXT_CACHE = WEIGHTS / "cat_text_embeds.pt"

CLIP_MODEL = "openai/clip-vit-base-patch32"

# ─────────── Category filtering ───────────
# Categories with fewer occurrences in train+test are dropped. Combined with
# the explicit deny-list in filter.py this leaves ~75 fashion categories.
MIN_CATEGORY_FREQ = 100
# Outfits with fewer than this many surviving items after filtering are dropped.
MIN_OUTFIT_SIZE = 3

# ─────────── Model ───────────
EMBED_DIM = 128
GNN_HOPS = 2          # rounds of message passing inside the outfit clique
DROPOUT = 0.2

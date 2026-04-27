"""Project paths and runtime config."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(exist_ok=True)

IMAGE_DIR = Path("C:\\Users\\shrey\\OneDrive\\Desktop\\Outfit-recommendation\\dataset\\images").resolve()

# Splits — new dataset schema (items[] with name/price/likes/image/categoryid)
TRAIN_FILE = "train_no_dup.json"
VALID_FILE = "valid_no_dup.json"
TEST_FILE  = "test_no_dup.json"
FITB_FILE  = "fill_in_blank_test.json"
COMPAT_FILE = "fashion_compatibility_prediction.txt"

# ─── Filter ───
MIN_CATEGORY_FREQ = 100
MIN_OUTFIT_SIZE = 3

# ─── Model ───
EMBED_DIM = 128
GNN_HOPS = 2
DROPOUT = 0.2
VISUAL_DIM = 2048    # ResNet50 pooled feature size, used only if visuals.pt present

# ─── Cached artefacts ───
NGNN_CKPT   = WEIGHTS / "itemgnn.pt"
ITEMS_CACHE = WEIGHTS / "items.pt"
VISUAL_CACHE = WEIGHTS / "visuals.pt"

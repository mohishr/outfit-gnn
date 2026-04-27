"""Optional: extract per-item visual features with a pretrained CNN.

Run once to enrich the GNN with image content:
    pip install torchvision
    python -m src.visual            # ~30 min on CPU for 113k items

Output: weights/visuals.pt  — Tensor[num_items, VISUAL_DIM]
The training and inference paths auto-detect this file and turn on the
visual stream of the GNN. Without it, the model still works on names +
categories alone.
"""
import time
from pathlib import Path

import torch
from PIL import Image

from .config import VISUAL_CACHE, VISUAL_DIM, IMAGE_DIR
from .items import ItemIndex


def _resnet50_extractor():
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    m = resnet50(weights=weights)
    m.fc = torch.nn.Identity()  # drop final classifier; pooled feature is 2048-d
    m.eval()
    return m, weights.transforms()


@torch.no_grad()
def extract(batch_size: int = 32):
    try:
        import torchvision  # noqa: F401
    except ImportError:
        raise SystemExit("torchvision not installed. Run: pip install torchvision")

    items = ItemIndex.load_or_build()
    N = len(items)
    print(f"[visual] indexing {N:,} items...")

    model, transform = _resnet50_extractor()

    feats = torch.zeros(N, VISUAL_DIM)
    missing = []
    t0 = time.time()
    batch_imgs, batch_idx = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs)
        feats[batch_idx] = model(x).cpu()
        batch_imgs.clear()
        batch_idx.clear()

    for i in range(N):
        p = items.image_path(i)
        if p is None:
            missing.append(i)
            continue
        try:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(transform(img))
            batch_idx.append(i)
        except Exception:
            missing.append(i)
        if len(batch_imgs) >= batch_size:
            flush()
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{N}]  missing={len(missing)}  elapsed={time.time()-t0:.0f}s")
    flush()

    print(f"[visual] missing images: {len(missing)} / {N}")
    torch.save(feats, VISUAL_CACHE)
    print(f"saved -> {VISUAL_CACHE}  shape={tuple(feats.shape)}")


if __name__ == "__main__":
    extract()

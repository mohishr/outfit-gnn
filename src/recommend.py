"""End-to-end recommendation pipeline (filtered, production-ready).

  STAGE 1 — Prompt → category preference vector (TextEncoder, CLIP-or-fallback).
  STAGE 2 — Score every candidate outfit:
                prompt_match = mean(text_score[c_i])
                compat       = NGNN.outfit_score(rcats(O))     ← precomputed
                final        = α · prompt_match + (1-α) · compat
  STAGE 3 — Return top-k outfits with set_id-based item ids and image URLs.

Data: only the FILTERED outfits (fashion-only categories) — set by filter.py.
Beauty / tech / home items never appear in recommendations.

Why outfit retrieval, not item generation: the GNN scores categories, not
items — every item of category C has the same embedding. Per-item ranking
needs item-level features (visual or textual) or item-id nodes; documented
as the next upgrade in CLAUDE.md.
"""
from __future__ import annotations

import torch

from .config import NGNN_CKPT, EMBED_DIM, GNN_HOPS, DROPOUT
from .dataset import load_clean, remap
from .model import NGNN
from .text_encoder import TextEncoder


class Recommender:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        d = load_clean()
        self.names = d["names"]
        self.cid2rcid = d["cid2rcid"]
        self.outfits = d["train"] + d["test"]
        self.stats = d["stats"]

        ckpt = torch.load(NGNN_CKPT, map_location="cpu", weights_only=False)
        if ckpt["num_cats"] != d["num_cats"]:
            raise RuntimeError(
                f"checkpoint has num_cats={ckpt['num_cats']} but filtered dataset "
                f"has {d['num_cats']}. Run `python -m src.train --dummy` (or train) "
                f"to regenerate the checkpoint after a filter change."
            )
        self.model = NGNN(
            ckpt["num_cats"],
            dim=ckpt.get("dim", EMBED_DIM),
            hops=ckpt.get("hops", GNN_HOPS),
            dropout=ckpt.get("dropout", DROPOUT),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.ckpt_meta = {"auc": ckpt.get("auc"), "dummy": ckpt.get("dummy", False)}

        self.text = TextEncoder()
        self._precompute_compat()

    @torch.no_grad()
    def _precompute_compat(self):
        self._compat = []
        for o in self.outfits:
            rcats = remap(o, self.cid2rcid)
            s = float(self.model.outfit_score(torch.tensor(rcats))) if rcats else 0.0
            self._compat.append(s)

    @torch.no_grad()
    def recommend(self, prompt: str, k: int = 5):
        cat_scores = self.text.score_categories(prompt)
        ranked = []
        for o, compat in zip(self.outfits, self._compat):
            cats = o["items_category"]
            if not cats:
                continue
            prompt_match = sum(cat_scores.get(c, 0.0) for c in cats) / len(cats)
            final = self.alpha * prompt_match + (1 - self.alpha) * compat
            ranked.append((final, prompt_match, compat, o))
        ranked.sort(key=lambda x: -x[0])

        out = []
        for final, pm, cs, o in ranked[:k]:
            items = [
                {
                    "item_id": f"{o['set_id']}_{idx}",
                    "set_id": o["set_id"],
                    "index": idx,
                    "category_id": c,
                    "category": self.names.get(c, str(c)),
                    "image_url": f"/image/{o['set_id']}/{idx}",
                    "prompt_score": round(cat_scores.get(c, 0.0), 3),
                }
                for c, idx in zip(o["items_category"], o["items_index"])
            ]
            out.append({
                "set_id": o["set_id"],
                "items": items,
                "compatibility_score": round(cs, 3),
                "prompt_match_score": round(pm, 3),
                "final_score": round(final, 3),
            })
        return out


if __name__ == "__main__":
    import json, sys
    p = " ".join(sys.argv[1:]) or "summer cozy blue vibes"
    print(json.dumps(Recommender().recommend(p, 3), indent=2))

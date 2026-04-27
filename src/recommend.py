"""Item-level recommendation pipeline.

Stage 1 — Prompt → item-level prompt-match score
    Tokenise the prompt with the same vocabulary used to embed item names.
    Mean-pool the model's word embeddings to get a prompt vector. For each
    item compute cosine similarity between the prompt vector and the item's
    name embedding (in the same trained space). Min-max normalise to [0,1].

    Items whose names look like the prompt — "blue silk dress", "summer
    sandals" — score high regardless of category overlap.

Stage 2 — GNN compatibility per outfit (precomputed)
    For every outfit O, build item features (name + category + visual if
    cached), run the GNN, take mean(sigmoid(pair_logits)) as outfit_score.

Stage 3 — Combine + rank outfits
    For O = (item_1 .. item_n):
        prompt_score(O) = mean(prompt_score[item_i])
        final          = α · prompt_score(O) + (1-α) · compat_score(O)
    Each result item carries name, category, image_url, and its individual
    prompt_score so the UI can highlight which items matched.

Stage 4 — Item-level fill-in given a partial outfit (`fill_blank`)
    Given a list of item ids and a target category id, score every candidate
    item in that category by the GNN compat of (partial ∪ {candidate}) plus
    that candidate's own prompt match. Returns top items.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import NGNN_CKPT, EMBED_DIM, GNN_HOPS, DROPOUT, VISUAL_DIM, VISUAL_CACHE, IMAGE_DIR
from .dataset import load_clean
from .items import ItemIndex
from .model import ItemGNN


class Recommender:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.d = load_clean()
        self.items = ItemIndex.load_or_build()
        self.visuals = self._load_visuals()

        ckpt = torch.load(NGNN_CKPT, map_location="cpu", weights_only=False)
        self.model = ItemGNN(
            num_cats=ckpt["num_cats"], vocab_size=ckpt["vocab_size"],
            dim=ckpt.get("dim", EMBED_DIM), hops=ckpt.get("hops", GNN_HOPS),
            dropout=ckpt.get("dropout", DROPOUT),
            visual_dim=ckpt.get("visual_dim", VISUAL_DIM),
            use_visual=ckpt.get("use_visual", False),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.ckpt_meta = {"auc": ckpt.get("auc"), "fitb": ckpt.get("fitb"),
                          "dummy": ckpt.get("dummy", False),
                          "use_visual": ckpt.get("use_visual", False)}

        # Pre-compute every item's name embedding (in word-embedding space)
        self._build_caches()

    def _load_visuals(self):
        if not VISUAL_CACHE.exists():
            return None
        return torch.load(VISUAL_CACHE, map_location="cpu", weights_only=False)

    @torch.no_grad()
    def _build_caches(self):
        """Cache name embeddings for every item, then GNN compat for every outfit."""
        items = self.items
        N = len(items)
        # name embeddings (same model.word_emb pool used during training)
        chunks = []
        chunk_size = 4096
        for s in range(0, N, chunk_size):
            ints = list(range(s, min(s + chunk_size, N)))
            toks = [items.name_tokens[i] for i in ints]
            chunks.append(self.model.name_embed(toks))
        self.name_emb = torch.cat(chunks, 0)                     # [N, dim]
        self.name_emb_n = F.normalize(self.name_emb, dim=-1)

        # outfit-level compat (uses item_features incl. category + visual if any)
        self._compat = []
        for set_id, ints in items.outfit_to_items.items():
            if len(ints) < 2:
                self._compat.append((set_id, ints, 0.0))
                continue
            toks = [items.name_tokens[i] for i in ints]
            cats = torch.tensor([items.cat_dense[i] for i in ints])
            vis = self.visuals[ints] if (self.visuals is not None and self.model.use_visual) else None
            h0 = self.model.item_features(toks, cats, vis)
            score = float(torch.sigmoid(self.model.pair_logits(h0)).mean())
            self._compat.append((set_id, ints, score))

    @torch.no_grad()
    def _prompt_to_item_scores(self, prompt: str) -> torch.Tensor:
        toks = self.items.encode_prompt(prompt)
        t = torch.tensor(toks, dtype=torch.long)
        q = self.model.word_emb(t).mean(0)
        q = F.normalize(q, dim=-1)
        sims = self.name_emb_n @ q                              # [N], in [-1, 1]
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
        return sims                                              # [N] in [0, 1]

    def _item_record(self, idx: int, prompt_score: float) -> dict:
        items = self.items
        iid = items.item_ids[idx]
        set_id, sub = iid.rsplit("_", 1)
        name = items.names[idx] or "(unnamed)"
        cat = items.cats[idx]
        cat_name = self.d["names"].get(cat, str(cat))
        return {
            "item_id": iid,
            "set_id": set_id,
            "index": int(sub),
            "name": name,
            "category_id": cat,
            "category": cat_name,
            "image_url": f"/image/{set_id}/{sub}",
            "prompt_score": round(float(prompt_score), 3),
        }

    @torch.no_grad()
    def recommend(self, prompt: str, k: int = 5):
        item_prompt = self._prompt_to_item_scores(prompt)
        ranked = []
        for set_id, ints, compat in self._compat:
            if not ints:
                continue
            ps = float(item_prompt[ints].mean())
            final = self.alpha * ps + (1 - self.alpha) * compat
            ranked.append((final, ps, compat, set_id, ints))
        ranked.sort(key=lambda x: -x[0])

        out = []
        for final, ps, cs, set_id, ints in ranked[:k]:
            items = sorted(
                (self._item_record(i, item_prompt[i]) for i in ints),
                key=lambda x: x["index"],
            )
            out.append({
                "set_id": set_id,
                "items": items,
                "compatibility_score": round(cs, 3),
                "prompt_match_score": round(ps, 3),
                "final_score": round(final, 3),
            })
        return out

    @torch.no_grad()
    def fill_blank(self, partial_iids: list[str], target_cat: int | None = None,
                    prompt: str = "", top_k: int = 5):
        """Suggest the best item to add to a partial outfit, optionally restricted
        to a target category id. Combines GNN compat with prompt match."""
        items = self.items
        partial = [items.id_to_int[i] for i in partial_iids if i in items.id_to_int]
        if not partial:
            return []

        candidates = (
            [i for i, c in enumerate(items.cats) if c == target_cat]
            if target_cat is not None else list(range(len(items)))
        )
        # cap candidates to keep latency bounded
        if len(candidates) > 5000:
            import random
            candidates = random.sample(candidates, 5000)

        item_prompt = self._prompt_to_item_scores(prompt) if prompt else None

        scores = []
        for c in candidates:
            ints = partial + [c]
            toks = [items.name_tokens[i] for i in ints]
            cats = torch.tensor([items.cat_dense[i] for i in ints])
            vis = self.visuals[ints] if (self.visuals is not None and self.model.use_visual) else None
            h0 = self.model.item_features(toks, cats, vis)
            compat = float(torch.sigmoid(self.model.pair_logits(h0)).mean())
            ps = float(item_prompt[c]) if item_prompt is not None else 0.0
            final = self.alpha * ps + (1 - self.alpha) * compat
            scores.append((final, ps, compat, c))
        scores.sort(key=lambda x: -x[0])
        return [
            {
                **self._item_record(c, ps),
                "compatibility_score": round(cs, 3),
                "final_score": round(f, 3),
            }
            for f, ps, cs, c in scores[:top_k]
        ]


if __name__ == "__main__":
    import json, sys
    p = " ".join(sys.argv[1:]) or "summer cozy blue vibes"
    print(json.dumps(Recommender().recommend(p, 3), indent=2))

"""Prompt → category preferences (stage 1 of the pipeline).

Primary path: CLIP text encoder.
    The CLIP text tower embeds the prompt and each category-name template
    ("a fashion item: <name>") into the same space. Cosine similarity gives
    a continuous score per category. Min-max normalised to [0, 1].
    Category embeddings are computed once and cached at TEXT_CACHE.

Fallback path: a curated vibe synonym table + substring matching.
    Used automatically when `transformers` isn't installed or CLIP can't
    download. Lower quality but the pipeline still works.

Both paths only consider the *filtered* fashion categories — beauty/tech/
home items are excluded earlier by filter.py, so they can't pollute prompt
matching either.
"""
import re
import torch
import torch.nn.functional as F

from .config import CLIP_MODEL, TEXT_CACHE
from .dataset import load_clean


VIBES = {
    "summer":   ["shorts", "sandals", "sunglasses", "tops", "skirts", "swimwear"],
    "winter":   ["coats", "sweaters", "boots", "gloves", "hats", "scarves"],
    "spring":   ["dresses", "skirts", "flats", "cardigans"],
    "fall":     ["sweaters", "boots", "cardigans", "jackets"],
    "autumn":   ["sweaters", "boots", "cardigans", "jackets"],
    "cozy":     ["sweaters", "cardigans", "socks", "hats", "scarves"],
    "warm":     ["sweaters", "coats", "boots"],
    "cold":     ["coats", "sweaters", "gloves", "boots"],
    "casual":   ["jeans", "tops", "sneakers", "tshirts", "tees"],
    "formal":   ["dresses", "pumps", "jewelry", "watches", "blouses"],
    "elegant":  ["gowns", "dresses", "pumps", "jewelry"],
    "sporty":   ["activewear", "sneakers", "shorts"],
    "athletic": ["activewear", "sneakers", "shorts"],
    "beach":    ["swimwear", "sandals", "sunglasses", "hats"],
    "office":   ["blouses", "pumps", "pants", "skirts"],
    "party":    ["dresses", "pumps", "clutches", "jewelry"],
    "date":     ["dresses", "pumps", "jewelry"],
    "minimal":  ["tops", "pants", "flats"],
    "boho":     ["dresses", "sandals", "jewelry"],
    "blue":     ["jeans"],
    "denim":    ["jeans"],
}


class TextEncoder:
    def __init__(self):
        self.names = load_clean()["names"]                  # filtered cid -> name
        self._cids = sorted(self.names)
        self._clip = self._init_clip()
        self._cat_embeds = self._compute_cat_embeds() if self._clip else None
        self._fallback_index = self._build_fallback_index()

    # ──────────────────── CLIP path ────────────────────
    def _init_clip(self):
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
        except ImportError:
            return None
        try:
            tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
            mdl = CLIPTextModel.from_pretrained(CLIP_MODEL).eval()
            return {"tok": tok, "mdl": mdl}
        except Exception as e:
            print(f"[text] CLIP unavailable ({e}); using keyword fallback")
            return None

    @torch.no_grad()
    def _embed(self, texts):
        toks = self._clip["tok"](texts, padding=True, truncation=True, return_tensors="pt")
        out = self._clip["mdl"](**toks).last_hidden_state.mean(dim=1)
        return F.normalize(out, dim=-1)

    def _compute_cat_embeds(self):
        if TEXT_CACHE.exists():
            d = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
            if d.get("cids") == self._cids:
                return d["embeds"]
        prompts = [f"a fashion item: {self.names[c]}" for c in self._cids]
        embeds = self._embed(prompts)
        torch.save({"cids": self._cids, "embeds": embeds}, TEXT_CACHE)
        return embeds

    # ──────────────────── Fallback path ────────────────────
    def _build_fallback_index(self):
        idx = {}
        for cid, name in self.names.items():
            for tok in re.findall(r"[a-z]+", name.lower()):
                idx.setdefault(tok, set()).add(cid)
        for vibe, seeds in VIBES.items():
            for s in seeds:
                idx.setdefault(vibe, set()).update(idx.get(s, set()))
        return idx

    # ──────────────────── Public ────────────────────
    @torch.no_grad()
    def score_categories(self, prompt: str) -> dict:
        prompt = prompt.strip()
        if self._clip is not None:
            q = self._embed([prompt])
            sims = (q @ self._cat_embeds.T).squeeze(0)
            sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
            return dict(zip(self._cids, sims.tolist()))
        toks = re.findall(r"[a-z]+", prompt.lower())
        hits = set()
        for t in toks:
            hits |= self._fallback_index.get(t, set())
        return {c: (1.0 if c in hits else 0.0) for c in self._cids}

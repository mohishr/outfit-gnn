"""Item index — assigns every (set_id, index) pair a dense item id, tokenises
its name into a shared vocabulary, and resolves its on-disk image path.

This index is the bridge between the dataset and the model:

    item_id (int)  ↔  set_id_idx (string)  ↔  {name, name_tokens, cat, image}

Vocab is built once from all item names across train+valid+test, then
frozen and saved to `weights/items.pt` so training and inference share
exactly the same token ids.

Why our own vocab (and not e.g. CLIP): CLIP isn't installed in this env and
the item-name corpus is small (~10k unique tokens). A 64-dim trainable
word-embedding table learnt jointly with the GNN converges fast and gives
us a *shared* embedding space for prompts and item names without external
deps.
"""
from __future__ import annotations

import re
import torch
from collections import Counter
from pathlib import Path

from .config import IMAGE_DIR, ITEMS_CACHE
from .dataset import load_clean


_TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD = 0
UNK = 1


def tokenise(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class ItemIndex:
    """Frozen item table. Build once, reload from cache afterwards."""

    def __init__(
        self,
        item_ids: list[str],          # ["214181831_1", ...]  length N
        names: list[str],
        cats: list[int],              # raw polyvore cid (not dense)
        cat_dense: list[int],         # dense cat idx for embedding lookup
        name_tokens: list[list[int]], # per-item list of token ids
        word2id: dict[str, int],
        outfit_to_items: dict[str, list[int]],   # set_id -> [item_id]
    ):
        self.item_ids = item_ids
        self.names = names
        self.cats = cats
        self.cat_dense = cat_dense
        self.name_tokens = name_tokens
        self.word2id = word2id
        self.id2word = {i: w for w, i in word2id.items()}
        self.outfit_to_items = outfit_to_items
        self.id_to_int = {iid: i for i, iid in enumerate(item_ids)}

    # ──────────── factories ────────────
    @classmethod
    def build(cls, *, min_word_freq: int = 2):
        d = load_clean()
        cid2rcid = d["cid2rcid"]

        # Pass 1: vocab from all item names
        wc = Counter()
        for split in ("train", "valid", "test"):
            for o in d[split]:
                for it in o["items"]:
                    wc.update(tokenise(it["name"]))
        word2id = {"<pad>": PAD, "<unk>": UNK}
        for w, n in wc.most_common():
            if n >= min_word_freq:
                word2id[w] = len(word2id)

        def encode(text):
            return [word2id.get(t, UNK) for t in tokenise(text)] or [UNK]

        # Pass 2: build item table
        item_ids, names, cats, cat_dense, name_tokens = [], [], [], [], []
        outfit_to_items: dict[str, list[int]] = {}
        for split in ("train", "valid", "test"):
            for o in d[split]:
                ids_in_outfit = []
                for it in o["items"]:
                    iid = f"{o['set_id']}_{it['index']}"
                    item_idx = len(item_ids)
                    item_ids.append(iid)
                    names.append(it["name"])
                    cats.append(it["categoryid"])
                    cat_dense.append(cid2rcid[it["categoryid"]])
                    name_tokens.append(encode(it["name"]))
                    ids_in_outfit.append(item_idx)
                outfit_to_items[o["set_id"]] = ids_in_outfit

        return cls(item_ids, names, cats, cat_dense, name_tokens, word2id, outfit_to_items)

    @classmethod
    def load_or_build(cls):
        if ITEMS_CACHE.exists():
            try:
                d = torch.load(ITEMS_CACHE, map_location="cpu", weights_only=False)
                return cls(**d)
            except Exception as e:
                print(f"[items] cache load failed ({e}); rebuilding")
        idx = cls.build()
        idx.save()
        return idx

    def save(self):
        torch.save(
            {
                "item_ids": self.item_ids,
                "names": self.names,
                "cats": self.cats,
                "cat_dense": self.cat_dense,
                "name_tokens": self.name_tokens,
                "word2id": self.word2id,
                "outfit_to_items": self.outfit_to_items,
            },
            ITEMS_CACHE,
        )

    # ──────────── helpers ────────────
    def __len__(self):
        return len(self.item_ids)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def encode_prompt(self, text: str) -> list[int]:
        return [self.word2id.get(t, UNK) for t in tokenise(text)] or [UNK]

    def image_path(self, item_int: int) -> Path | None:
        iid = self.item_ids[item_int]
        set_id, idx = iid.rsplit("_", 1)
        for ext in ("jpg", "jpeg", "png", "webp"):
            p = IMAGE_DIR / set_id / f"{idx}.{ext}"
            if p.is_file():
                return p
        return None


if __name__ == "__main__":
    idx = ItemIndex.load_or_build()
    print(f"items: {len(idx):,}")
    print(f"vocab: {idx.vocab_size:,}")
    print(f"outfits indexed: {len(idx.outfit_to_items):,}")
    print(f"sample item: id={idx.item_ids[0]}  name={idx.names[0]!r}  cat={idx.cats[0]}  tokens={idx.name_tokens[0]}")
    n_with_img = sum(1 for i in range(min(200, len(idx))) if idx.image_path(i))
    print(f"images present (first 200 sampled): {n_with_img}/200")

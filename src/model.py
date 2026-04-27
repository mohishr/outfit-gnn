"""ItemGNN — item-level graph neural network for outfit compatibility.

Each *item* (not category) is a node. The node feature is built from three
streams:

    name    EmbeddingBag(vocab, dim)   mean-pools the item-name tokens
    cat     Embedding(num_cats, dim)   category-aware bias
    visual  Linear(2048, dim)          projects ResNet-style visual feats
                                        (skipped when no visuals.pt cached)

    item_feat = name_emb + cat_emb + (visual_emb if visuals else 0)

The same word table that embeds item names also embeds the user's prompt at
recommendation time — so prompt and item descriptions live in a *shared*
learned space. Cosine similarity gives a per-item prompt-match score.

Then the GNN runs T hops of leave-one-out message passing inside the outfit
clique, and a pair MLP scores compatibility on the refined embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _bag_lookup(word_emb: nn.Embedding, token_lists: list[list[int]]) -> torch.Tensor:
    """Mean-pool word embeddings over each item's token list. Returns [N, dim]."""
    out = []
    for toks in token_lists:
        t = torch.tensor(toks, dtype=torch.long, device=word_emb.weight.device)
        out.append(word_emb(t).mean(0))
    return torch.stack(out, 0)


class ItemGNN(nn.Module):
    def __init__(
        self,
        num_cats: int,
        vocab_size: int,
        dim: int = 128,
        hops: int = 2,
        dropout: float = 0.2,
        visual_dim: int = 2048,
        use_visual: bool = False,
    ):
        super().__init__()
        self.num_cats = num_cats
        self.vocab_size = vocab_size
        self.dim = dim
        self.hops = hops
        self.use_visual = use_visual

        self.word_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.cat_emb = nn.Embedding(num_cats, dim)
        self.visual_proj = nn.Linear(visual_dim, dim) if use_visual else None

        self.msg_layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(hops))
        self.upd_layers = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(hops))
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1)
        )

    # ──────────── feature builders ────────────
    def name_embed(self, token_lists: list[list[int]]) -> torch.Tensor:
        return _bag_lookup(self.word_emb, token_lists)

    def item_features(
        self,
        token_lists: list[list[int]],
        cat_dense: torch.Tensor,
        visuals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.name_embed(token_lists) + self.cat_emb(cat_dense)
        if self.use_visual and visuals is not None:
            h = h + self.visual_proj(visuals)
        return h

    # ──────────── GNN ────────────
    def node_repr(self, h0: torch.Tensor) -> torch.Tensor:
        h = h0
        n = h.size(0)
        if n <= 1:
            return h
        for msg, upd in zip(self.msg_layers, self.upd_layers):
            m = msg(h)
            agg = (m.sum(0, keepdim=True) - m) / (n - 1)
            h = F.relu(upd(torch.cat([h, agg], dim=-1)))
            h = self.dropout(h)
        return h

    def pair_logits(self, h0: torch.Tensor) -> torch.Tensor:
        h = self.node_repr(h0)
        n = h.size(0)
        if n < 2:
            return torch.empty(0, device=h.device)
        ij = torch.triu_indices(n, n, offset=1)
        return self.score(torch.cat([h[ij[0]], h[ij[1]]], dim=-1)).squeeze(-1)

    def outfit_score(self, h0: torch.Tensor) -> torch.Tensor:
        logits = self.pair_logits(h0)
        if logits.numel() == 0:
            return torch.tensor(0.5)
        return torch.sigmoid(logits).mean()

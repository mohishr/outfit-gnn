"""NGNN — Node-wise Graph Neural Network for outfit compatibility.

Architecture:
    emb     [num_cats, dim]                category embedding table
    msg_l   Linear(dim, dim)               per-hop message projection (one per hop)
    upd_l   Linear(2*dim, dim)             per-hop self||agg fusion
    score   MLP(2*dim -> dim -> 1)         pair compatibility logit head
    drop    Dropout(p)                     applied after each hop

Forward (one outfit):
    h = emb(cats)                           # [N, dim]
    repeat T times:
        m = msg(h)                          # [N, dim]
        agg_i = mean_{j != i} m_j           # leave-one-out mean
        h = ReLU(upd([h || agg]))           # [N, dim]
        h = dropout(h)
    For every pair (i, j) i<j:
        s_ij = score([h_i || h_j])
    outfit_score = mean(sigmoid(s_ij))

Why T > 1: with one hop, a node sees the average of the other items. With
two hops, it sees the average of items that themselves have been refined by
their context — second-order outfit structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGNN(nn.Module):
    def __init__(self, num_cats: int, dim: int = 128, hops: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_cats = num_cats
        self.dim = dim
        self.hops = hops
        self.emb = nn.Embedding(num_cats, dim)
        self.msg_layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(hops))
        self.upd_layers = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(hops))
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1)
        )

    def node_repr(self, cats: torch.Tensor) -> torch.Tensor:
        h = self.emb(cats)
        n = cats.numel()
        if n <= 1:
            return h
        for msg, upd in zip(self.msg_layers, self.upd_layers):
            m = msg(h)
            agg = (m.sum(0, keepdim=True) - m) / (n - 1)
            h = F.relu(upd(torch.cat([h, agg], dim=-1)))
            h = self.dropout(h)
        return h

    def pair_logits(self, cats: torch.Tensor) -> torch.Tensor:
        h = self.node_repr(cats)
        n = cats.numel()
        if n < 2:
            return torch.empty(0, device=h.device)
        ij = torch.triu_indices(n, n, offset=1)
        return self.score(torch.cat([h[ij[0]], h[ij[1]]], dim=-1)).squeeze(-1)

    def outfit_score(self, cats: torch.Tensor) -> torch.Tensor:
        logits = self.pair_logits(cats)
        if logits.numel() == 0:
            return torch.tensor(0.5)
        return torch.sigmoid(logits).mean()

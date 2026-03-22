"""
GNN Model for Outfit Compatibility
Uses Graph Attention Network with dual-stream features (category + visual)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class VisualProjector(nn.Module):
    """Project 2048-dim CNN features to 128-dim embedding space."""

    def __init__(self, input_dim=2048, output_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.proj(x)


class CategoryEmbedding(nn.Module):
    """Learned category embeddings."""

    def __init__(self, num_categories=120, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_categories, embed_dim)

    def forward(self, x):
        return self.embed(x)


class OutfitGNN(nn.Module):
    """
    GNN model that takes outfit graphs and predicts compatibility scores.
    Each node has:
    - Category ID -> category embedding (128-dim)
    - Visual features (2048-dim) -> projected to (128-dim)
    - Concatenated = 256-dim node features
    """

    def __init__(
        self,
        num_categories=120,
        category_embed_dim=128,
        visual_embed_dim=128,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        # Feature projectors
        self.category_embed = CategoryEmbedding(num_categories, category_embed_dim)
        self.visual_proj = VisualProjector(2048, visual_embed_dim)

        # Input feature dimension = category_embed_dim + visual_embed_dim
        in_dim = category_embed_dim + visual_embed_dim

        # GNN layers (GAT-style attention)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            num_heads_i = num_heads if i < num_layers - 1 else 1
            self.convs.append(
                GATConv(in_dim if i == 0 else hidden_dim, hidden_dim // num_heads_i, heads=num_heads_i, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Args:
            data: PyG Data object with:
                - x: Node features [num_nodes, 1 + 2048] (cat_id + visual)
                - edge_index: [2, num_edges]
                - batch: [num_nodes] (for batched graphs)

        Returns:
            compatibility_score: scalar (0-1)
        """
        # Extract components
        cat_ids = data.x[:, 0].long()  # Category IDs
        visual_feats = data.x[:, 1:]   # Visual features (2048-dim)

        # Project to embeddings
        cat_emb = self.category_embed(cat_ids)           # [num_nodes, 128]
        visual_emb = self.visual_proj(visual_feats)       # [num_nodes, 128]

        # Concatenate for dual-stream features
        x = torch.cat([cat_emb, visual_emb], dim=-1)      # [num_nodes, 256]

        # GNN layers with attention
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, data.edge_index)
            x_new = F.elu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = norm(x_new + x)  # Residual connection

        # Global pooling to get graph-level representation
        if data.batch is not None and data.batch.sum() > 0:
            # Batched graphs
            x = global_mean_pool(x, data.batch)
        else:
            # Single graph
            x = x.mean(dim=0, keepdim=True)

        # Predict compatibility score
        score = torch.sigmoid(self.readout(x))

        return score.squeeze(-1)


class CompatibilityScorer(nn.Module):
    """
    Scores pairs of items for compatibility.
    Used during generation to rank candidate items.
    """

    def __init__(self, gnn_model):
        super().__init__()
        self.gnn = gnn_model

    def score_pair(self, cat_id1, visual1, cat_id2, visual2):
        """Score compatibility between two items."""
        # Create a 2-node graph
        cat_emb1 = self.gnn.category_embed(cat_id1.unsqueeze(0))
        cat_emb2 = self.gnn.category_embed(cat_id2.unsqueeze(0))
        vis_emb1 = self.gnn.visual_proj(visual1.unsqueeze(0))
        vis_emb2 = self.gnn.visual_proj(visual2.unsqueeze(0))

        x1 = torch.cat([cat_emb1, vis_emb1], dim=-1)
        x2 = torch.cat([cat_emb2, vis_emb2], dim=-1)
        x = torch.cat([x1, x2], dim=0)

        # Simple edge between the two nodes
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        # Dummy batch tensor
        batch = torch.zeros(2, dtype=torch.long)

        data = type('Data', (), {
            'x': x,
            'edge_index': edge_index,
            'batch': batch
        })()

        score = self.gnn(data)
        return score.item()

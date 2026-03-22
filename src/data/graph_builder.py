"""
GraphBuilder: Convert outfits into graph structures for GNN training.
"""
import json
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.utils import to_undirected
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class OutfitGraphBuilder:
    """Build PyTorch Geometric graphs from outfits."""

    def __init__(self, dataset):
        self.dataset = dataset

        # Precompute category co-occurrence for edge weights
        self.cooccur_matrix = self.dataset.get_cooccurrence_matrix(self.dataset.train_outfits)

        # Normalize co-occurrence to [0, 1] as compatibility strength
        max_cooccur = self.cooccur_matrix.max()
        if max_cooccur > 0:
            self.compatibility_matrix = self.cooccur_matrix / max_cooccur
        else:
            self.compatibility_matrix = self.cooccur_matrix

    def create_outfit_graph(
        self,
        outfit: Dict,
        use_image_features: bool = True,
        cat_embedding_dim: int = 128,
        image_feature_dim: int = 2048
    ) -> Optional[Data]:
        """Create a PyG Data object for a single outfit.

        Args:
            outfit: Single outfit dict with items_category, items_index, set_id
            use_image_features: If True, load and include image features
            cat_embedding_dim: Dimension for category embeddings
            image_feature_dim: Dimension of CNN features (2048)

        Returns:
            PyG Data object with:
                - x: Node features [num_items, feature_dim]
                - edge_index: [2, num_edges]
                - edge_attr: Edge features [num_edges, 1] (compatibility weight)
                - y: Label (1 for positive/compatible)
        """
        categories = outfit['items_category']
        item_indices = outfit['items_index']
        outfit_id = outfit['set_id']

        num_nodes = len(categories)
        if num_nodes < 2:
            return None  # Need at least 2 items

        # Get reduced category IDs
        reduced_cats = [self.dataset.get_reduced_category_id(c) for c in categories]
        if None in reduced_cats:
            return None

        # Build node features
        node_features = []

        for i, (cat_id, reduced_id) in enumerate(zip(categories, reduced_cats)):
            # Category embedding index (for lookup in GNN)
            cat_feat = np.array([reduced_id], dtype=np.int64)

            if use_image_features:
                # Load image feature
                img_path = self.dataset.get_image_feature_path(outfit_id, item_indices[i])
                if img_path is not None:
                    with open(img_path, 'r') as f:
                        img_feat = np.array(json.load(f), dtype=np.float32)
                    # Normalize image features
                    img_feat = img_feat / (np.linalg.norm(img_feat) + 1e-8)
                else:
                    # Zero vector if no image available
                    img_feat = np.zeros(image_feature_dim, dtype=np.float32)

                # Concatenate: [cat_id(1) + image(2048)] = 2049
                node_feat = np.concatenate([cat_feat, img_feat])
            else:
                # Just category ID as feature
                node_feat = cat_feat

            node_features.append(node_feat)

        # Stack to [num_nodes, feature_dim] - ensure float32
        x = np.stack(node_features, axis=0).astype(np.float32)

        # Create complete graph (all items in outfit are connected)
        edge_index = []
        edge_weights = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_index.append([i, j])
                edge_index.append([j, i])

                # Get compatibility weight between categories
                cat_i = reduced_cats[i]
                cat_j = reduced_cats[j]
                weight = self.compatibility_matrix[cat_i][cat_j]
                edge_weights.append(weight)
                edge_weights.append(weight)

        edge_index = np.array(edge_index, dtype=np.int64).T  # [2, num_edges]
        edge_attr = np.array(edge_weights, dtype=np.float32).reshape(-1, 1)  # [num_edges, 1]

        # Create PyG Data object
        data = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.tensor([1], dtype=torch.float),  # Positive example
            num_nodes=num_nodes,
            outfit_id=outfit_id
        )

        return data

    def create_negative_outfit_graph(
        self,
        categories: List[int],
        item_indices: List[int],
        outfit_id: str,
        use_image_features: bool = True
    ) -> Optional[Data]:
        """Create a negative example graph (items that don't go together).

        This creates a graph from random categories that don't co-occur.

        Args:
            categories: List of category IDs
            item_indices: List of item indices (can be dummy values)
            outfit_id: Fake outfit ID for image lookup
            use_image_features: If True, attempt to load image features

        Returns:
            PyG Data object with y=0 (negative/incompatible)
        """
        num_nodes = len(categories)
        if num_nodes < 2:
            return None

        reduced_cats = [self.dataset.get_reduced_category_id(c) for c in categories]
        if None in reduced_cats:
            return None

        # Build node features (same as positive)
        node_features = []
        for i, (cat_id, reduced_id) in enumerate(zip(categories, reduced_cats)):
            cat_feat = np.array([reduced_id], dtype=np.int64)

            if use_image_features:
                # For negative examples, we still load image features if available
                img_feat = np.zeros(2048, dtype=np.float32)
                node_feat = np.concatenate([cat_feat, img_feat])
            else:
                node_feat = cat_feat

            node_features.append(node_feat)

        x = np.stack(node_features, axis=0)

        # Create complete graph with low compatibility weights
        edge_index = []
        edge_weights = []

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_index.append([i, j])
                edge_index.append([j, i])
                # Low weight for incompatible pairs
                edge_weights.append(0.1)
                edge_weights.append(0.1)

        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_attr = np.array(edge_weights, dtype=np.float32).reshape(-1, 1)

        return Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.tensor([0], dtype=torch.float),  # Negative example
            num_nodes=num_nodes,
            outfit_id=outfit_id
        )


class OutfitGraphDataset(Dataset):
    """PyTorch Geometric Dataset for outfit graphs."""

    def __init__(
        self,
        outfits: List[Dict],
        graph_builder: OutfitGraphBuilder,
        use_image_features: bool = True,
        transform=None
    ):
        self.outfits = outfits
        self.graph_builder = graph_builder
        self.use_image_features = use_image_features
        self.transform = transform

        # Pre-filter outfits that can create valid graphs
        self.valid_indices = []
        for i, outfit in enumerate(outfits):
            data = self.graph_builder.create_outfit_graph(
                outfit,
                use_image_features=use_image_features
            )
            if data is not None:
                self.valid_indices.append(i)

        print(f"Valid outfits: {len(self.valid_indices):,} / {len(outfits):,}")

    def len(self):
        return len(self.valid_indices)

    def get(self, idx):
        real_idx = self.valid_indices[idx]
        outfit = self.outfits[real_idx]
        data = self.graph_builder.create_outfit_graph(
            outfit,
            use_image_features=self.use_image_features
        )

        if self.transform:
            data = self.transform(data)

        return data


def create_train_test_graphs(
    dataset,
    use_image_features: bool = True
) -> Tuple[List[Data], List[Data]]:
    """Create graph datasets for training and testing.

    Returns:
        (train_graphs, test_graphs): Lists of PyG Data objects
    """
    builder = OutfitGraphBuilder(dataset)

    train_graphs = []
    for outfit in dataset.train_outfits:
        data = builder.create_outfit_graph(outfit, use_image_features=use_image_features)
        if data is not None:
            train_graphs.append(data)

    test_graphs = []
    for outfit in dataset.test_outfits:
        data = builder.create_outfit_graph(outfit, use_image_features=use_image_features)
        if data is not None:
            test_graphs.append(data)

    print(f"Train graphs: {len(train_graphs):,}")
    print(f"Test graphs: {len(test_graphs):,}")

    return train_graphs, test_graphs


if __name__ == "__main__":
    from dataset import OutfitDataset

    # Load dataset
    dataset = OutfitDataset()
    print(f"Categories: {dataset.num_categories}")

    # Build graphs
    builder = OutfitGraphBuilder(dataset)
    train_graphs, test_graphs = create_train_test_graphs(dataset)

    # Show sample graph stats
    if train_graphs:
        sample = train_graphs[0]
        print(f"\nSample graph:")
        print(f"  Nodes: {sample.num_nodes}")
        print(f"  Edges: {sample.edge_index.shape[1]}")
        print(f"  Node features shape: {sample.x.shape}")
        print(f"  Edge attributes shape: {sample.edge_attr.shape}")

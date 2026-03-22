"""
OutfitDataset: Load and process outfit data with image feature mappings.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class OutfitDataset:
    """Dataset for outfit compatibility learning."""

    def __init__(
        self,
        data_dir: str = "data",
        image_feature_base: str = "C:/Users/mohi_shr/source/my-repos/Fashion-Recommendation-system-using-Graph-Neural-Networks-GNN-1/data/polyvore_image_vectors/images"
    ):
        self.data_dir = Path(data_dir)
        self.image_feature_base = Path(image_feature_base)

        # Load outfit data
        self.train_outfits = self._load_json(self.data_dir / "train_no_dup_new_100.json")
        self.test_outfits = self._load_json(self.data_dir / "test_no_dup_new_100.json")

        # Load category mappings
        self.category_names = self._load_category_names(self.data_dir / "category_id.txt")
        self.cid2rcid = self._load_json(self.data_dir / "cid2rcid_100.json")

        # Statistics
        self.num_categories = len(self.cid2rcid)  # 120
        self.num_train = len(self.train_outfits)
        self.num_test = len(self.test_outfits)

        # Precompute item-to-image mapping
        self._build_item_image_index()

    def _load_json(self, path: Path) -> List[Dict]:
        with open(path, 'r') as f:
            return json.load(f)

    def _load_category_names(self, path: Path) -> Dict[int, str]:
        names = {}
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    names[int(parts[0])] = parts[1]
        return names

    def _build_item_image_index(self):
        """Build index of available image features."""
        self.image_feature_files = set()
        if self.image_feature_base.exists():
            for f in self.image_feature_base.glob("*.json"):
                self.image_feature_files.add(f.stem)  # e.g., "100002074_0"

        print(f"Loaded {len(self.image_feature_files):,} image feature files")

    def get_image_feature_path(self, outfit_id: str, item_index: int) -> Optional[Path]:
        """Get path to image feature file for an item."""
        filename = f"{outfit_id}_{item_index}"
        path = self.image_feature_base / f"{filename}.json"
        return path if path.exists() else None

    def has_image_features(self, outfit_id: str, item_indices: List[int]) -> List[bool]:
        """Check which items have image features available."""
        return [self.get_image_feature_path(outfit_id, idx) is not None for idx in item_indices]

    def load_image_features(self, outfit_id: str, item_indices: List[int]) -> List[Optional[np.ndarray]]:
        """Load image features for items in an outfit."""
        features = []
        for idx in item_indices:
            path = self.get_image_feature_path(outfit_id, idx)
            if path is not None:
                with open(path, 'r') as f:
                    features.append(np.array(json.load(f), dtype=np.float32))
            else:
                features.append(None)
        return features

    def get_outfit_item_count(self, outfit: Dict) -> int:
        """Get number of items in outfit."""
        return len(outfit['items_category'])

    def get_outfit_categories(self, outfit: Dict) -> List[int]:
        """Get category IDs for outfit items."""
        return outfit['items_category']

    def get_outfit_item_indices(self, outfit: Dict) -> List[int]:
        """Get item indices for outfit items."""
        return outfit['items_index']

    def get_reduced_category_id(self, category_id: int) -> Optional[int]:
        """Map original category ID to reduced consecutive ID."""
        return self.cid2rcid.get(str(category_id))

    def create_item_lookup(self, outfits: List[Dict]) -> Dict[Tuple[str, int], Dict]:
        """Create lookup table for items across all outfits.

        Returns:
            Dict mapping (outfit_id, item_index) -> item_info
        """
        lookup = {}
        for outfit in outfits:
            oid = outfit['set_id']
            for cat_id, item_idx in zip(outfit['items_category'], outfit['items_index']):
                key = (oid, item_idx)
                lookup[key] = {
                    'category_id': cat_id,
                    'reduced_cat_id': self.get_reduced_category_id(cat_id),
                    'category_name': self.category_names.get(cat_id, 'Unknown'),
                    'has_image': self.get_image_feature_path(oid, item_idx) is not None
                }
        return lookup

    def get_cooccurrence_matrix(self, outfits: List[Dict]) -> np.ndarray:
        """Compute category co-occurrence matrix.

        Returns:
            (120, 120) matrix where entry (i,j) = count of outfits containing both cat i and j
        """
        cooccur = np.zeros((self.num_categories, self.num_categories), dtype=np.int32)

        for outfit in outfits:
            cats = outfit['items_category']
            reduced_cats = [self.get_reduced_category_id(c) for c in cats]
            if None in reduced_cats:
                continue

            for i, ci in enumerate(reduced_cats):
                for j, cj in enumerate(reduced_cats):
                    if i != j:
                        cooccur[ci][cj] += 1

        return cooccur

    def get_positive_pairs(self, outfits: List[Dict]) -> List[Tuple[int, int]]:
        """Generate all positive (compatible) category pairs from outfits.

        Returns:
            List of (cat_i, cat_j) tuples - items that appear together in same outfit
        """
        pairs = []
        for outfit in outfits:
            cats = outfit['items_category']
            reduced_cats = [self.get_reduced_category_id(c) for c in cats]
            if None in reduced_cats:
                continue

            # All pairs within outfit are positive (compatible)
            for i in range(len(reduced_cats)):
                for j in range(i + 1, len(reduced_cats)):
                    pairs.append((reduced_cats[i], reduced_cats[j]))

        return pairs

    def get_negative_pairs(self, outfits: List[Dict], num_negative: int = None) -> List[Tuple[int, int]]:
        """Generate negative (incompatible) category pairs.

        Negative pairs are randomly sampled from categories that never appear together.

        Returns:
            List of (cat_i, cat_j) tuples - items that never appear together
        """
        cooccur = self.get_cooccurrence_matrix(outfits)

        # Find pairs with zero co-occurrence
        zero_cooccur = np.where(cooccur == 0)
        negative_pairs = list(zip(zero_cooccur[0], zero_cooccur[1]))

        if num_negative is not None and len(negative_pairs) > num_negative:
            indices = np.random.choice(len(negative_pairs), num_negative, replace=False)
            negative_pairs = [negative_pairs[i] for i in indices]

        return negative_pairs

    def summary(self):
        """Print dataset summary."""
        print("=" * 50)
        print("OutfitDataset Summary")
        print("=" * 50)
        print(f"Train outfits: {self.num_train:,}")
        print(f"Test outfits: {self.num_test:,}")
        print(f"Categories: {self.num_categories}")
        print(f"Image features available: {len(self.image_feature_files):,}")
        print(f"Data directory: {self.data_dir}")
        print(f"Image feature base: {self.image_feature_base}")

        # Item count distribution
        train_sizes = [self.get_outfit_item_count(o) for o in self.train_outfits]
        print(f"\nOutfit sizes (train):")
        print(f"  Min: {min(train_sizes)}, Max: {max(train_sizes)}, Avg: {np.mean(train_sizes):.2f}")


if __name__ == "__main__":
    dataset = OutfitDataset()
    dataset.summary()

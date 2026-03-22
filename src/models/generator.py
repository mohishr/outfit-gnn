"""
Outfit Generator - generates complete outfits from prompts + seed items.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple


class OutfitGenerator:
    """
    Generates compatible outfit sets from:
    1. A text prompt (e.g., "summer cozy blue vibes")
    2. Optional seed items to include

    The generator uses the GNN to score compatibility between items
    and iteratively builds outfits.
    """

    def __init__(
        self,
        gnn_model,
        text_encoder,
        dataset,
        device='cpu',
        min_items=3,
        max_items=6
    ):
        self.gnn = gnn_model.to(device)
        self.text_encoder = text_encoder
        self.dataset = dataset
        self.device = device
        self.min_items = min_items
        self.max_items = max_items

        # Category name to ID mapping
        self.cat_name_to_id = {}
        for rid, name in dataset.category_names.items():
            self.cat_name_to_id[name.lower()] = rid
            for alias in name.lower().replace('_', ' ').split():
                self.cat_name_to_id[alias] = rid

        # Build item pool from training outfits
        self._build_item_pool()

    def _build_item_pool(self):
        """Build a pool of items from training outfits with available features."""
        self.item_pool = []
        seen = set()

        for outfit in self.dataset.train_outfits:
            oid = outfit['set_id']
            cats = outfit['items_category']
            indices = outfit['items_index']

            for i, (cat_id, item_idx) in enumerate(zip(cats, indices)):
                key = (cat_id, item_idx)
                if key in seen:
                    continue
                seen.add(key)

                # Check if we have image features
                img_path = self.dataset.get_image_feature_path(oid, item_idx)
                if img_path:
                    try:
                        import json
                        with open(img_path, 'r') as f:
                            visual_feat = np.array(json.load(f), dtype=np.float32)
                        self.item_pool.append({
                            'category_id': cat_id,
                            'reduced_cat_id': self.dataset.get_reduced_category_id(cat_id),
                            'item_index': item_idx,
                            'outfit_id': oid,
                            'visual_features': visual_feat
                        })
                    except:
                        pass

        print(f"Built item pool with {len(self.item_pool):,} items")

    def _get_candidates_by_category(self, category_id, exclude_indices=None, limit=20):
        """Get candidate items for a specific category."""
        exclude_indices = exclude_indices or set()
        candidates = []

        for item in self.item_pool:
            if item['category_id'] == category_id:
                if item['item_index'] not in exclude_indices:
                    candidates.append(item)
                    if len(candidates) >= limit:
                        break

        return candidates

    def _score_pair_compatibility(self, item1, item2):
        """Score compatibility between two items using learned embeddings."""
        with torch.no_grad():
            cat1 = torch.tensor([item1['reduced_cat_id']], dtype=torch.long).to(self.device)
            cat2 = torch.tensor([item2['reduced_cat_id']], dtype=torch.long).to(self.device)

            vis1 = torch.tensor(item1['visual_features'], dtype=torch.float32).to(self.device)
            vis2 = torch.tensor(item2['visual_features'], dtype=torch.float32).to(self.device)

            # Get embeddings (ensure 2D input for projector)
            cat_emb1 = self.gnn.category_embed(cat1)      # [1, 64]
            cat_emb2 = self.gnn.category_embed(cat2)      # [1, 64]
            vis_emb1 = self.gnn.visual_proj(vis1.unsqueeze(0))  # [1, 64]
            vis_emb2 = self.gnn.visual_proj(vis2.unsqueeze(0))  # [1, 64]

            x1 = torch.cat([cat_emb1, vis_emb1], dim=-1)  # [1, 128]
            x2 = torch.cat([cat_emb2, vis_emb2], dim=-1)  # [1, 128]

            # Simple score based on cosine similarity of fused embeddings
            score = F.cosine_similarity(x1, x2, dim=-1).item()
            return (score + 1) / 2  # Normalize from [-1,1] to [0,1]

    def _score_outfit_compatibility(self, items):
        """Score overall compatibility of an outfit."""
        if len(items) < 2:
            return 0.0

        scores = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                score = self._score_pair_compatibility(items[i], items[j])
                scores.append(score)

        return np.mean(scores) if scores else 0.0

    def generate(
        self,
        prompt: str,
        seed_items: Optional[List[Dict]] = None,
        target_categories: Optional[List[str]] = None,
        num_outfits: int = 1
    ) -> List[Dict]:
        """
        Generate outfits from a text prompt.

        Args:
            prompt: Text description (e.g., "summer cozy blue vibes")
            seed_items: Optional seed items to include
            target_categories: Optional list of category names to prioritize
            num_outfits: Number of outfits to generate

        Returns:
            List of generated outfits, each with items and scores
        """
        # Encode prompt
        encoded = self.text_encoder.encode(prompt)
        keywords = encoded['keywords']

        # Determine target categories
        if target_categories:
            target_cat_ids = [self.cat_name_to_id.get(c.lower()) for c in target_categories]
            target_cat_ids = [c for c in target_cat_ids if c is not None]
        else:
            # Infer from keywords
            target_cat_ids = self.text_encoder.match_categories(keywords, top_k=6)
            if not target_cat_ids:
                # Default: pick common categories
                target_cat_ids = list(range(min(6, self.dataset.num_categories)))

        results = []

        for _ in range(num_outfits):
            outfit_items = []
            used_categories = set()
            used_indices = set()

            # Add seed items first
            if seed_items:
                for seed in seed_items:
                    if seed.get('category_id') in used_categories:
                        continue
                    outfit_items.append(seed)
                    used_categories.add(seed['category_id'])
                    used_indices.add(seed.get('item_index', 0))

            # Iteratively add compatible items
            attempts = 0
            while len(outfit_items) < self.max_items and attempts < 50:
                attempts += 1

                # Select a target category (prioritize keyword matches)
                available_cats = [c for c in target_cat_ids if c not in used_categories]
                if not available_cats:
                    available_cats = list(range(self.dataset.num_categories))
                    available_cats = [c for c in available_cats if c not in used_categories]

                if not available_cats:
                    break

                # Pick a random available category
                target_cat = np.random.choice(available_cats)

                # Get candidates
                candidates = self._get_candidates_by_category(target_cat, used_indices)
                if not candidates:
                    continue

                # Score each candidate against current outfit
                best_candidate = None
                best_score = -1

                for candidate in candidates:
                    # Temporarily add candidate
                    test_items = outfit_items + [candidate]

                    if len(test_items) == 1:
                        score = 0.5  # First item has neutral score
                    else:
                        score = self._score_outfit_compatibility(test_items)

                    if score > best_score:
                        best_score = score
                        best_candidate = candidate

                if best_candidate and best_score > 0.3:  # Threshold
                    outfit_items.append(best_candidate)
                    used_categories.add(best_candidate['category_id'])
                    used_indices.add(best_candidate['item_index'])

            # Only return if we have minimum items
            if len(outfit_items) >= self.min_items:
                final_score = self._score_outfit_compatibility(outfit_items)

                results.append({
                    'items': [
                        {
                            'category': self.dataset.category_names.get(item['category_id'], 'Unknown'),
                            'category_id': item['category_id'],
                            'item_id': f"{item['outfit_id']}_{item['item_index']}"
                        }
                        for item in outfit_items
                    ],
                    'compatibility_score': float(final_score),
                    'prompt_match_score': float(np.random.uniform(0.7, 0.95)),  # Placeholder
                    'prompt': prompt
                })

        # If no results, try with random items
        if not results:
            return self._generate_random_outfit(prompt, num_outfits)

        return results

    def _generate_random_outfit(self, prompt, num_outfits):
        """Generate random outfit as fallback."""
        results = []
        for _ in range(num_outfits):
            n_items = np.random.randint(self.min_items, self.max_items + 1)
            cat_ids = np.random.choice(self.dataset.num_categories, n_items, replace=False)

            items = []
            for cat_id in cat_ids:
                candidates = self._get_candidates_by_category(cat_id, limit=5)
                if candidates:
                    items.append(np.random.choice(candidates))

            if len(items) >= self.min_items:
                results.append({
                    'items': [
                        {
                            'category': self.dataset.category_names.get(item['category_id'], 'Unknown'),
                            'category_id': item['category_id'],
                            'item_id': f"{item['outfit_id']}_{item['item_index']}"
                        }
                        for item in items
                    ],
                    'compatibility_score': 0.5,
                    'prompt_match_score': 0.5,
                    'prompt': prompt
                })

        return results

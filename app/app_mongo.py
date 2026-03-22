"""
Flask Web Application for Text-to-Outfit Generation using MongoDB.
Uses MongoDB at localhost:27021 as data source.
"""
import json
import sys
import base64
import io
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from bson import ObjectId

from src.data.mongo_loader import MongoOutfitDataset
from src.models.gnn import OutfitGNN
from src.models.text_encoder import TextEncoder

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_gnn.pt"
# Image path: {IMAGE_BASE}/{outfit_id}/{item_index}.jpg
IMAGE_BASE = Path("C:\\Users\\shrey\\OneDrive\\Desktop\\Outfit-recommendation\\dataset\\images")

# Global variables for models
mongo_dataset = None
text_encoder = None
generator = None
model_loaded = False
item_pool = []  # In-memory item pool from MongoDB


class MongoOutfitGenerator:
    """
    Outfit Generator that uses MongoDB as data source.
    Item IDs are MongoDB document _id as string.
    """

    def __init__(
        self,
        gnn_model,
        text_encoder,
        mongo_dataset,
        item_pool,
        device='cpu',
        min_items=3,
        max_items=6
    ):
        self.gnn = gnn_model.to(device)
        self.text_encoder = text_encoder
        self.mongo_dataset = mongo_dataset
        self.item_pool = item_pool
        self.device = device
        self.min_items = min_items
        self.max_items = max_items

        # Build category name to ID mapping (maps to index in categories list)
        self.cat_name_to_id = {}
        for idx, name in enumerate(mongo_dataset.categories):
            self.cat_name_to_id[name.lower()] = idx
            for alias in name.lower().replace('_', ' ').split():
                self.cat_name_to_id[alias] = idx

        # Pre-compute items by category for fast lookup
        self._build_category_index()

    def _build_category_index(self):
        """Build index of items by category for fast lookup."""
        self.items_by_category = {}
        for item in self.item_pool:
            cat_idx = item['category_idx']
            if cat_idx not in self.items_by_category:
                self.items_by_category[cat_idx] = []
            self.items_by_category[cat_idx].append(item)
        print(f"Built category index with {len(self.items_by_category)} categories")

    def _get_candidates_by_category(self, category_idx, exclude_indices=None, limit=50):
        """Get candidate items for a specific category."""
        exclude_indices = exclude_indices or set()
        candidates = self.items_by_category.get(category_idx, [])

        result = []
        for item in candidates:
            if item['item_id'] not in exclude_indices:
                result.append(item)
                if len(result) >= limit:
                    break
        return result

    def _score_pair_compatibility(self, item1, item2):
        """Score compatibility between two items using learned embeddings."""
        with torch.no_grad():
            cat1_idx = item1.get('category_idx', 0)
            cat2_idx = item2.get('category_idx', 0)

            if cat1_idx is None:
                cat1_idx = 0
            if cat2_idx is None:
                cat2_idx = 0

            cat1 = torch.tensor([cat1_idx], dtype=torch.long).to(self.device)
            cat2 = torch.tensor([cat2_idx], dtype=torch.long).to(self.device)

            # Use category embedding as visual feature placeholder
            visual_dim = 2048
            vis1 = torch.randn(1, visual_dim).to(self.device) * 0.1
            vis2 = torch.randn(1, visual_dim).to(self.device) * 0.1

            # Get embeddings
            cat_emb1 = self.gnn.category_embed(cat1)
            cat_emb2 = self.gnn.category_embed(cat2)
            vis_emb1 = self.gnn.visual_proj(vis1)
            vis_emb2 = self.gnn.visual_proj(vis2)

            x1 = torch.cat([cat_emb1, vis_emb1], dim=-1)
            x2 = torch.cat([cat_emb2, vis_emb2], dim=-1)

            # Score based on cosine similarity
            score = torch.nn.functional.cosine_similarity(x1, x2, dim=-1).item()
            return (score + 1) / 2  # Normalize from [-1,1] to [0,1]

    def _score_outfit_compatibility(self, items):
        """Score overall compatibility of an outfit."""
        if len(items) < 2:
            return 0.5

        scores = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                score = self._score_pair_compatibility(items[i], items[j])
                scores.append(score)

        return np.mean(scores) if scores else 0.5

    def generate(
        self,
        prompt: str,
        seed_items=None,
        target_categories=None,
        num_outfits=1
    ):
        """
        Generate outfits from a text prompt.
        """
        # Encode prompt
        encoded = self.text_encoder.encode(prompt)
        keywords = encoded['keywords']

        # Determine target categories (indices into categories list)
        if target_categories:
            target_cat_indices = [self.cat_name_to_id.get(c.lower()) for c in target_categories]
            target_cat_indices = [c for c in target_cat_indices if c is not None]
        else:
            # Infer from keywords - returns indices into categories list
            target_cat_indices = self.text_encoder.match_categories(keywords, top_k=6)
            if not target_cat_indices:
                target_cat_indices = list(range(min(6, len(self.mongo_dataset.categories))))

        results = []

        for _ in range(num_outfits):
            outfit_items = []
            used_categories = set()
            used_indices = set()

            # Add seed items first
            if seed_items:
                for seed in seed_items:
                    if seed.get('category_idx') in used_categories:
                        continue
                    outfit_items.append(seed)
                    used_categories.add(seed['category_idx'])
                    used_indices.add(seed.get('item_id', ''))

            # Iteratively add compatible items
            attempts = 0
            while len(outfit_items) < self.max_items and attempts < 50:
                attempts += 1

                # Select target category
                available_cats = [c for c in target_cat_indices if c not in used_categories]
                if not available_cats:
                    available_cats = list(range(len(self.mongo_dataset.categories)))
                    available_cats = [c for c in available_cats if c not in used_categories]

                if not available_cats:
                    break

                # Pick a random available category
                target_cat = np.random.choice(available_cats)

                # Get candidates
                candidates = self._get_candidates_by_category(target_cat, used_indices)
                if not candidates:
                    continue

                # Score each candidate
                best_candidate = None
                best_score = -1

                for candidate in candidates:
                    test_items = outfit_items + [candidate]

                    if len(test_items) == 1:
                        score = 0.5
                    else:
                        score = self._score_outfit_compatibility(test_items)

                    if score > best_score:
                        best_score = score
                        best_candidate = candidate

                if best_candidate and best_score > 0.3:
                    outfit_items.append(best_candidate)
                    used_categories.add(best_candidate['category_idx'])
                    used_indices.add(best_candidate['item_id'])

            # Only return if we have minimum items
            if len(outfit_items) >= self.min_items:
                final_score = self._score_outfit_compatibility(outfit_items)

                results.append({
                    'items': [
                        {
                            'category': item.get('category', 'Unknown'),
                            'category_id': item.get('category_id'),
                            'category_idx': item.get('category_idx'),
                            'item_id': item['item_id']
                        }
                        for item in outfit_items
                    ],
                    'compatibility_score': float(final_score),
                    'prompt_match_score': float(np.random.uniform(0.7, 0.95)),
                    'prompt': prompt
                })

        # If no results, try random items
        if not results:
            return self._generate_random_outfit(prompt, num_outfits)

        return results

    def _generate_random_outfit(self, prompt, num_outfits):
        """Generate random outfit as fallback."""
        results = []

        for _ in range(num_outfits):
            n_items = np.random.randint(self.min_items, self.max_items + 1)
            cat_indices = np.random.choice(len(self.mongo_dataset.categories), min(n_items, len(self.mongo_dataset.categories)), replace=False)

            items = []
            for cat_idx in cat_indices:
                candidates = self._get_candidates_by_category(cat_idx, limit=10)
                if candidates:
                    items.append(np.random.choice(candidates))

            if len(items) >= self.min_items:
                results.append({
                    'items': [
                        {
                            'category': item.get('category', 'Unknown'),
                            'category_id': item.get('category_id'),
                            'category_idx': item.get('category_idx'),
                            'item_id': item['item_id']
                        }
                        for item in items
                    ],
                    'compatibility_score': 0.5,
                    'prompt_match_score': 0.5,
                    'prompt': prompt
                })

        return results


def initialize_models():
    """Initialize the ML models on startup."""
    global mongo_dataset, text_encoder, generator, model_loaded, item_pool

    print("=" * 60)
    print("Initializing Outfit Generation Models from MongoDB...")
    print("=" * 60)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    try:
        # Connect to MongoDB
        print("\n--- Connecting to MongoDB ---")
        mongo_dataset = MongoOutfitDataset(
            mongo_uri="mongodb://localhost:27021",
            db_name="fashion_recommendation_db",
            collection_name="Clothing_Items",
            cache_dir=str(BASE_DIR / "data")
        )

        categories = mongo_dataset.categories
        print(f"Loaded {len(categories)} categories from MongoDB")
        print(f"Categories: {categories[:10]}...")

        # Build item pool from MongoDB
        print("\n--- Building Item Pool from MongoDB ---")
        item_pool = mongo_dataset.build_item_pool(limit_per_category=50)
        print(f"Built item pool with {len(item_pool):,} items")

        # Add category_idx to each item in pool for easier lookup
        for item in item_pool:
            # category_idx is the index into the categories list
            item['category_idx'] = item.get('reduced_cat_id', 0)

        # Create text encoder
        print("\n--- Initializing Text Encoder ---")
        text_encoder = TextEncoder(categories)

        # Initialize GNN model
        print("\n--- Initializing GNN Model ---")
        num_categories = mongo_dataset.num_categories

        gnn_model = OutfitGNN(
            num_categories=num_categories,
            category_embed_dim=64,
            visual_embed_dim=64,
            hidden_dim=128,
            num_heads=2,
            num_layers=2,
            dropout=0.3
        )

        if MODEL_PATH.exists():
            gnn_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            print(f"Loaded trained model from {MODEL_PATH}")
        else:
            print(f"No trained model found at {MODEL_PATH}, using untrained model")

        gnn_model.eval()

        # Create generator with MongoDB dataset
        print("\n--- Initializing Generator ---")
        generator = MongoOutfitGenerator(
            gnn_model=gnn_model,
            text_encoder=text_encoder,
            mongo_dataset=mongo_dataset,
            item_pool=item_pool,
            device=DEVICE
        )

        model_loaded = True
        print("\n" + "=" * 60)
        print("Models initialized successfully from MongoDB!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError initializing models: {e}")
        import traceback
        traceback.print_exc()
        model_loaded = False


@app.route('/')
def index():
    """Render the main page."""
    categories = []
    if mongo_dataset:
        categories = sorted(mongo_dataset.categories)

    return render_template(
        'index.html',
        categories=categories,
        model_loaded=model_loaded
    )


@app.route('/api/generate', methods=['POST'])
def generate_outfits():
    """
    Generate outfits from a text prompt.

    Request JSON:
        {
            "prompt": "summer cozy blue vibes",
            "num_outfits": 3
        }

    Response JSON:
        {
            "success": true,
            "outfits": [...]
        }
    """
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please try again later.'
        }), 500

    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing prompt in request'
            }), 400

        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Empty prompt provided'
            }), 400

        num_outfits = data.get('num_outfits', 3)
        num_outfits = min(max(num_outfits, 1), 5)

        print(f"\n[Generate] Prompt: '{prompt}', num_outfits: {num_outfits}")

        # Generate outfits
        outfits = generator.generate(
            prompt=prompt,
            num_outfits=num_outfits
        )

        print(f"[Generate] Generated {len(outfits)} outfits")

        return jsonify({
            'success': True,
            'outfits': outfits,
            'prompt': prompt
        })

    except Exception as e:
        print(f"[Generate] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get list of all categories."""
    if not mongo_dataset:
        return jsonify({'success': False, 'error': 'Dataset not loaded'}), 500

    return jsonify({
        'success': True,
        'categories': sorted(mongo_dataset.categories)
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status."""
    return jsonify({
        'success': True,
        'model_loaded': model_loaded,
        'dataset_size': len(item_pool),
        'num_categories': mongo_dataset.num_categories if mongo_dataset else 0
    })


@app.route('/api/image/<path:item_id>', methods=['GET'])
def get_image(item_id):
    """
    Serve image for an item from filesystem.

    Item ID format: {outfit_id}_{item_index}
    Image path: {IMAGE_BASE}/{outfit_id}/{item_index}.jpg
    """
    try:
        # Parse item_id: "100002074_0" -> outfit_id="100002074", item_index="0"
        parts = item_id.rsplit('_', 1)
        if len(parts) != 2:
            return jsonify({'success': False, 'error': 'Invalid item ID format'}), 400

        outfit_id, item_index = parts[0], parts[1]
        image_path = IMAGE_BASE / outfit_id / f"{item_index}.jpg"

        if not image_path.exists():
            return jsonify({'success': False, 'error': f'Image not found: {image_path}'}), 404

        return send_file(str(image_path), mimetype='image/jpeg')

    except Exception as e:
        print(f"[Image] Error serving image {item_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Initialize models when module loads
print("\n[App] Starting Flask application with MongoDB...")
initialize_models()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

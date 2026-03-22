"""
Flask Web Application for Text-to-Outfit Generation
"""
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from flask import Flask, render_template, request, jsonify

from src.data.dataset import OutfitDataset
from src.models.gnn import OutfitGNN
from src.models.text_encoder import TextEncoder
from src.models.generator import OutfitGenerator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_FEATURE_BASE = Path("C:/Users/mohi_shr/source/my-repos/Fashion-Recommendation-system-using-Graph-Neural-Networks-GNN-1/data/polyvore_image_vectors/images")
IMAGE_BASE = Path("C:/Users/mohi_shr/source/my-repos/Fashion-Recommendation-system-using-Graph-Neural-Networks-GNN-1/data/polyvore-images")
MODEL_PATH = BASE_DIR / "models" / "best_gnn.pt"

# Global variables for models
dataset = None
text_encoder = None
generator = None
model_loaded = False


def initialize_models():
    """Initialize the ML models on startup."""
    global dataset, text_encoder, generator, model_loaded

    print("=" * 60)
    print("Initializing Outfit Generation Models...")
    print("=" * 60)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    try:
        # Load dataset
        print("\n--- Loading Dataset ---")
        dataset = OutfitDataset(
            data_dir=str(DATA_DIR),
            image_feature_base=str(IMAGE_FEATURE_BASE)
        )

        # Get category names list
        categories = [dataset.category_names.get(i, f"cat_{i}") for i in range(dataset.num_categories)]
        print(f"Loaded {dataset.num_train} training outfits, {dataset.num_test} test outfits")
        print(f"Categories: {dataset.num_categories}")

        # Create text encoder
        print("\n--- Initializing Text Encoder ---")
        text_encoder = TextEncoder(categories)

        # Load GNN model
        print("\n--- Loading GNN Model ---")
        gnn_model = OutfitGNN(
            num_categories=dataset.num_categories,
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

        # Create generator
        print("\n--- Initializing Generator ---")
        generator = OutfitGenerator(
            gnn_model=gnn_model,
            text_encoder=text_encoder,
            dataset=dataset,
            device=DEVICE
        )

        model_loaded = True
        print("\n" + "=" * 60)
        print("Models initialized successfully!")
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
    if dataset:
        categories = [dataset.category_names.get(i, f"cat_{i}") for i in range(dataset.num_categories)]

    return render_template(
        'index.html',
        categories=sorted(categories),
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
            "outfits": [
                {
                    "items": [
                        {"category": "Tops", "item_id": "..."},
                        {"category": "Shorts", "item_id": "..."}
                    ],
                    "compatibility_score": 0.94,
                    "prompt_match_score": 0.89
                },
                ...
            ]
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
        num_outfits = min(max(num_outfits, 1), 5)  # Clamp between 1-5

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
    if not dataset:
        return jsonify({'success': False, 'error': 'Dataset not loaded'}), 500

    categories = [dataset.category_names.get(i, f"cat_{i}") for i in range(dataset.num_categories)]
    return jsonify({
        'success': True,
        'categories': sorted(categories)
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status."""
    return jsonify({
        'success': True,
        'model_loaded': model_loaded,
        'dataset_size': dataset.num_train if dataset else 0,
        'num_categories': dataset.num_categories if dataset else 0
    })


@app.route('/api/image/<path:item_id>', methods=['GET'])
def get_image(item_id):
    """
    Serve image for an item.

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
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        from flask import send_file
        return send_file(str(image_path), mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Initialize models when module loads
print("\n[App] Starting Flask application...")
initialize_models()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

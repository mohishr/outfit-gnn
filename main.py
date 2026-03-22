"""
End-to-end outfit generation demo.
Prototype using 5% of data.
"""
import json
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import OutfitDataset
from src.models.gnn import OutfitGNN
from src.models.text_encoder import TextEncoder
from src.models.generator import OutfitGenerator


def load_trained_model(dataset, model_path='models/best_gnn.pt'):
    """Load trained GNN model."""
    model = OutfitGNN(
        num_categories=dataset.num_categories,
        category_embed_dim=64,
        visual_embed_dim=64,
        hidden_dim=128,
        num_heads=2,
        num_layers=2,
        dropout=0.3
    )

    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded trained model from {model_path}")
    else:
        print(f"No trained model found at {model_path}, using untrained model")

    model.eval()
    return model


def main():
    print("=" * 60)
    print("Text-to-Outfit Generation Demo")
    print("=" * 60)

    # Settings
    DATA_DIR = Path("data")
    IMAGE_FEATURE_BASE = Path("C:/Users/mohi_shr/source/my-repos/Fashion-Recommendation-system-using-Graph-Neural-Networks-GNN-1/data/polyvore_image_vectors/images")

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {DEVICE}")

    # Load dataset
    print("\n--- Loading Dataset ---")
    dataset = OutfitDataset(
        data_dir=str(DATA_DIR),
        image_feature_base=str(IMAGE_FEATURE_BASE)
    )

    # Get category names list
    categories = [dataset.category_names.get(i, f"cat_{i}") for i in range(dataset.num_categories)]

    # Create text encoder
    print("\n--- Initializing Text Encoder ---")
    text_encoder = TextEncoder(categories)

    # Load GNN model
    print("\n--- Loading GNN Model ---")
    gnn_model = load_trained_model(dataset)

    # Create generator
    print("\n--- Initializing Generator ---")
    generator = OutfitGenerator(
        gnn_model=gnn_model,
        text_encoder=text_encoder,
        dataset=dataset,
        device=DEVICE
    )

    # Demo prompts
    prompts = [
        "summer cozy blue vibes",
        "casual fall outfit",
        "elegant evening wear",
        "sporty athletic look",
        "boho beach style"
    ]

    print("\n" + "=" * 60)
    print("GENERATING OUTFITS FROM PROMPTS")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n{'='*40}")
        print(f"Prompt: '{prompt}'")
        print("=" * 40)

        # Generate outfits
        outfits = generator.generate(
            prompt=prompt,
            num_outfits=2  # Generate 2 outfits per prompt
        )

        for i, outfit in enumerate(outfits):
            print(f"\n  Outfit {i+1}:")
            print(f"    Compatibility Score: {outfit['compatibility_score']:.3f}")
            print(f"    Prompt Match Score: {outfit['prompt_match_score']:.3f}")
            print(f"    Items:")
            for item in outfit['items']:
                print(f"      - {item['category']}")

        if not outfits:
            print("\n  (No valid outfits generated)")

    print("\n" + "=" * 60)
    print("API Usage Example:")
    print("=" * 60)
    print("""
from src.data.dataset import OutfitDataset
from src.models.gnn import OutfitGNN
from src.models.text_encoder import TextEncoder
from src.models.generator import OutfitGenerator

# Initialize
dataset = OutfitDataset()
gnn = OutfitGNN(num_categories=dataset.num_categories)
encoder = TextEncoder(categories=[...])
generator = OutfitGenerator(gnn, encoder, dataset)

# Generate outfit
outfits = generator.generate("summer cozy blue vibes")
print(outfits[0])
""")


if __name__ == "__main__":
    main()

"""
Training script for outfit compatibility GNN.
Uses 5% of data for quick prototype.
"""
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import OutfitDataset
from src.data.graph_builder import OutfitGraphBuilder, OutfitGraphDataset
from src.models.gnn import OutfitGNN


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def sample_5_percent(dataset, seed=42):
    """Sample 5% of outfits for training/testing."""
    set_seed(seed)

    n_train = int(len(dataset.train_outfits) * 0.05)
    n_test = int(len(dataset.test_outfits) * 0.05)

    train_sampled = random.sample(dataset.train_outfits, n_train)
    test_sampled = random.sample(dataset.test_outfits, n_test)

    return train_sampled, test_sampled


def create_negative_samples(outfits, dataset, num_neg_per_pos=1):
    """Create negative (incompatible) outfit samples."""
    negative_outfits = []

    # Get unique categories
    all_cats = list(range(dataset.num_categories))

    for outfit in outfits:
        for _ in range(num_neg_per_pos):
            # Randomly sample categories that don't co-occur
            num_items = len(outfit['items_category'])
            neg_cats = random.sample(all_cats, num_items)

            # Check that this combination is unlikely
            neg_outfit = {
                'set_id': f"neg_{outfit['set_id']}_{random.randint(0, 9999)}",
                'items_category': neg_cats,
                'items_index': [0] * num_items
            }
            negative_outfits.append(neg_outfit)

    return negative_outfits


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        predictions = model(batch)
        labels = batch.y

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, loader, device):
    """Evaluate model on given data."""
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Binary accuracy (threshold at 0.5)
    pred_binary = (predictions > 0.5).astype(int)
    labels_binary = (labels > 0.5).astype(int)
    accuracy = (pred_binary == labels_binary).mean()

    # MSE
    mse = ((predictions - labels) ** 2).mean()

    return {
        'accuracy': accuracy,
        'mse': mse,
        'predictions': predictions,
        'labels': labels
    }


def main():
    print("=" * 60)
    print("Outfit Compatibility Training (5% Prototype)")
    print("=" * 60)

    # Settings
    DATA_DIR = Path("data")
    IMAGE_FEATURE_BASE = Path("C:/Users/mohi_shr/source/my-repos/Fashion-Recommendation-system-using-Graph-Neural-Networks-GNN-1/data/polyvore_image_vectors/images")
    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)

    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # Load dataset
    print("\n--- Loading Dataset ---")
    dataset = OutfitDataset(
        data_dir=str(DATA_DIR),
        image_feature_base=str(IMAGE_FEATURE_BASE)
    )
    dataset.summary()

    # Sample 5%
    print("\n--- Sampling 5% of Data ---")
    train_outfits, test_outfits = sample_5_percent(dataset)
    print(f"Train outfits (5%): {len(train_outfits)}")
    print(f"Test outfits (5%): {len(test_outfits)}")

    # Create negative samples
    print("\n--- Creating Negative Samples ---")
    train_neg = create_negative_samples(train_outfits, dataset, num_neg_per_pos=1)
    test_neg = create_negative_samples(test_outfits, dataset, num_neg_per_pos=1)
    print(f"Train negative samples: {len(train_neg)}")
    print(f"Test negative samples: {len(test_neg)}")

    # Build graphs
    print("\n--- Building Graphs ---")
    builder = OutfitGraphBuilder(dataset)

    train_graphs = []
    for outfit in train_outfits:
        data = builder.create_outfit_graph(outfit, use_image_features=True)
        if data is not None:
            train_graphs.append(data)

    for outfit in train_neg:
        data = builder.create_outfit_graph(outfit, use_image_features=True)
        if data is not None:
            train_graphs.append(data)

    test_graphs = []
    for outfit in test_outfits:
        data = builder.create_outfit_graph(outfit, use_image_features=True)
        if data is not None:
            test_graphs.append(data)

    for outfit in test_neg:
        data = builder.create_outfit_graph(outfit, use_image_features=True)
        if data is not None:
            test_graphs.append(data)

    print(f"Train graphs: {len(train_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")

    # Create loaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    print("\n--- Creating Model ---")
    model = OutfitGNN(
        num_categories=dataset.num_categories,
        category_embed_dim=64,  # Smaller for quick prototype
        visual_embed_dim=64,
        hidden_dim=128,
        num_heads=2,
        num_layers=2,
        dropout=0.3
    ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    print("\n--- Training ---")
    best_acc = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step()

        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            train_metrics = evaluate(model, train_loader, DEVICE)
            test_metrics = evaluate(model, test_loader, DEVICE)

            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Acc: {train_metrics['accuracy']:.4f}, MSE: {train_metrics['mse']:.4f}")
            print(f"  Test Acc: {test_metrics['accuracy']:.4f}, MSE: {test_metrics['mse']:.4f}")

            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                # Save best model
                torch.save(model.state_dict(), MODEL_DIR / 'best_gnn.pt')
                print(f"  -> Saved best model (acc: {best_acc:.4f})")

    # Save final model
    torch.save(model.state_dict(), MODEL_DIR / 'final_gnn.pt')
    print(f"\nFinal model saved to {MODEL_DIR / 'final_gnn.pt'}")

    # Final evaluation
    print("\n--- Final Evaluation ---")
    final_metrics = evaluate(model, test_loader, DEVICE)
    print(f"Final Test Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final Test MSE: {final_metrics['mse']:.4f}")

    # Save metrics
    metrics = {
        'train_samples': len(train_graphs),
        'test_samples': len(test_graphs),
        'best_accuracy': float(best_acc),
        'final_accuracy': float(final_metrics['accuracy']),
        'final_mse': float(final_metrics['mse']),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }

    with open(MODEL_DIR / 'training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

# Outfit Dataset Documentation

## Overview

This dataset contains fashion outfit combinations represented as sets of item categories. It is designed for training Graph Neural Networks (GNNs) to learn item compatibility and generate/recommend outfits.

## Files

| File | Description |
|------|-------------|
| `category_id.txt` | Maps category IDs to human-readable category names (120 categories) |
| `cid2rcid_100.json` | Maps original category IDs to reduced consecutive IDs (0-119) for embedding layers |
| `category_summarize_100.json` | Category metadata including item instances (`setid_position`) |
| `train_no_dup_new_100.json` | Training set: 16,983 unique outfits |
| `test_no_dup_new_100.json` | Test set: 2,697 unique outfits |
| `polyvore_image_vectors/images/*.json` | Pre-extracted CNN image features (2048-dim vectors per item) |

### Image Features

Pre-extracted image features from a CNN model for visual-based outfit modeling.

**Location:** `polyvore_image_vectors/images/` (external path)

**File Pattern:** `{outfit_id}_{item_position}.json`

**Feature Vector:** 2048-dimensional CNN features per item

**Total Files:** 444,371 image feature vectors

**Example:**
```json
// File: 100002074_0.json
[1.053, 3.443, 1.812, 2.239, ...]  // 2048 values
```

**Usage:**
- Node features in GNN: Use image embeddings instead of (or alongside) category embeddings
- Visual compatibility: Learn item similarity in visual feature space
- Generation ranking: Score generated items by visual similarity to training distribution

**Loading Image Features:**
```python
import json

IMAGE_FEATURE_PATH = "Fashion-Recommendation-system-using-Graph-Neural-Networks-GNN-1/data/polyvore_image_vectors/images/"

def load_image_feature(outfit_id: str, item_index: int):
    """Load 2048-dim CNN feature for an item."""
    file_path = f"{IMAGE_FEATURE_PATH}{outfit_id}_{item_index}.json"
    with open(file_path, 'r') as f:
        return json.load(f)  # Returns list of 2048 floats

# Example: Load features for first outfit
outfit = train_outfits[0]
for i, cat_id in enumerate(outfit['items_category']):
    item_idx = outfit['items_index'][i]
    features = load_image_feature(outfit['set_id'], item_idx)
    print(f"Category {cat_id}, Item {item_idx}: {len(features)}-dim vector")
```

## Data Format

### Outfit JSON Structure
```json
{
  "set_id": "119704139",           // Unique outfit identifier
  "items_category": [17, 236, 9],  // Category IDs of items in outfit
  "items_index": [1, 2, 3]         // Position of each item (1-indexed)
}
```

### Category Mapping
```
2  Clothing
3  Dresses
4  Day Dresses
5  Cocktail Dresses
...
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Training outfits | 16,983 |
| Test outfits | 2,697 |
| Unique categories | 120 |
| Items per outfit (train) | 3-8 (avg: 6.17) |
| Items per outfit (test) | 3-8 (avg: 5.33) |

## Graph Representation for GNN

### Why Graph Structure?

Each outfit naturally forms a **clique** (fully connected subgraph) where:
- **Nodes** = Items in the outfit
- **Node Features** = 2048-dim CNN image embeddings (when available)
- **Edges** = Compatibility/co-occurrence relationships between items

```
Example Outfit:
  Items: [Tops, Necklaces, Shoulder Bags, Sandals]
  Node Features: [2048-dim image vectors from CNN]

  Graph:
    Tops --- Necklaces
     |    \    |
     |     \   |
    Bags --- Sandals
```

### Dual Feature Representation

Each item can be represented by:
1. **Category ID** → Learned category embedding (120 categories)
2. **Image Feature** → 2048-dim CNN vector (visual appearance)

**Recommended:** Concatenate both for richer item representation

### Top Category Co-occurrences

| Category Pair | Frequency |
|--------------|-----------|
| Necklaces + Earrings | 1,151 |
| Earrings + Bracelets & Bangles | 1,110 |
| Shoulder Bags + Sunglasses | 1,047 |
| Shoulder Bags + Earrings | 1,041 |
| Pumps + Earrings | 1,012 |

## GNN Training Objectives

### 1. Compatibility Learning (Link Prediction)
Learn if two item categories are compatible to wear together.

**Task**: Given two item categories, predict compatibility score [0-1]

### 2. Outfit Completion (Node Classification/Link Prediction)
Given partial outfit, predict missing items.

**Task**: Given outfit with K items, predict the (8-K) missing items

### 3. Outfit Generation
Generate complete valid outfits from scratch or from a prompt.

**Task**: Generate a set of compatible items forming a complete outfit

## Model Architecture Recommendations

### Input Features
- **Node Features (Category)**: Learned category embeddings (120 categories → 128-dim)
- **Node Features (Visual)**: 2048-dim CNN image features (from pre-extracted vectors)
- **Combined Node Features**: Concatenation of category + visual embeddings (2176-dim)
- **Edge Features**: Co-occurrence frequency, category similarity

### Architecture Options

1. **GCN/GAT** with dual-node-features: category embeddings + image CNN features
2. **Graph Auto-Encoder** for learning item embeddings in visual space
3. **Vision-Language GNN** combining image features, category, and text prompts

### Dual-Stream GNN Architecture

```
Category ID → Category Embedding (128-dim)
                          │
                          ▼
Image Feature (2048-dim) → CNN/MLP Projector (128-dim)
                          │
                          ▼
                  Concatenate → 256-dim fused features
                          │
                          ▼
                    Graph Attention Layers
                          │
                          ▼
                Compatibility Score Prediction
```

### Loss Functions
- Binary cross-entropy (compatible/incompatible pairs)
- Contrastive loss (compatible vs incompatible)
- Reconstruction loss (for auto-encoder approaches)

## Usage Example

```python
import json

# Load outfit data
with open('train_no_dup_new_100.json', 'r') as f:
    train_outfits = json.load(f)

# Load category names
cat_names = {}
with open('category_id.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            cat_names[int(parts[0])] = parts[1]

# Example: Build adjacency matrix for one outfit
outfit = train_outfits[0]
categories = outfit['items_category']
n = len(categories)

# Fully connected graph for this outfit
adj_matrix = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
```

## Integration with Text Prompts

For text-conditioned generation (e.g., "summer cozy blue vibes"):

1. **Encode text prompt** using CLIP to get text embedding [512-dim]
2. **Encode item images** using CLIP or use pre-extracted 2048-dim CNN features
3. **Condition GNN** on text embedding to generate compatible items
4. **Filter/Rank** generated items by:
   - Compatibility score (from GNN)
   - CLIP similarity to text prompt

## Citation

This dataset structure is inspired by fashion compatibility modeling research using GNNs.

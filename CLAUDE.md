# Project Goal: Text-to-Outfit Generation with GNN

## Vision

Build a system that generates complete, compatible outfit sets from natural language prompts.

**Input Example:**
```
"Suggest me outfit for summer cozy blue vibes"
```

**Output Example:**
```
{
  "items": [
    {"category": "Tops", "item_id": "..."},
    {"category": "Shorts", "item_id": "..."},
    {"category": "Sandals", "item_id": "..."},
    {"category": "Sunglasses", "item_id": "..."},
    {"category": "Necklaces", "item_id": "..."}
  ],
  "compatibility_score": 0.94,
  "prompt_match_score": 0.89
}
```

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
Text Prompt ───────▶│  Text Encoder (CLIP)                   │
                    │  → Text Embedding: [512]               │
                    └────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │  GNN Compatibility Model                │
                    │                                         │
Category ──────────▶│  ┌─────────────┐    ┌─────┐            │
Embedding (128)    │  │   Fusion    │───▶│ GAT │ ────▶      │
                    │  │  (256-dim) │    │     │              │
Image ─────────────▶│  └─────────────┘    └─────┘              │
Feature (2048)     │  CNN→128 projection                       │
                    │                                         │
                    │  Learn: item ↔ item compatibility       │
                    └────────────────┬──────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │  Generation Module                      │
                    │  - Given seed items + prompt            │
                    │  - Generate/add compatible items         │
                    │  - Score and rank complete outfits      │
                    └─────────────────────────────────────────┘
```

### Dual-Feature Input

Each item has TWO representations:
1. **Category Embedding** (128-dim): Learned from category ID
2. **Visual Embedding** (128-dim): Projected 2048-dim CNN features

Concatenated → 256-dim fused node feature for GNN

## Project Phases

### Phase 1: Data Processing & Analysis
- [x] Explore dataset structure
- [x] Build category embeddings from co-occurrence
- [x] Create train/test splits
- [x] Identify image feature location (444K pre-extracted 2048-dim vectors)
- [ ] Build graph adjacency matrices from outfits
- [ ] Create mapping: outfit items ↔ image feature files

### Phase 2: GNN Model Development
- [ ] Implement dual-stream node features (category + visual)
- [ ] Build CNN projector (2048 → 128 dim)
- [ ] Build Graph Attention Network (GAT) for compatibility scoring
- [ ] Train on outfit compatibility (link prediction)
- [ ] Validate on held-out test outfits

### Phase 3: Text Conditioning
- [ ] Integrate text encoder (CLIP)
- [ ] Learn joint text-item embeddings
- [ ] Condition generation on text prompts

### Phase 4: Generation & Evaluation
- [ ] Implement outfit generation from seed items + prompt
- [ ] Build complete outfit reconstruction
- [ ] Evaluate: compatibility scores, prompt matching
- [ ] User interface for prompt input

## Key Technical Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| GNN Architecture | GAT (Graph Attention Network) | Learns adaptive importance between item pairs |
| Node Features | Category (128-dim) + Visual (128-dim) | Dual representation for richer semantics |
| Visual Projector | MLP (2048 → 128) | Learn visual compatibility patterns |
| Compatibility Score | Edge prediction probability | Direct optimization for compatibility |
| Text Encoding | CLIP | Strong fashion/visual-text alignment |
| Generation Strategy | Iterative item addition | Allows controlled outfit construction |

## Dataset for GNN Training

### Outfit Data
- **16,983 training outfits** (cliques of compatible items)
- **120 unique categories**
- **Co-occurrence patterns** = ground truth compatibility

### Image Features (Pre-extracted)
- **Location:** `Fashion-Recommendation-system-using-Graph-Neural-Networks-GNN-1/data/polyvore_image_vectors/images/`
- **Pattern:** `{outfit_id}_{item_position}.json`
- **Vector Size:** 2048-dimensional CNN features
- **Total Files:** 444,371 image feature vectors

### Data Linking
```
outfit["set_id"] + "_" + item_index → image_feature_file
Example: "100002074_0.json" → CNN vector [2048-dim]
```

**GNN learns:**
1. Node representations combining category AND visual features
2. Edge weights representing compatibility strength
3. Generalization: predict compatibility for unseen category pairs

## Success Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Compatibility Score | Avg compatibility within generated outfit | > 0.85 |
| Prompt Match | CLIP similarity between prompt and generated items | > 0.80 |
| Diversity | variety in generated outfits for same prompt | > 0.70 |
| Validity | % of generated outfits matching training distribution | > 0.90 |

## File Structure (To Be Created)

```
final-outfit/
├── CLAUDE.md              # This file
├── data/
│   └── README.md          # Dataset documentation
├── src/
│   ├── data/              # Data loading & processing
│   │   ├── dataset.py
│   │   ├── graph_builder.py
│   │   └── image_features.py   # NEW: Image feature loader
│   ├── models/            # Model definitions
│   │   ├── gnn.py         # GAT architecture
│   │   ├── text_encoder.py
│   │   ├── visual_projector.py # NEW: CNN→embedding MLP
│   │   └── generator.py
│   └── training/
│       ├── train_compat.py
│       └── train_gen.py
├── notebooks/
│   └── exploration.ipynb
└── README.md
```

## Current Status

**Done:**
- Dataset exploration and documentation
- Understood outfit-as-graph representation
- Identified GNN training objectives
- Located 444K pre-extracted image features (2048-dim CNN vectors)

**Next Steps:**
1. Create mapping between outfit items and image feature files
2. Build dual-stream node features (category + visual)
3. Implement GNN for compatibility scoring
4. Train and validate on test set

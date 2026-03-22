# NGNN Outfit Intelligence - Web App

A Flask web application for generating complete, compatible outfit sets from natural language prompts.

## Setup

1. **Install dependencies:**
```bash
cd final-outfit
pip install -r requirements.txt
```

2. **Run the app:**
```bash
python app/app.py
# or
bash app/run.sh
```

3. **Open in browser:**
```
http://localhost:5000
```

## Features

- **Text-to-Outfit Generation**: Enter prompts like "summer cozy blue vibes" to generate compatible outfit combinations
- **Compatibility Scoring**: Each generated outfit is scored for internal compatibility
- **Prompt Matching**: Outfits are ranked by how well they match your text description
- **Visual Items**: Each outfit card displays the category and item identifiers

## Example Prompts

- "summer cozy blue vibes"
- "casual fall outfit"
- "elegant evening wear"
- "sporty athletic look"
- "boho beach style"
- "minimalist modern streetwear"

## API

### Generate Outfits
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "summer cozy blue vibes", "num_outfits": 3}'
```

### Check Status
```bash
curl http://localhost:5000/api/status
```

### Get Categories
```bash
curl http://localhost:5000/api/categories
```

## Project Structure

```
final-outfit/
├── app/
│   ├── app.py           # Flask application
│   ├── index.html       # Standalone HTML (reference)
│   └── templates/
│       └── index.html   # Jinja2 template for web UI
├── src/
│   ├── data/
│   │   └── dataset.py   # OutfitDataset
│   └── models/
│       ├── gnn.py       # OutfitGNN (GAT)
│       ├── text_encoder.py
│       └── generator.py # OutfitGenerator
└── models/
    └── best_gnn.pt      # Trained GNN weights
```

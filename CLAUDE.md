# Outfit GNN — Item-Level Text-to-Outfit Recommender

Text prompt → top-k Polyvore outfits, ranked by an item-level GNN
compatibility score combined with a prompt-match score computed in the
*same* embedding space as the item names. Optionally enriched with
ResNet50 visual features per item.

## Layout

```
data/                           polyvore json (new schema with item names)
  train_no_dup.json             outfits w/ items[].{name, categoryid, image, ...}
  valid_no_dup.json
  test_no_dup.json
  fill_in_blank_test.json       FITB benchmark (4 candidates, 1 correct)
  fashion_compatibility_prediction.txt
  category_id.txt
src/
  config.py                     paths + IMAGE_DIR + filter & model dims
  filter.py                     deny-list of non-fashion categories
  dataset.py                    new-schema loader, deny-list filter
  items.py                      item index, vocab, name tokens (cached)
  model.py                      ItemGNN: name + cat + visual -> T-hop msg pass
  visual.py                     OPTIONAL: ResNet50 feature extractor (torchvision)
  train.py                      contrastive training, AUC + FITB eval
  recommend.py                  pipeline: prompt -> outfits / fill-in-blank
app/
  app.py                        Flask: /, /recommend, /fill_blank, /image/<set>/<i>
  templates/index.html          editorial UI (cream + gold)
weights/
  items.pt                      cached item index + vocab + tokens
  visuals.pt                    OPTIONAL: [num_items, 2048] visual feats
  itemgnn.pt                    model checkpoint (auc, fitb, use_visual stored)
```

## Why item-level (not category-level)

Previous version embedded *categories* — every item with `categoryid=46`
(Sandals) had the same vector. The GNN could only learn "Sandals fit with
Skinny Jeans". This couldn't choose *which* sandals.

This version embeds *items* using:
1. **Name embedding** — `EmbeddingBag` over the item-name tokens
   ("citizens humanity high rise rocket hem jean" → mean of 7 word vectors).
2. **Category embedding** — keeps the coarse cluster signal.
3. **Visual embedding** — ResNet50-pooled feature, projected via a
   `Linear(2048, dim)`. Optional, on if `weights/visuals.pt` exists.

Sum of the three goes into the GNN. Each *item* is a node now.

## Prompt matching in the trained space

The same word-embedding table that builds item-name features also embeds
the user's prompt at recommendation time. Tokenise the prompt → mean of
its word vectors → cosine similarity with every item's name vector
(both in the same learnt space). This means "blue silk dress" actually
finds items literally named "blue silk dress" — a huge upgrade over
matching prompt against category names.

## NGNN architecture

```
word_emb     Embedding(vocab=2589, 128)
cat_emb      Embedding(num_cats=106, 128)
visual_proj  Linear(2048, 128)              # optional, gated by use_visual

item_feat    = mean(word_emb(name_tokens))
              + cat_emb(cat_dense)
              + visual_proj(visual_feat)    # if use_visual

T = 2 hops of leave-one-out msg passing within an outfit clique:
    m   = msg(h)
    agg = (sum(m) - m) / (n-1)
    h   = ReLU(upd([h || agg]))   then dropout(0.2)

score    MLP(2*128 -> 128 -> 1)            # pair compatibility logit
outfit_score(O) = mean(sigmoid(pair_logits(O)))
```

## Training (item-level contrastive)

Per real outfit O of size n:
- positive features = item_feat(O)
- hard negative   = item_feat(O with one slot replaced by a uniformly random
                    item from the corpus)
- loss = BCE(pair_logits(pos), 1) + BCE(pair_logits(neg), 0)

AdamW(lr=2e-3, wd=1e-4), cosine LR, grad-clip 1.0.

## Evaluation

| Metric | What it measures |
|---|---|
| Outfit AUC | `P(score(real outfit) > score(corrupted outfit))` |
| FITB@500   | Polyvore fill-in-the-blank: 4 candidates, pick ground truth |

Both are logged each epoch; the best-AUC checkpoint is saved with
metadata embedded.

## Pipeline

**1. Recommend (prompt → outfits)**
For every cached outfit:
```
prompt_score(O) = mean over items i of cos(prompt_vec, name_vec(i))
compat_score(O) = mean(sigmoid(pair_logits(item_feat(O))))   # precomputed
final = α · prompt_score + (1-α) · compat_score
```
Return top-k outfits with their items.

**2. Fill blank (partial outfit + target category → best item)**
For every candidate item in the target category:
```
score(c) = α · prompt_score(c) + (1-α) · compat_score(partial ∪ {c})
```
Return top-k items.

## Running

```bash
pip install -r requirements.txt              # torch + flask + numpy + pillow

# (optional) image features:
pip install torchvision
python -m src.visual                         # ~30 min CPU, 113k items

python -m src.dataset                        # filter stats
python -m src.items                          # build item index + vocab
python -m src.train --dummy                  # random weights for the demo
python -m src.train                          # real training, ~5-10 min/epoch CPU
python -m src.train --eval                   # AUC + FITB on test split

python -m src.recommend "summer cozy blue vibes"
python app/app.py                            # http://localhost:5000
```

## Data labelling — to answer the recurring question

There is **no explicit "compatible / incompatible" label file**. Compatibility
is derived from co-occurrence in real polyvore sets:

- **Positives**: items that appear in the same set are treated as compatible.
  Polyvore sets were curated by humans to be coherent outfits, so this is a
  strong (if noisy) signal.
- **Hard negatives** (training): a real outfit with one slot replaced by a
  random item from the corpus.

`fashion_compatibility_prediction.txt` provides labelled pos/neg outfits for
benchmarking (1 = compatible, 0 = not), and `fill_in_blank_test.json` gives
multiple-choice fill-in-blank questions — both used for evaluation.

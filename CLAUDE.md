# Outfit GNN — Text-to-Outfit Recommender (Production)

Text prompt → top-k Polyvore outfits, ranked by an NGNN compatibility score
combined with a CLIP prompt-match score. Filtered to a fashion-only category
set so beauty/tech/home noise can't pollute recommendations.

## Layout

```
data/                          polyvore json (train/test outfit cliques, category map)
src/
  config.py                    paths + IMAGE_DIR + filter thresholds + model dims
  filter.py                    deny-list of non-fashion categories
  dataset.py                   load + filter outfits, dense category remap (cached)
  model.py                     NGNN: emb -> T-hop msg pass + dropout -> pair MLP
  text_encoder.py              CLIP text encoder (fallback: keyword + vibe table)
  recommend.py                 3-stage pipeline (Recommender class + CLI)
  train.py                     contrastive training, AUC eval, best-ckpt saving
app/
  app.py                       Flask: GET / · POST /recommend · GET /image/<set>/<i>
  templates/index.html         editorial UI (cream + gold, serif + mono)
weights/
  ngnn.pt                      checkpoint (stores num_cats, dim, hops, AUC)
  cat_text_embeds.pt           cached CLIP category embeddings
```

## Filtering — why and what

The polyvore dataset includes Lipstick (2326 occurrences), Tech Accessories
(2454), Eyeshadow, Nail Polish, Home Decor, Floral Decor, Books, Toys,
Stationery, Drinkware, Food, Font etc. They co-occur with apparel inside
"sets" but contribute no outfit-compatibility signal — they only dilute it.

`filter.py` carries an explicit deny-list of ~43 non-fashion category names.
Combined with `MIN_CATEGORY_FREQ` from `config.py`, this keeps **78
fashion categories** out of the 380 in the raw category file (120 of which
have any data). 16,431 / 16,983 train outfits and 2,521 / 2,697 test outfits
survive (≥ 3 fashion items each).

After filtering, `dataset.py` rebuilds a contiguous remap so the GNN
embedding table has exactly `num_cats = 78` entries.

## Pipeline

**Stage 1 — Prompt → category preference vector (TextEncoder)**
CLIP text encoder embeds the prompt and every kept category name
("a fashion item: <name>"). Cosine similarity → min-max normalised score
per category. Falls back to a curated vibe synonym table (summer →
shorts/sandals/sunglasses, winter → coats/sweaters/boots, etc.) if CLIP
isn't available.

**Stage 2 — Score every candidate outfit**
```
prompt_match = mean(cat_score[c_i])               in [0, 1]
compat       = NGNN.outfit_score(rcats(O))         in [0, 1]   (precomputed)
final        = α · prompt_match + (1-α) · compat
```

**Stage 3 — Return top-k outfits**
Each item carries `set_id`, `index`, category name, and `image_url`
(`/image/<set_id>/<index>` — served from `$IMAGE_DIR/<set_id>/<index>.jpg`).

## NGNN

```
emb:        Embedding(78, 128)                     category vectors
msg_l:      [Linear(128, 128)] × hops              per-hop message projection
upd_l:      [Linear(256, 128)] × hops              fuse self || mean(neighbours)
dropout:    p=0.2 after each hop and inside scorer
score:      MLP(256 -> 128 -> 1)                   pair compatibility logit
```

`hops = 2`: each item sees 2-hop outfit context — neighbours that themselves
were refined by their own neighbours.

## Training

Per real outfit O (size n):
- `pos = O`                         — all internal pairs are positive
- `easy_neg = n random distinct categories` — fully random outfit
- `hard_neg = O with one slot replaced by a random category` — almost-real

Loss = BCE on positive pairs + BCE on both negative sets.
Optimiser: AdamW (`lr=2e-3`, `wd=1e-4`), cosine LR schedule, grad-clip 1.0.

Hard negatives matter: a model that only sees fully-random negatives learns
to detect "is this set incoherent at all" (easy). Hard negatives push it
to learn "is this *the right* item for this slot", which is what
recommendation needs.

## Evaluation — AUC on the held-out test split

Per test outfit:
```
pos_score = NGNN.outfit_score(real_cats)
neg_score = NGNN.outfit_score(real_cats with one slot replaced)
```
Then `AUC = P(pos_score > neg_score)` over all (pos, neg) pairs.

The training loop logs AUC every epoch and keeps the best-AUC state. The
checkpoint records its AUC; the UI surfaces it as a status pill.

## Why outfit retrieval, not item generation

The GNN scores categories, not items — every item of category C has the same
embedding. Per-item ranking inside a category needs item-level features
(visual or textual) or one-node-per-item training. The honest framing today
is: "rank existing outfits by category-level compatibility weighted against
prompt match, return their items".

Upgrade path: replace `cid2rcid` with a dense item-id remap (`set_id_idx ->
int`), keep the NGNN otherwise unchanged. Each item becomes its own node;
co-occurrence trains item-specific embeddings. ~30 line change in
`dataset.py` and `model.py`.

## Run

```bash
pip install -r requirements.txt          # torch, flask, numpy, transformer

python app/app.py                        # http://localhost:5000
```

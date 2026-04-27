"""Train ItemGNN on item-level co-occurrence with hard negatives.

LOSS — outfit-level contrastive
    For each real outfit O of size n:
        positive features = item_feat(O)
        negative features = item_feat(O with one slot replaced by a random
                                       item drawn uniformly from the corpus)
    pos_logits = pair_logits(positive features)   ← all pairs labelled 1
    neg_logits = pair_logits(negative features)   ← all pairs labelled 0
    loss = BCE(pos, 1) + BCE(neg, 0)

    Hard negatives only — with item-level features, fully-random outfits
    become trivially separable; the slot-swap negative is what teaches the
    GNN to recognise *the right item for a given outfit position*.

EVAL — outfit AUC on the held-out test split
    pos_score = outfit_score(real items)
    neg_score = outfit_score(corrupted items, one slot swapped)
    AUC = P(pos_score > neg_score) over all (pos, neg) pairs.

    Plus FITB accuracy on `fill_in_blank_test.json` if available.

CLI:
    python -m src.train             full training (default 5 epochs)
    python -m src.train --dummy     random weights for the demo
    python -m src.train --eval      evaluate current ckpt on test
"""
import json
import random
import sys
import time

import torch
import torch.nn.functional as F

from .config import (
    DATA, NGNN_CKPT, EMBED_DIM, GNN_HOPS, DROPOUT, VISUAL_DIM, VISUAL_CACHE, FITB_FILE
)
from .dataset import load_clean
from .items import ItemIndex
from .model import ItemGNN


# ───────────── helpers ─────────────
def _outfit_item_ints(outfit, items: ItemIndex):
    """Return a list of item-integer ids for a parsed outfit dict."""
    out = []
    for it in outfit["items"]:
        iid = f"{outfit['set_id']}_{it['index']}"
        if iid in items.id_to_int:
            out.append(items.id_to_int[iid])
    return out


def _features_for(items: ItemIndex, ints: list[int], visuals: torch.Tensor | None,
                  model: ItemGNN) -> torch.Tensor:
    toks = [items.name_tokens[i] for i in ints]
    cats = torch.tensor([items.cat_dense[i] for i in ints], dtype=torch.long)
    vis = visuals[ints] if (visuals is not None and model.use_visual) else None
    return model.item_features(toks, cats, vis)


def _swap_one(ints: list[int], num_items: int) -> list[int]:
    out = list(ints)
    i = random.randrange(len(out))
    j = random.randrange(num_items)
    while j == out[i]:
        j = random.randrange(num_items)
    out[i] = j
    return out


@torch.no_grad()
def _outfit_score(model, h0):
    if h0.size(0) < 2:
        return 0.5
    return float(torch.sigmoid(model.pair_logits(h0)).mean())


@torch.no_grad()
def auc(pos_scores, neg_scores) -> float:
    p = torch.tensor(pos_scores).unsqueeze(1)
    n = torch.tensor(neg_scores).unsqueeze(0)
    return float(((p > n).float() + 0.5 * (p == n).float()).mean())


@torch.no_grad()
def evaluate_auc(model, items, outfits, visuals):
    model.eval()
    pos, neg = [], []
    N = len(items)
    for o in outfits:
        ints = _outfit_item_ints(o, items)
        if len(ints) < 2:
            continue
        pos.append(_outfit_score(model, _features_for(items, ints, visuals, model)))
        neg.append(_outfit_score(model, _features_for(items, _swap_one(ints, N), visuals, model)))
    return auc(pos, neg)


@torch.no_grad()
def evaluate_fitb(model, items, visuals, max_n=500):
    """Fill-in-the-blank: 4 candidates, pick the one with highest outfit_score."""
    f = DATA / FITB_FILE
    if not f.exists():
        return None
    questions = json.loads(f.read_text())[:max_n]
    correct = 0
    total = 0
    model.eval()
    for q in questions:
        partial_iids = q["question"]
        candidates = q["answers"]
        partial_ints = [items.id_to_int[i] for i in partial_iids if i in items.id_to_int]
        cand_ints = [items.id_to_int.get(i) for i in candidates]
        if any(c is None for c in cand_ints) or len(partial_ints) < 1:
            continue
        scores = []
        for c in cand_ints:
            full = partial_ints + [c]
            scores.append(_outfit_score(model, _features_for(items, full, visuals, model)))
        if scores.index(max(scores)) == 0:   # answer 0 is the ground-truth
            correct += 1
        total += 1
    return correct / max(1, total) if total else None


# ───────────── train ─────────────
def train(epochs: int = 5, lr: float = 2e-3, wd: float = 1e-4,
          batch_outfits: int = 64, seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)

    d = load_clean()
    items = ItemIndex.load_or_build()
    visuals = _load_visuals_if_any(len(items))
    use_visual = visuals is not None

    train_o, test_o = d["train"], d["test"]
    print(f"[data] cats={d['num_cats']} items={len(items):,} vocab={items.vocab_size:,} "
          f"train={len(train_o)} test={len(test_o)}  visuals={'yes' if use_visual else 'no'}")

    model = ItemGNN(
        num_cats=d["num_cats"], vocab_size=items.vocab_size,
        dim=EMBED_DIM, hops=GNN_HOPS, dropout=DROPOUT,
        visual_dim=VISUAL_DIM, use_visual=use_visual,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    N = len(items)
    best_auc, best_state = -1.0, None
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        random.shuffle(train_o)
        loss_sum, n = 0.0, 0
        for i in range(0, len(train_o), batch_outfits):
            opt.zero_grad()
            batch_loss = 0.0
            for o in train_o[i : i + batch_outfits]:
                ints = _outfit_item_ints(o, items)
                if len(ints) < 2:
                    continue
                pos_h = _features_for(items, ints, visuals, model)
                neg_h = _features_for(items, _swap_one(ints, N), visuals, model)
                pl = model.pair_logits(pos_h)
                nl = model.pair_logits(neg_h)
                loss = (
                    F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl))
                    + F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl))
                )
                batch_loss = batch_loss + loss
                n += pl.numel() + nl.numel()
            if isinstance(batch_loss, float):
                continue
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(batch_loss)
        sched.step()

        a = evaluate_auc(model, items, test_o, visuals)
        msg = f"epoch {ep+1}/{epochs}  loss/pair={loss_sum/max(1,n):.5f}  test_AUC={a:.4f}"
        if a > best_auc:
            best_auc = a
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            msg += "  *best*"
        print(msg)

    model.load_state_dict(best_state)
    fitb = evaluate_fitb(model, items, visuals)
    print(f"[final] best_AUC={best_auc:.4f}  FITB@500={fitb}")
    _save(model, d["num_cats"], items.vocab_size, best_auc, fitb, dummy=False)
    print(f"[done] elapsed {time.time()-t0:.1f}s")


def _load_visuals_if_any(n_items):
    if not VISUAL_CACHE.exists():
        return None
    v = torch.load(VISUAL_CACHE, map_location="cpu", weights_only=False)
    if v.shape[0] != n_items:
        print(f"[visual] cache size mismatch ({v.shape[0]} vs {n_items}); ignoring")
        return None
    return v


def _save(model, num_cats, vocab_size, auc_v, fitb, dummy):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_cats": num_cats, "vocab_size": vocab_size,
            "dim": EMBED_DIM, "hops": GNN_HOPS, "dropout": DROPOUT,
            "use_visual": bool(model.use_visual), "visual_dim": VISUAL_DIM,
            "auc": auc_v, "fitb": fitb, "dummy": dummy,
        },
        NGNN_CKPT,
    )
    print(f"saved -> {NGNN_CKPT}  (auc={auc_v}, fitb={fitb}, dummy={dummy})")


def dummy(seed: int = 0):
    torch.manual_seed(seed)
    d = load_clean()
    items = ItemIndex.load_or_build()
    use_visual = VISUAL_CACHE.exists()
    model = ItemGNN(
        num_cats=d["num_cats"], vocab_size=items.vocab_size,
        dim=EMBED_DIM, hops=GNN_HOPS, dropout=DROPOUT,
        visual_dim=VISUAL_DIM, use_visual=use_visual,
    )
    _save(model, d["num_cats"], items.vocab_size, None, None, dummy=True)


def eval_only():
    d = load_clean()
    items = ItemIndex.load_or_build()
    visuals = _load_visuals_if_any(len(items))
    ckpt = torch.load(NGNN_CKPT, map_location="cpu", weights_only=False)
    model = ItemGNN(
        num_cats=ckpt["num_cats"], vocab_size=ckpt["vocab_size"],
        dim=ckpt.get("dim", EMBED_DIM), hops=ckpt.get("hops", GNN_HOPS),
        dropout=ckpt.get("dropout", DROPOUT), visual_dim=ckpt.get("visual_dim", VISUAL_DIM),
        use_visual=ckpt.get("use_visual", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    a = evaluate_auc(model, items, d["test"], visuals)
    fitb = evaluate_fitb(model, items, visuals)
    print(f"AUC={a:.4f}  FITB@500={fitb}  (saved auc={ckpt.get('auc')}, fitb={ckpt.get('fitb')})")


if __name__ == "__main__":
    if "--dummy" in sys.argv:
        dummy()
    elif "--eval" in sys.argv:
        eval_only()
    else:
        train()

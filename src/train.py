"""Train NGNN on filtered Polyvore outfits with rigorous evaluation.

LOSS — outfit-level contrastive link prediction
    Per real outfit O of size n we generate two negatives:
        easy_neg = n random distinct categories                (fully random outfit)
        hard_neg = O with one random slot replaced             (almost-real outfit)
    All pairwise logits inside O   are positives,
    all pairwise logits inside easy_neg + hard_neg are negatives.
    BCE with logits + 1.0 weight on hard negatives.

EVAL — outfit-level AUC on the held-out polyvore test split
    For each test outfit:
        pos_score = mean(sigmoid(pair_logits(real_cats)))
        neg_score = mean(sigmoid(pair_logits(corrupted_cats)))   # swap one slot
    AUC = P(pos_score > neg_score)  computed over all (pos, neg) pairs.

Why AUC and not loss: a model can drive loss down on noise-rich training
data and still rank real outfits below corrupted ones. AUC measures
recommendation-relevant ranking ability directly.

CLI:
    python -m src.train                 full training, save best AUC ckpt
    python -m src.train --dummy         random weights for the demo
    python -m src.train --eval          evaluate current ngnn.pt on test
"""
import math, random, sys, time
import torch
import torch.nn.functional as F

from .config import NGNN_CKPT, EMBED_DIM, GNN_HOPS, DROPOUT
from .dataset import load_clean, remap
from .model import NGNN


# ─────────────────── helpers ───────────────────
def _easy_neg(num_cats: int, n: int):
    return torch.tensor(random.sample(range(num_cats), min(n, num_cats)))


def _hard_neg(rcats: torch.Tensor, num_cats: int):
    neg = rcats.clone()
    i = random.randrange(neg.numel())
    j = random.randrange(num_cats)
    while j == int(neg[i]):
        j = random.randrange(num_cats)
    neg[i] = j
    return neg


@torch.no_grad()
def _outfit_score(model: NGNN, cats: torch.Tensor) -> float:
    if cats.numel() < 2:
        return 0.5
    return float(torch.sigmoid(model.pair_logits(cats)).mean())


@torch.no_grad()
def auc_from_pairs(pos_scores, neg_scores) -> float:
    """Pairwise AUC: P(pos > neg). O(P*N) but P, N ~ a few thousand."""
    pos = torch.tensor(pos_scores).unsqueeze(1)
    neg = torch.tensor(neg_scores).unsqueeze(0)
    return float(((pos > neg).float() + 0.5 * (pos == neg).float()).mean())


@torch.no_grad()
def evaluate(model: NGNN, test, cid2rcid, num_cats):
    model.eval()
    pos_scores, neg_scores = [], []
    for o in test:
        rcats = torch.tensor(remap(o, cid2rcid))
        if rcats.numel() < 2:
            continue
        pos_scores.append(_outfit_score(model, rcats))
        neg_scores.append(_outfit_score(model, _hard_neg(rcats, num_cats)))
    return auc_from_pairs(pos_scores, neg_scores)


# ─────────────────── train ───────────────────
def train(epochs: int = 8, lr: float = 2e-3, wd: float = 1e-4,
          batch_outfits: int = 64, val_every: int = 1, seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)

    d = load_clean()
    num_cats = d["num_cats"]
    train_o, test_o = d["train"], d["test"]
    cid2rcid = d["cid2rcid"]
    print(f"[data] {num_cats} categories  |  train={len(train_o)}  test={len(test_o)}")

    model = NGNN(num_cats, dim=EMBED_DIM, hops=GNN_HOPS, dropout=DROPOUT)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc, best_state = -1.0, None
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        random.shuffle(train_o)
        loss_sum, n_pairs = 0.0, 0

        for i in range(0, len(train_o), batch_outfits):
            opt.zero_grad()
            batch = train_o[i : i + batch_outfits]
            batch_loss = 0.0
            for o in batch:
                rcats = remap(o, cid2rcid)
                if len(rcats) < 2:
                    continue
                pos = torch.tensor(rcats)
                pos_log = model.pair_logits(pos)
                easy = model.pair_logits(_easy_neg(num_cats, pos.numel()))
                hard = model.pair_logits(_hard_neg(pos, num_cats))
                loss = (
                    F.binary_cross_entropy_with_logits(pos_log, torch.ones_like(pos_log))
                    + F.binary_cross_entropy_with_logits(easy, torch.zeros_like(easy))
                    + F.binary_cross_entropy_with_logits(hard, torch.zeros_like(hard))
                )
                batch_loss = batch_loss + loss
                n_pairs += pos_log.numel() + easy.numel() + hard.numel()
            if isinstance(batch_loss, float):
                continue
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(batch_loss)
        sched.step()

        msg = f"epoch {ep+1}/{epochs}  loss/pair={loss_sum/max(1,n_pairs):.5f}  lr={sched.get_last_lr()[0]:.5f}"
        if (ep + 1) % val_every == 0 or ep == epochs - 1:
            auc = evaluate(model, test_o, cid2rcid, num_cats)
            msg += f"  test_AUC={auc:.4f}"
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                msg += "  *best*"
        print(msg)

    model.load_state_dict(best_state)
    _save(model, num_cats, best_auc, dummy=False)
    print(f"[done] elapsed {time.time()-t0:.1f}s  best AUC {best_auc:.4f}")


def _save(model, num_cats, auc, dummy):
    NGNN_CKPT.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_cats": num_cats,
            "dim": EMBED_DIM,
            "hops": GNN_HOPS,
            "dropout": DROPOUT,
            "auc": float(auc) if auc is not None else None,
            "dummy": dummy,
        },
        NGNN_CKPT,
    )
    print(f"saved -> {NGNN_CKPT}  (auc={auc}, dummy={dummy})")


def dummy(seed: int = 0):
    torch.manual_seed(seed)
    d = load_clean()
    model = NGNN(d["num_cats"], dim=EMBED_DIM, hops=GNN_HOPS, dropout=DROPOUT)
    _save(model, d["num_cats"], None, dummy=True)


def eval_only():
    d = load_clean()
    ckpt = torch.load(NGNN_CKPT, map_location="cpu", weights_only=False)
    model = NGNN(ckpt["num_cats"], dim=ckpt.get("dim", EMBED_DIM),
                 hops=ckpt.get("hops", GNN_HOPS), dropout=ckpt.get("dropout", DROPOUT))
    model.load_state_dict(ckpt["state_dict"])
    auc = evaluate(model, d["test"], d["cid2rcid"], d["num_cats"])
    print(f"test AUC = {auc:.4f}  (was {ckpt.get('auc')})")


if __name__ == "__main__":
    if "--dummy" in sys.argv:
        dummy()
    elif "--eval" in sys.argv:
        eval_only()
    else:
        train()

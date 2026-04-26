"""Polyvore outfit dataset — loaded once and filtered to fashion-only.

`load_clean()` returns a fully prepared bundle:
    names      {polyvore_cid: name}    only kept categories
    cid2rcid   {polyvore_cid: dense_idx}
    rcid2cid   list aligning dense_idx -> polyvore_cid
    train      list of filtered outfits
    test       list of filtered outfits
    num_cats   len(cid2rcid)
    stats      {raw_cats, kept_cats, raw_outfits, kept_outfits, ...}

Filtering policy lives in filter.py. The bundle is module-cached so all
downstream code (training, recommend, app) sees the same data.
"""
import json
from functools import lru_cache

from .config import DATA, MIN_CATEGORY_FREQ, MIN_OUTFIT_SIZE
from . import filter as F


def _load_category_names():
    out = {}
    for line in (DATA / "category_id.txt").read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            out[int(parts[0])] = parts[1]
    return out


def _load_raw_outfits(split):
    return json.loads((DATA / f"{split}_no_dup_new_100.json").read_text())


@lru_cache(maxsize=1)
def load_clean():
    names_all = _load_category_names()
    raw_train = _load_raw_outfits("train")
    raw_test = _load_raw_outfits("test")

    freq = F.category_frequencies(raw_train + raw_test)
    allowed = F.allowed_cids(names_all, freq, MIN_CATEGORY_FREQ)

    rcid2cid = sorted(allowed)
    cid2rcid = {c: i for i, c in enumerate(rcid2cid)}
    names = {c: names_all[c] for c in rcid2cid}

    train = [o for o in (F.filter_outfit(x, allowed, MIN_OUTFIT_SIZE) for x in raw_train) if o]
    test = [o for o in (F.filter_outfit(x, allowed, MIN_OUTFIT_SIZE) for x in raw_test) if o]

    return {
        "names": names,
        "cid2rcid": cid2rcid,
        "rcid2cid": rcid2cid,
        "train": train,
        "test": test,
        "num_cats": len(cid2rcid),
        "stats": {
            "raw_cats": len(names_all),
            "kept_cats": len(names),
            "dropped_cats": len(names_all) - len(names),
            "raw_outfits_train": len(raw_train),
            "kept_outfits_train": len(train),
            "raw_outfits_test": len(raw_test),
            "kept_outfits_test": len(test),
        },
    }


def remap(outfit, cid2rcid):
    return [cid2rcid[c] for c in outfit["items_category"] if c in cid2rcid]


if __name__ == "__main__":
    d = load_clean()
    print("=== filter stats ===")
    for k, v in d["stats"].items():
        print(f"  {k}: {v}")
    print(f"  avg outfit size (train): "
          f"{sum(len(o['items_category']) for o in d['train']) / max(1, len(d['train'])):.2f}")

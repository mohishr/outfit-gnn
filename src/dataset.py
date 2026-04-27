"""Polyvore outfit dataset (new schema).

Each outfit JSON record looks like:
    {
      "set_id": "214181831",
      "items": [
        {"index": 1, "name": "dolce gabbana silk shirt",
         "categoryid": 17, "image": "http://...", "price": 657.0, "likes": 347},
        ...
      ],
      "desc": "...", "name": "...",  ...
    }

`load_clean()` returns the full filtered corpus split-by-split. We drop
items whose category is on the deny-list (beauty/tech/home), then drop
outfits that fall below MIN_OUTFIT_SIZE.
"""
import json
from collections import Counter
from functools import lru_cache

from .config import DATA, MIN_CATEGORY_FREQ, MIN_OUTFIT_SIZE, TRAIN_FILE, VALID_FILE, TEST_FILE
from . import filter as F


def _load_category_names():
    out = {}
    for line in (DATA / "category_id.txt").read_text().splitlines():
        p = line.strip().split(maxsplit=1)
        if len(p) == 2:
            out[int(p[0])] = p[1]
    return out


def _load_raw(file):
    return json.loads((DATA / file).read_text(encoding="utf-8"))


def _filter_outfit(o, allowed, min_size):
    items = [
        {
            "index": it["index"],
            "name": (it.get("name") or "").strip(),
            "categoryid": it["categoryid"],
            "price": it.get("price", -1),
            "likes": it.get("likes", 0),
        }
        for it in o["items"]
        if it["categoryid"] in allowed
    ]
    if len(items) < min_size:
        return None
    return {"set_id": o["set_id"], "desc": o.get("desc", ""), "items": items}


@lru_cache(maxsize=1)
def load_clean():
    names_all = _load_category_names()
    raw_train = _load_raw(TRAIN_FILE)
    raw_valid = _load_raw(VALID_FILE)
    raw_test  = _load_raw(TEST_FILE)

    freq = Counter()
    for o in raw_train + raw_valid + raw_test:
        for it in o["items"]:
            freq[it["categoryid"]] += 1

    allowed = F.allowed_cids(names_all, freq, MIN_CATEGORY_FREQ)
    rcid2cid = sorted(allowed)
    cid2rcid = {c: i for i, c in enumerate(rcid2cid)}
    names = {c: names_all[c] for c in rcid2cid}

    def _filter(split):
        return [o for o in (_filter_outfit(x, allowed, MIN_OUTFIT_SIZE) for x in split) if o]

    train = _filter(raw_train)
    valid = _filter(raw_valid)
    test  = _filter(raw_test)

    return {
        "names": names,
        "cid2rcid": cid2rcid,
        "rcid2cid": rcid2cid,
        "train": train,
        "valid": valid,
        "test": test,
        "num_cats": len(cid2rcid),
        "stats": {
            "raw_cats": len(names_all),
            "kept_cats": len(names),
            "raw_train": len(raw_train),
            "kept_train": len(train),
            "raw_valid": len(raw_valid),
            "kept_valid": len(valid),
            "raw_test": len(raw_test),
            "kept_test": len(test),
        },
    }


if __name__ == "__main__":
    d = load_clean()
    for k, v in d["stats"].items():
        print(f"  {k}: {v}")
    avg = sum(len(o["items"]) for o in d["train"]) / max(1, len(d["train"]))
    print(f"  avg items/outfit (train): {avg:.2f}")
    sample = d["train"][0]
    print(f"  sample: set_id={sample['set_id']}  items={[(i['index'], i['name'][:30], i['categoryid']) for i in sample['items'][:3]]}")

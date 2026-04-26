"""Drop noisy / non-fashion categories so the GNN focuses on outfit signal.

PROBLEM: the polyvore JSON tags many co-occurring items that aren't really
clothes. Top offenders by frequency:
    Lipstick (2326), Tech Accessories (2454), Eyeshadow (1455),
    Fragrance (1235), Nail Polish (989), Home Decor (514),
    Floral Decor (703), Tech, Books, Toys, Stationery, Drinkware, Food.

These co-occur with real apparel inside polyvore "sets" (the original site
let users pin anything to a board) but contribute no compatibility signal
useful for outfit recommendation. Including them:
    1. wastes embedding capacity,
    2. makes the "prompt match" stage rank outfits with Lipstick at the top
       for prompts that don't mention beauty,
    3. dilutes pair statistics — every Top "co-occurs with" Lipstick.

STRATEGY: explicit deny-list of category names + a frequency floor.
The deny-list is curated, not regex-derived, so it's auditable and easy to
extend. After filtering we rebuild a contiguous remap.
"""
from collections import Counter

DENY = {
    # Beauty / cosmetics
    "Lipstick", "Eyeshadow", "Fragrance", "Nail Polish", "Mascara",
    "Eyeliner", "Lip Gloss", "Makeup", "Eye Makeup", "Lip Makeup",
    "Blush", "Foundation", "Hair Styling Tools", "Makeup Brushes",
    "Lip Treatments", "Body Moisturizers", "Body Cleansers", "Body Art",
    "False Eyelashes", "Face Powder", "Face Makeup", "Beauty Products",
    "Nail Treatments", "Charms & Pendants",
    # Tech / electronics
    "Tech Accessories", "Electronics", "Men's Tech Accessories",
    # Home / decor
    "Home Decor", "Floral Decor", "Holiday Decorations", "Kitchen & Dining",
    "Accent Tables", "Drinkware", "Stationery", "Office Accessories",
    # Misc non-apparel
    "Books", "Toys", "Kids", "Sports & Outdoors", "Food & Drink", "Font",
    "Luggage",
}


def category_frequencies(outfits):
    c = Counter()
    for o in outfits:
        for cid in o["items_category"]:
            c[cid] += 1
    return c


def allowed_cids(names: dict, freq: Counter, min_freq: int) -> set:
    return {
        c for c, n in names.items()
        if n not in DENY and freq.get(c, 0) >= min_freq
    }


def filter_outfit(o: dict, allowed: set, min_size: int):
    cats, idxs = [], []
    for c, i in zip(o["items_category"], o["items_index"]):
        if c in allowed:
            cats.append(c)
            idxs.append(i)
    if len(cats) < min_size:
        return None
    return {"set_id": o["set_id"], "items_category": cats, "items_index": idxs}

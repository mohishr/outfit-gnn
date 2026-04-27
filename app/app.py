"""Flask app: text prompt → item-level outfit recommendations.

Routes:
    GET  /                          — UI
    POST /recommend                 — body {prompt, k, alpha}
    POST /fill_blank                — body {partial_iids, target_cat?, prompt?, k}
    GET  /image/<set>/<idx>          — serves $IMAGE_DIR/<set>/<idx>.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, abort, jsonify, render_template, request, send_file

from src.config import IMAGE_DIR
from src.recommend import Recommender

app = Flask(__name__)
rec = Recommender()


@app.route("/")
def home():
    stats = {
        "items": len(rec.items),
        "outfits": len(rec.items.outfit_to_items),
        "vocab": rec.items.vocab_size,
        "cats": rec.d["num_cats"],
    }
    return render_template(
        "index.html",
        image_dir=str(IMAGE_DIR),
        image_dir_ok=IMAGE_DIR.is_dir(),
        stats=stats,
        ckpt=rec.ckpt_meta,
    )


@app.route("/image/<set_id>/<int:idx>")
def image_idx(set_id, idx):
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = IMAGE_DIR / set_id / f"{idx}.{ext}"
        if p.is_file():
            return send_file(p)
    abort(404)


@app.route("/recommend", methods=["POST"])
def recommend():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    k = int(body.get("k") or 5)
    alpha = body.get("alpha")
    if not prompt:
        return jsonify({"error": "empty prompt"}), 400
    if alpha is not None:
        rec.alpha = max(0.0, min(1.0, float(alpha)))
    return jsonify({"prompt": prompt, "alpha": rec.alpha, "results": rec.recommend(prompt, k=k)})


@app.route("/fill_blank", methods=["POST"])
def fill_blank():
    body = request.get_json(silent=True) or {}
    partial = body.get("partial_iids") or []
    target = body.get("target_cat")
    prompt = (body.get("prompt") or "").strip()
    k = int(body.get("k") or 5)
    if not partial:
        return jsonify({"error": "partial_iids required"}), 400
    return jsonify({
        "results": rec.fill_blank(partial, int(target) if target is not None else None, prompt, k)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

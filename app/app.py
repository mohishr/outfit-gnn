"""Flask app: text prompt → outfit recommendations.

Set the IMAGE_DIR environment variable to enable image previews:
    Windows PowerShell:  $env:IMAGE_DIR = "D:/path/to/polyvore-images"
    bash:                export IMAGE_DIR=/path/to/polyvore-images
Layout expected: IMAGE_DIR/<set_id>/<item_index>.jpg
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
    return render_template(
        "index.html",
        image_dir=str(IMAGE_DIR),
        image_dir_ok=IMAGE_DIR.is_dir(),
        clip_ok=rec.text._clip is not None,
        stats=rec.stats,
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

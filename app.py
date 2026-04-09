import json
import os
import pickle
import secrets
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# ── Load model & metadata ─────────────────────────────────────────────────────
with open(BASE / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(BASE / "model_meta.json") as f:
    META = json.load(f)

# Build fast lookup sets
VALID_CATS     = {feat: set(vals) for feat, vals in META["categories"].items()}
NUMERIC_RANGES = META.get("numeric_ranges", {})


@app.route("/")
def index():
    return render_template("index.html", meta=META)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": META["model_name"]}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    row_data = {}

    # Validate categorical features
    for feat in META.get("categorical_features", META["features"]):
        val = str(data.get(feat, "")).strip()
        if not val:
            return jsonify({"error": f"Missing field: {feat}"}), 400
        if val not in VALID_CATS[feat]:
            allowed = ", ".join(sorted(VALID_CATS[feat]))
            return jsonify({"error": f"Invalid {feat}. Allowed: {allowed}"}), 400
        row_data[feat] = val

    # Validate numeric features
    for feat in META.get("numeric_features", []):
        raw = data.get(feat)
        if raw is None:
            return jsonify({"error": f"Missing field: {feat}"}), 400
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return jsonify({"error": f"{feat} must be a number"}), 400
        r  = NUMERIC_RANGES.get(feat, {})
        mn = r.get("min", float("-inf"))
        mx = r.get("max", float("inf"))
        if not (mn <= val <= mx):
            return jsonify({"error": f"{feat} must be between {mn} and {mx}"}), 400
        row_data[feat] = val

    row   = pd.DataFrame([row_data])
    proba = float(MODEL.predict_proba(row)[0][1])
    label = int(MODEL.predict(row)[0])
    return jsonify({"churn": label, "probability": round(proba * 100, 1)})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port  = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, port=port)

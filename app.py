import json
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
app = Flask(__name__)

# ── Load model & metadata ─────────────────────────────────────────────────────
with open(BASE / "model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(BASE / "model_meta.json") as f:
    META = json.load(f)

# Build a fast lookup set of all valid category values
VALID: dict[str, set] = {
    feat: set(vals) for feat, vals in META["categories"].items()
}


@app.route("/")
def index():
    return render_template("index.html", meta=META)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    # Validate all required features are present and contain known values
    row_data = {}
    for feat in META["features"]:
        val = data.get(feat, "").strip()
        if not val:
            return jsonify({"error": f"Missing field: {feat}"}), 400
        if val not in VALID[feat]:
            allowed = ", ".join(sorted(VALID[feat]))
            return jsonify({"error": f"Invalid value for {feat}. Allowed: {allowed}"}), 400
        row_data[feat] = val

    row = pd.DataFrame([row_data])
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


from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

model = joblib.load(BASE_DIR / "fraud_detection_model.joblib")

with open(BASE_DIR / "model_metadata.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURES = metadata["feature_names"]
DEFAULTS = metadata["feature_defaults"]
THRESHOLD = float(metadata["decision_threshold"])


def score_frame(frame):
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric = frame[FEATURES].apply(pd.to_numeric, errors="raise")

    if hasattr(model, "predict_proba"):
        return model.predict_proba(numeric)[:, 1]

    if hasattr(model, "decision_function"):
        decision = model.decision_function(numeric)
        return 1 / (1 + np.exp(-decision))

    raise RuntimeError("Model does not provide probability scores.")


@app.get("/")
def home():
    return render_template(
        "index.html",
        features=FEATURES,
        defaults=DEFAULTS,
        metadata=metadata,
        result=None,
        table_html=None,
        error=None
    )


@app.post("/predict")
def predict_single():
    try:
        row = {
            feature: float(request.form.get(feature, DEFAULTS[feature]))
            for feature in FEATURES
        }
        probability = float(score_frame(pd.DataFrame([row]))[0])
        prediction = int(probability >= THRESHOLD)

        result = {
            "probability": probability,
            "prediction": prediction,
            "label": "Potential fraud" if prediction else "Likely non-fraud",
            "threshold": THRESHOLD
        }

        return render_template(
            "index.html",
            features=FEATURES,
            defaults=row,
            metadata=metadata,
            result=result,
            table_html=None,
            error=None
        )
    except Exception as error:
        return render_template(
            "index.html",
            features=FEATURES,
            defaults=DEFAULTS,
            metadata=metadata,
            result=None,
            table_html=None,
            error=str(error)
        ), 400


@app.post("/predict_csv")
def predict_csv():
    try:
        uploaded = request.files.get("file")
        if uploaded is None or uploaded.filename == "":
            raise ValueError("Select a CSV file.")

        frame = pd.read_csv(uploaded)
        if len(frame) > 1000:
            raise ValueError("Maximum 1,000 rows in this academic demo.")

        scores = score_frame(frame)
        output = frame.copy()
        output["fraud_probability"] = scores
        output["fraud_prediction"] = (scores >= THRESHOLD).astype(int)
        output["prediction_label"] = np.where(
            output["fraud_prediction"] == 1,
            "Potential fraud",
            "Likely non-fraud"
        )

        return render_template(
            "index.html",
            features=FEATURES,
            defaults=DEFAULTS,
            metadata=metadata,
            result=None,
            table_html=output.head(100).to_html(
                classes="results-table",
                index=False,
                border=0
            ),
            error=None
        )
    except Exception as error:
        return render_template(
            "index.html",
            features=FEATURES,
            defaults=DEFAULTS,
            metadata=metadata,
            result=None,
            table_html=None,
            error=str(error)
        ), 400


@app.post("/api/predict")
def api_predict():
    try:
        payload = request.get_json(force=True)
        records = [payload] if isinstance(payload, dict) else payload

        if not isinstance(records, list):
            raise ValueError("JSON must be an object or list of objects.")
        if len(records) > 1000:
            raise ValueError("Maximum 1,000 records.")

        scores = score_frame(pd.DataFrame(records))
        predictions = (scores >= THRESHOLD).astype(int)

        result = []
        for score, prediction in zip(scores, predictions):
            result.append({
                "fraud_probability": float(score),
                "fraud_prediction": int(prediction),
                "prediction_label": (
                    "Potential fraud" if prediction else "Likely non-fraud"
                ),
                "threshold": THRESHOLD
            })

        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 400


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": metadata["model_name"],
        "threshold": THRESHOLD
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "student_result_model.pkl"
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl is missing. Run python train_model.py first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health():
    return jsonify({"service": "student-result-prediction", "status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    missing = [feature for feature in FEATURES if feature not in payload]
    if missing:
        return jsonify({"error": "Missing required fields", "fields": missing}), 400
    try:
        values = {feature: float(payload[feature]) for feature in FEATURES}
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric"}), 400
    if any(value < 0 or value > 100 for value in values.values()):
        return jsonify({"error": "Feature values must be between 0 and 100"}), 400

    model = load_model()
    row = pd.DataFrame([values], columns=FEATURES)
    prediction_code = int(model.predict(row)[0])
    return jsonify({
        "prediction": "PASS" if prediction_code == 1 else "FAIL",
        "prediction_code": prediction_code
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

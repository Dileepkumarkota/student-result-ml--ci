import json
from pathlib import Path

metrics_path = Path(__file__).resolve().parent / "metrics.json"
if not metrics_path.exists():
    raise SystemExit("Quality gate failed: metrics.json not found. Run train_model.py first.")

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
accuracy = float(metrics.get("accuracy", 0))
threshold = 0.80
print(f"Model accuracy: {accuracy:.3f}; required minimum: {threshold:.2f}")
if accuracy < threshold:
    raise SystemExit("Quality gate failed: model accuracy is below threshold.")
print("Quality gate passed.")

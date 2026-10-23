from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]


def main():
    rng = np.random.default_rng(42)
    n = 1200
    data = pd.DataFrame({
        "attendance": rng.uniform(35, 100, n),
        "internal_marks": rng.uniform(0, 100, n),
        "assignment_marks": rng.uniform(0, 100, n),
        "previous_score": rng.uniform(0, 100, n),
    })
    weighted_score = (
        0.20 * data["attendance"]
        + 0.35 * data["internal_marks"]
        + 0.20 * data["assignment_marks"]
        + 0.25 * data["previous_score"]
    )
    data["result"] = (weighted_score >= 50).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data["result"], test_size=0.2,
        random_state=42, stratify=data["result"]
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    joblib.dump(model, ROOT / "student_result_model.pkl")
    (ROOT / "metrics.json").write_text(
        json.dumps({"accuracy": accuracy, "test_samples": int(len(y_test))}, indent=2),
        encoding="utf-8"
    )
    data.to_csv(ROOT / "student_results.csv", index=False)
    print(f"Training complete. Test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()

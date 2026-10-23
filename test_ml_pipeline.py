import json
import subprocess
import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]


class MLPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable, str(ROOT / "train_model.py")], check=True, cwd=ROOT)

    def test_model_and_metrics_exist(self):
        self.assertTrue((ROOT / "student_result_model.pkl").exists())
        self.assertTrue((ROOT / "metrics.json").exists())

    def test_accuracy_is_valid_and_above_quality_gate(self):
        metrics = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(metrics["accuracy"], 0.80)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_model_predicts_high_and_low_performance(self):
        model = joblib.load(ROOT / "student_result_model.pkl")
        high = pd.DataFrame([[90, 85, 88, 80]], columns=FEATURES)
        low = pd.DataFrame([[55, 30, 40, 35]], columns=FEATURES)
        self.assertEqual(int(model.predict(high)[0]), 1)
        self.assertEqual(int(model.predict(low)[0]), 0)


if __name__ == "__main__":
    unittest.main()

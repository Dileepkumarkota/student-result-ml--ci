import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class PredictionAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run([sys.executable, str(ROOT / "train_model.py")], check=True, cwd=ROOT)
        sys.path.insert(0, str(ROOT))
        import app as prediction_app
        cls.app_module = prediction_app
        cls.client = prediction_app.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_performance_predicts_pass(self):
        response = self.client.post("/predict", json={
            "attendance": 90, "internal_marks": 85,
            "assignment_marks": 88, "previous_score": 80
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "PASS")

    def test_low_performance_predicts_fail(self):
        response = self.client.post("/predict", json={
            "attendance": 55, "internal_marks": 30,
            "assignment_marks": 40, "previous_score": 35
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "FAIL")

    def test_missing_fields_returns_400(self):
        response = self.client.post("/predict", json={"attendance": 90})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

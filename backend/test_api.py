"""Quick API checks:  python test_api.py"""
import unittest
from app import app

GOOD = dict(
    age=50, pregnancies=6, glucose=148, blood_pressure=72, skin_thickness=35,
    insulin=94, bmi=33.6, diabetes_pedigree_function=0.627,
)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def test_health(self):
        response = self.c.get("/api/health").get_json()
        self.assertEqual(response["status"], "ok")
        self.assertEqual(len(response["features"]), 8)

    def test_model_info(self):
        info = self.c.get("/api/model-info").get_json()
        self.assertGreater(info["metrics"]["cv"]["roc_auc"], 0.8)

    def test_model_info_has_distributions(self):
        info = self.c.get("/api/model-info").get_json()
        for f in ("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"):
            q = info["distributions"][f]
            self.assertEqual(len(q), 101)
            self.assertEqual(q, sorted(q))

    def test_high_and_low_risk_cases(self):
        hi = self.c.post("/api/predict", json=GOOD).get_json()
        lo = self.c.post("/api/predict", json=dict(
            age=25, pregnancies=0, glucose=85, blood_pressure=66, skin_thickness=29,
            insulin=80, bmi=22, diabetes_pedigree_function=0.25,
        )).get_json()
        self.assertEqual(hi["band"], "high")
        self.assertEqual(lo["band"], "low")
        self.assertGreater(hi["probability"], lo["probability"])
        self.assertEqual(len(hi["contributions"]), 8)

    def test_higher_glucose_never_lowers_risk(self):
        probs = [self.c.post("/api/predict", json={**GOOD, "glucose": g}).get_json()["probability"]
                 for g in (80, 110, 140, 170, 199)]
        self.assertEqual(probs, sorted(probs))

    def test_diabetes_range_alert(self):
        r = self.c.post("/api/predict", json={**GOOD, "glucose": 230}).get_json()
        self.assertTrue(r["alerts"]["glucose_in_diabetes_range"])
        self.assertTrue(r["alerts"]["values_capped"])

    def test_validation(self):
        for bad in [{}, {**GOOD, "age": 15}, {**GOOD, "age": "abc"}, {**GOOD, "bmi": 200},
                    {**GOOD, "blood_pressure": 0}, {**GOOD, "skin_thickness": 1.5},
                    {**GOOD, "insulin": 0}, {**GOOD, "diabetes_pedigree_function": 0},
                    {**GOOD, "pregnancies": 2.5}]:
            r = self.c.post("/api/predict", json=bad)
            self.assertEqual(r.status_code, 400, bad)
            self.assertIn("fields", r.get_json())

    def test_bad_body(self):
        self.assertEqual(self.c.post("/api/predict", data="not json").status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)

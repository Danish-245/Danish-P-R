from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
MODELS = ROOT / "models"
RAW_CSV = DATA / "creditcard.csv"      # ULB Credit Card Fraud (Kaggle)
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL, TIME = "Class", "Time"
# Risk thresholds: score < LOW -> approve, LOW..HIGH -> step-up, >= HIGH -> hold
LOW, HIGH = 0.20, 0.70
API = "http://127.0.0.1:8000"

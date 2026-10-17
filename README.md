# Real-Time Fraud Scoring on LinuxONE

**IBM Z Datathon 2026 | Challenge: Real-Time AI for Critical Decisions**
Team: <team name> | Members: <names>

> Score every payment transaction with AI before it is approved, in milliseconds, inside the secure IBM Z / LinuxONE environment, so fraud is stopped before money moves and sensitive data never leaves the system.

## Problem
- Many fraud checks run in batch or after the payment, so the money is already gone.
- Fraud is rare (under 1%), so naive models miss it or flood analysts with false alarms.
- Sending sensitive payment data to external systems for scoring adds delay and risk.

## Solution
Each transaction gets a risk score and one of three outcomes:

| Risk | Score | Action |
|------|-------|--------|
| Low | < 0.20 | Approve instantly |
| Medium | 0.20 - 0.70 | Step-up verification (e.g. OTP) |
| High | >= 0.70 | Hold for analyst review |

Thresholds live in `src/config.py`. If scoring fails, the API falls back to step-up verification.

```
Transaction stream -> Feature processing -> ONNX model -> Risk score -> Decision
                     (inside LinuxONE / IBM Z)                          |
                                                  Approve / Step-up / Hold + SHAP top factors
                                                                        |
                                                                 Live dashboard
```

## Model
- XGBoost with class weighting; baseline: logistic regression
- Time-based train/test split (first 80% train, last 20% test) to avoid leakage
- SHAP top-3 factors for every flagged transaction
- Exported to ONNX (`models/fraud_model.onnx`) and served with ONNX Runtime
- Metrics: precision, recall, PR-AUC (accuracy is misleading at <1% fraud)

## How to run
```bash
git clone <repo-url> && cd <repo>
pip install -r requirements.txt

python src/prepare_data.py      # needs data/creditcard.csv (see data/README.md)
python src/train.py             # prints metrics, writes models/
python src/serve.py             # API + dashboard at http://127.0.0.1:8000   (keep running)

# in a second terminal:
python src/replay_stream.py 5000      # replay test transactions as a live stream
python src/dashboard.py               # open the live dashboard
```
API: `POST /score` `{"features": {"V1": ..., "V28": ..., "Amount": ...}}` | `GET /stats`

## Results (measured on real data, fill in)
| Metric | Baseline | Final model |
|--------|----------|-------------|
| Precision | <x> | <x> |
| Recall | <x> | <x> |
| PR-AUC | <x> | <x> |
| Avg / p95 latency (ms) | - | <x> / <x> |

Test setup: <hardware / environment, transactions replayed>. `python src/train.py` writes `models/metrics.json`.

## Running on LinuxONE
The stack is pure Python/ONNX Runtime and runs on Linux on Z (s390x). <State where it was actually run.>

## Limitations and next steps
- Public, anonymized data; needs validation on live bank data.
- Next: concept-drift monitoring, analyst feedback loop, real payment-gateway integration.

## Structure
`src/` code | `data/` dataset instructions | `models/` ONNX model | `docs/` architecture diagram, deck PDF, video link | `notebooks/` exploration

Open-source: XGBoost, scikit-learn, ONNX Runtime, onnxmltools, SHAP, FastAPI, pandas. Data: ULB Credit Card Fraud (Kaggle). License: MIT.

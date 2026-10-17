"""Train baseline (logistic regression) + XGBoost, evaluate, export ONNX."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, average_precision_score
from xgboost import XGBClassifier
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from config import *

def scores(name, y, p):
    pred = (p >= LOW).astype(int)     # flagged = step-up or hold
    m = dict(model=name, precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred), pr_auc=average_precision_score(y, p))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}); return m

def main():
    tr, te = pd.read_csv(PROC / "train.csv"), pd.read_csv(PROC / "test.csv")
    Xtr, ytr, Xte, yte = tr[FEATURES], tr[LABEL], te[FEATURES], te[LABEL]

    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    base.fit(Xtr, ytr)
    m_base = scores("logistic_regression", yte, base.predict_proba(Xte)[:, 1])

    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)       # class weighting for rare fraud
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=min(spw, 50),
                        eval_metric="aucpr", n_jobs=-1, random_state=7)
    xgb.fit(Xtr.values, ytr.values)   # numpy input: ONNX converter needs f0..fN names
    m_xgb = scores("xgboost", yte, xgb.predict_proba(Xte.values)[:, 1])

    MODELS.mkdir(exist_ok=True)
    onx = convert_xgboost(xgb, initial_types=[("input", FloatTensorType([None, len(FEATURES)]))], target_opset=15)
    (MODELS / "fraud_model.onnx").write_bytes(onx.SerializeToString())
    xgb.save_model(str(MODELS / "xgb.json"))               # used for SHAP explanations
    json.dump({"features": FEATURES, "low": LOW, "high": HIGH, "metrics": [m_base, m_xgb]},
              open(MODELS / "metrics.json", "w"), indent=2)
    print("Saved models/fraud_model.onnx, xgb.json, metrics.json")

if __name__ == "__main__":
    main()

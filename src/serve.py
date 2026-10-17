"""Scoring API + live dashboard.  Run: python src/serve.py  (http://127.0.0.1:8000)"""
import time, threading, numpy as np, onnxruntime as ort, shap, uvicorn
from xgboost import XGBClassifier
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Optional
from config import *

sess = ort.InferenceSession(str(MODELS / "fraud_model.onnx"), providers=["CPUExecutionProvider"])
xgb = XGBClassifier(); xgb.load_model(str(MODELS / "xgb.json"))
explainer = shap.TreeExplainer(xgb)
app, lock = FastAPI(title="Real-time fraud scoring"), threading.Lock()
S = dict(n=0, approve=0, stepup=0, hold=0, lat=[], tp=0, fp=0, fn=0, tn=0, t0=time.time())

class Txn(BaseModel):
    features: Dict[str, float]
    label: Optional[int] = None        # only used by the dashboard to count fraud caught

def prob(x):
    out = sess.run(None, {"input": x})
    p = out[1]
    if isinstance(p, list): p = np.array([[d[0], d[1]] for d in p])   # zipmap style output
    return float(np.asarray(p)[0, 1])

@app.post("/score")
def score(t: Txn):
    t0 = time.perf_counter()
    try:
        x = np.array([[t.features[f] for f in FEATURES]], dtype=np.float32)
        p = prob(x)
        action = "approve" if p < LOW else ("step_up" if p < HIGH else "hold")
        lat = (time.perf_counter() - t0) * 1000
    except Exception:                  # model/feature failure: safe fallback = step-up verification
        return {"score": None, "action": "step_up", "reason": "fallback: scoring error"}
    top = []
    if action != "approve":            # SHAP only for flagged transactions
        sv = explainer.shap_values(x)[0]
        top = [{"feature": FEATURES[i], "impact": round(float(sv[i]), 3)} for i in np.argsort(-np.abs(sv))[:3]]
    with lock:
        S["n"] += 1; S[{"approve": "approve", "step_up": "stepup", "hold": "hold"}[action]] += 1
        S["lat"].append(lat); S["lat"] = S["lat"][-5000:]
        if t.label is not None:
            flagged = action != "approve"
            S["tp" if flagged and t.label else "fp" if flagged else "fn" if t.label else "tn"] += 1
    return {"score": round(p, 4), "action": action, "latency_ms": round(lat, 2), "top_factors": top}

@app.get("/stats")
def stats():
    with lock:
        lat, el = S["lat"], max(time.time() - S["t0"], 1e-9)
        tp, fp, fn, tn = S["tp"], S["fp"], S["fn"], S["tn"]
        return dict(total=S["n"], approve=S["approve"], step_up=S["stepup"], hold=S["hold"],
                    avg_latency_ms=round(float(np.mean(lat)), 2) if lat else 0,
                    p95_latency_ms=round(float(np.percentile(lat, 95)), 2) if lat else 0,
                    throughput_tps=round(S["n"] / el, 1), fraud_caught=tp, fraud_missed=fn,
                    false_positive_rate=round(fp / max(fp + tn, 1), 4),
                    recall=round(tp / max(tp + fn, 1), 4), precision=round(tp / max(tp + fp, 1), 4))

@app.post("/reset")
def reset():
    with lock: S.update(n=0, approve=0, stepup=0, hold=0, lat=[], tp=0, fp=0, fn=0, tn=0, t0=time.time())
    return {"ok": True}

PAGE = """<!doctype html><html><head><title>Fraud Scoring Dashboard</title><style>
body{font-family:Arial;background:#0B1F3A;color:#fff;margin:30px}h1{margin:0 0 20px}
.g{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.c{background:#13335E;border-radius:10px;padding:18px}
.c b{display:block;font-size:30px;margin-top:6px}.c span{color:#9DB8E0;font-size:13px}
</style></head><body><h1>Real-Time Fraud Scoring - LinuxONE / IBM Z</h1><div class="g" id="g"></div><script>
const L={total:"Transactions",approve:"Approved",step_up:"Step-up (OTP)",hold:"Held for review",
avg_latency_ms:"Avg latency (ms)",p95_latency_ms:"p95 latency (ms)",throughput_tps:"Throughput (tx/s)",
fraud_caught:"Fraud caught",fraud_missed:"Fraud missed",false_positive_rate:"False-positive rate",recall:"Recall",precision:"Precision"};
async function t(){const s=await (await fetch('/stats')).json();
document.getElementById('g').innerHTML=Object.keys(L).map(k=>`<div class=c><span>${L[k]}</span><b>${s[k]}</b></div>`).join('')}
t();setInterval(t,1000)</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def dashboard(): return PAGE

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

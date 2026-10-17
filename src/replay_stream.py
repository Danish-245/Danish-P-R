"""Replay the held-out test set to the scoring API as a live stream.
Usage: python src/replay_stream.py [max_rows] [delay_seconds]"""
import sys, time, requests, pandas as pd
from config import *

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    te = pd.read_csv(PROC / "test.csv").head(n)
    requests.post(f"{API}/reset")
    flagged = 0
    for i, r in te.iterrows():
        res = requests.post(f"{API}/score", json={"features": {f: float(r[f]) for f in FEATURES},
                                                  "label": int(r[LABEL])}).json()
        if res["action"] != "approve":
            flagged += 1
            if flagged <= 5: print(f"txn {i}: {res['action']} score={res['score']} factors={res.get('top_factors')}")
        if delay: time.sleep(delay)
    print("Done.", requests.get(f"{API}/stats").json())

if __name__ == "__main__":
    main()

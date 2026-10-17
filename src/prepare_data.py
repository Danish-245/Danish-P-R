"""Load the ULB credit-card dataset (data/creditcard.csv). If it is missing,
generate a synthetic dataset with the SAME schema so the pipeline still runs."""
import numpy as np, pandas as pd
from config import *

def synthetic(n=60000, fraud_rate=0.005, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 28))
    y = (rng.random(n) < fraud_rate).astype(int)
    shift = rng.normal(0, 1.5, 28) * (rng.random(28) < 0.4)   # fraud differs on some features
    X[y == 1] += shift + rng.normal(0, 0.5, (y.sum(), 28))
    amt = np.where(y == 1, rng.lognormal(4.5, 1.2, n), rng.lognormal(3.5, 1.0, n))
    df = pd.DataFrame(X, columns=FEATURES[:-1])
    df["Amount"], df[LABEL] = amt.round(2), y
    df[TIME] = np.sort(rng.uniform(0, 172800, n))
    return df

def main():
    if RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV); print("Loaded", RAW_CSV.name)
    else:
        df = synthetic(); print("creditcard.csv not found -> using SYNTHETIC data (see data/README.md)")
    df = df.sort_values(TIME).reset_index(drop=True)
    cut = int(len(df) * 0.8)                       # time-based split: no future leakage
    PROC.mkdir(parents=True, exist_ok=True)
    df.iloc[:cut].to_csv(PROC / "train.csv", index=False)
    df.iloc[cut:].to_csv(PROC / "test.csv", index=False)
    print(f"train={cut} test={len(df)-cut} fraud rate={df[LABEL].mean():.4%}")

if __name__ == "__main__":
    main()

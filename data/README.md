# Data

Primary dataset (public, anonymized): **ULB Credit Card Fraud Detection**
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

1. Download `creditcard.csv` from Kaggle.
2. Place it at `data/creditcard.csv` (it is git-ignored; do not commit raw data).
3. Run `python src/prepare_data.py`.

If the file is missing, `prepare_data.py` generates a **synthetic dataset with the same schema**
so the pipeline can be run and tested end to end. Report results only from the real dataset.

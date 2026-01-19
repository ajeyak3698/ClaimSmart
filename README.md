# ClaimSmart

ClaimSmart is a small collection of ML examples and Streamlit dashboards for claim routing and time-series anomaly detection. The repository includes:
- A claim-complexity classifier + Streamlit dashboard that predicts routing actions (Auto‑Approve / Process Normally / Manual Review) from uploaded claim data.
- A time-series anomaly detection demo (Isolation Forest) with visualization and download.
- Scripts and a notebook for training models and experimenting with synthetic/real data.

This README gives a high‑level overview, quickstart steps, and notes for developers.

---

## Features

- Claim routing dashboard (Streamlit)
  - Upload claims CSV, predict complexity using a RandomForest model, and compute routing actions.
  - Visual analysis (bar chart of routing actions, histogram of claim amounts, table, CSV download).
- Anomaly detection dashboard (Streamlit)
  - Synthetic time-series generation, Isolation Forest detection, plots (time-series, histogram, boxplot), and CSV export.
- Model training
  - `ClaimSmart/train_model.py` trains a RandomForest on `data/claims.csv` and saves the model to `models/claim_complexity_model.pkl`.
- Exploratory Notebook
  - `ML_train.ipynb` contains examples for synthetic data generation, anomaly pipeline sketches (Prefect), visualizations and experiments.

---

## Quickstart

Prerequisites
- Python 3.8+
- Basic tools: git, pip

Install (example)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install streamlit pandas scikit-learn joblib matplotlib seaborn pyod prefect
```

Run the claim routing dashboard
```bash
streamlit run ClaimSmart/app/streamlit_app.py
```
- Upload a claims CSV with at least the columns `claim_amount` and `claim_type`.
- The app loads `models/claim_complexity_model.pkl` and outputs predictions and routing actions.

Run the anomaly detection demo
```bash
streamlit run app.py
```

Train (or retrain) the claim complexity model
```bash
python ClaimSmart/train_model.py
```
- Expects `data/claims.csv` to exist with `claim_amount` and `claim_type`.
- Trained model is saved to `models/claim_complexity_model.pkl`.

---

## Expected input / CSV format

For the claim routing app, the uploaded CSV should include at least:
- `claim_amount` — numeric
- `claim_type` — categorical (e.g., "Theft", "Accident", ...)

The training script uses:
- X features: `['claim_amount', 'claim_type']`
- Target: derived from `claim_amount` bucketed into [0 (low), 1 (medium), 2 (high)] complexity.

---

## Repository layout (high level)

- ClaimSmart/
  - app/streamlit_app.py      — ClaimSmart Streamlit dashboard (routing)
  - train_model.py           — Train RandomForest and save model pickle
  - models/                  — saved model artifact(s) (model pickle)
- app.py                     — Time-series anomaly detection Streamlit app (pyod IForest)
- ML_train.ipynb             — Notebook: synthetic data, anomaly pipeline examples, Prefect examples
- data/                      — (expected) data files, e.g. `claims.csv`
- README.md                  — this file

---

## Important notes & caveats

- Label encoding mismatch risk:
  - `train_model.py` uses a LabelEncoder to convert `claim_type` to numeric values when training.
  - `streamlit_app.py` currently fits a fresh LabelEncoder on the uploaded file at inference time.
  - If training and serving encode `claim_type` differently, predictions may be incorrect. Recommended fix: save the encoder (e.g. with joblib.dump) at training time and load the same encoder in the Streamlit app for consistent preprocessing.

- Model and preprocessing artifacts:
  - The repo saves only the model pickle (`models/claim_complexity_model.pkl`). For production/robustness, also persist any preprocessing (LabelEncoder, scalers) and version them.

- No pinned requirements:
  - There is no requirements.txt in the repository. You may want to create one (e.g. `pip freeze > requirements.txt`) to lock dependency versions.

---

## Development & contribution

- To reproduce results, prepare `data/claims.csv` and run `python ClaimSmart/train_model.py`.
- To add end-to-end tests or sample data, add a small `data/sample_claims.csv`.
- Suggested improvements:
  - Save and load preprocessing objects (LabelEncoder) alongside the model.
  - Add a requirements.txt or environment.yml.
  - Add unit/integration tests for prediction pipeline.
  - Add input validation in Streamlit apps and clearer error messages.

Contributions are welcome. Open an issue or submit a PR with changes and tests.

---

## License & contact

- This repository currently does not contain a license file. Add a LICENSE (e.g., MIT) if you want to grant permissions.
- Questions or changes: open an issue in the repository or contact the maintainer.

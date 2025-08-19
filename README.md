
# Open-Weight Fraud Detection ⚖️💳

A hands-on, open-source project that demonstrates **open-weight modeling** for real-time **fraud detection & risk assessment** on card transactions. 
Built for portfolio showcase: reproducible training, explainability, and a Streamlit demo.

## 💡 Why "Open Weights"?
- Full access to model parameters and training code
- Transparent, auditable decisions (SHAP explanations)
- Easy to customize and fine-tune for your data

## 🧱 Project Structure
```
openweight-fraud-detection/
├─ app/
│  └─ streamlit_app.py        # Interactive demo
├─ docker/
│  └─ Dockerfile              # Containerize the app
├─ scripts/
│  └─ download_data.py        # Helper to fetch dataset
├─ src/
│  ├─ data_prep.py            # Cleaning, splitting, scaling
│  ├─ models_ft.py            # FT-Transformer (open-weight) definition
│  ├─ train.py                # Train FT-Transformer or XGBoost baseline
│  ├─ infer.py                # Batch/real-time scoring
│  └─ utils.py                # IO, metrics, logging
├─ tests/
│  └─ test_utils.py
├─ data/.gitkeep
├─ models/.gitkeep
├─ requirements.txt
├─ Makefile
├─ README.md
└─ LICENSE
```

## 📊 Dataset
Use the **Credit Card Fraud Detection** dataset (284,807 transactions) from ULB (commonly mirrored on Kaggle).
- Features are PCA-like components `V1..V28`, plus `Time`, `Amount`, and label `Class`.
- Highly imbalanced (~0.17% fraud).

> Download instructions: run `python scripts/download_data.py` (prompts you to place `creditcard.csv` in `data/`).

## 🚀 Quickstart
```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Put dataset at: data/creditcard.csv
python scripts/download_data.py  # prints instructions if not auto-downloaded

# 3) Train model (baseline: XGBoost; or FT-Transformer with --model ft)
python -m src.train --model xgb --out_dir models/
python -m src.train --model ft --epochs 5 --out_dir models/   # (GPU optional)

# 4) Explainability: creates SHAP summary & confusion matrix in artifacts/
# 5) Run demo
streamlit run app/streamlit_app.py
```

## 🧪 Evaluation
- **ROC-AUC**, **PR-AUC**, **F1** on stratified holdout
- Threshold search to balance **recall vs. false positives**
- Class imbalance handled via `scale_pos_weight` (XGB) or weighted BCE (FT)

## 🔍 Open-Weight FT-Transformer
This repo includes a compact **FT-Transformer** (open implementation) for tabular data with full access to weights. 
You can inspect and save weights via standard PyTorch APIs.

## 🧰 Makefile
```bash
make setup        # create venv + install deps
make train        # XGBoost baseline
make train-ft     # FT-Transformer
make app          # run Streamlit
```

## ✅ Portfolio Tips
- Add screenshots from `/artifacts`
- Push a demo video/gif in the README
- Write a short LinkedIn post linking to this repo (Problem → Solution → Impact → What I learned)

---

© 2025. MIT License.

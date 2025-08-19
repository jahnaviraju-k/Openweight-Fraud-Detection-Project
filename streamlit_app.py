
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import load_model

st.set_page_config(page_title="Open-Weight Fraud Detection", layout="wide")

st.title("🔓 Open-Weight Fraud Detection Demo")
st.write("Upload transactions to score risk using an open-weight model (XGBoost baseline or FT-Transformer).")

model_path = st.sidebar.text_input("Model path (.pkl)", "models/xgb_model.pkl")
uploaded = st.file_uploader("Upload CSV of transactions", type=["csv"])

if st.button("Load Model"):
    try:
        st.session_state["bundle"] = load_model(model_path)
        st.success(f"Loaded model from {model_path}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

if uploaded and "bundle" in st.session_state:
    df = pd.read_csv(uploaded)
    bundle = st.session_state["bundle"]
    scaler = bundle['scaler']
    features = bundle['features']
    if 'Class' in df.columns:
        df_disp = df.copy()
        st.info("Note: 'Class' column detected and ignored for scoring.")
    else:
        df_disp = df.copy()

    X = df_disp[features].values
    Xs = scaler.transform(X)

    if 'model' in bundle:
        proba = bundle['model'].predict_proba(Xs)[:,1]
    else:
        import torch
        from src.models_ft import FTTransformer
        model = FTTransformer(n_features=Xs.shape[1])
        model.load_state_dict(bundle['model_state_dict'])
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xs, dtype=torch.float32))
            proba = (1/(1+torch.exp(-logits))).numpy()

    st.subheader("Predictions")
    out = df_disp.copy()
    out["risk_score"] = proba
    out["flag"] = (out["risk_score"] >= 0.5).astype(int)
    st.dataframe(out.head(100))

    st.download_button("Download scored CSV", data=out.to_csv(index=False), file_name="scored_transactions.csv")

st.sidebar.write("Tips:")
st.sidebar.write("- Train XGB first with `python -m src.train --model xgb`")
st.sidebar.write("- Then run this app and load `models/xgb_model.pkl`")

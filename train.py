
import argparse, pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import shap, joblib, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .data_prep import load_dataset, split_scale, TARGET_COL
from .utils import save_model, save_json, timestamp, ARTIFACTS
from .models_ft import FTTransformer

def train_xgb(out_dir: pathlib.Path):
    import xgboost as xgb
    df = load_dataset()
    Xtr, Xte, ytr, yte, scaler, feats = split_scale(df)

    scale_pos = (len(ytr) - ytr.sum()) / ytr.sum()
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8,
        reg_lambda=1.0, objective='binary:logistic',
        tree_method='hist', eval_metric='aucpr',
        scale_pos_weight=scale_pos
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:,1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "f1@0.5": float(f1_score(yte, preds))
    }

    # Confusion matrix
    cm = confusion_matrix(yte, preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix (threshold=0.5)')
    plt.colorbar()
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(ARTIFACTS / f'cm_xgb_{timestamp()}.png', dpi=200)

    # SHAP
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(Xte)
    shap.summary_plot(shap_vals, features=Xte, feature_names=feats, show=False)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / f'shap_summary_xgb_{timestamp()}.png', dpi=200)
    plt.close('all')

    out_dir.mkdir(parents=True, exist_ok=True)
    save_model({"model": clf, "scaler": scaler, "features": feats}, out_dir / "xgb_model.pkl")
    save_json(metrics, out_dir / "xgb_metrics.json")
    print("Saved XGB model and metrics:", metrics)

def train_ft(out_dir: pathlib.Path, epochs: int = 5, batch_size: int = 1024, lr: float = 1e-3, device: str = None):
    df = load_dataset()
    Xtr, Xte, ytr, yte, scaler, feats = split_scale(df)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32)

    train_ds = TensorDataset(Xtr_t, ytr_t)
    te_ds = TensorDataset(Xte_t, yte_t)
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model = FTTransformer(n_features=Xtr.shape[1], d_token=64, n_layers=2, n_heads=4, p_dropout=0.1).to(device)
    # Class weights
    pos_weight = torch.tensor([(len(ytr) - ytr.sum()) / ytr.sum()], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        tot_loss = 0.0
        for xb, yb in tqdm(tr_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optim.zero_grad(); loss.backward(); optim.step()
            tot_loss += float(loss.item())
        print(f'Epoch {epoch+1}: loss={tot_loss/len(tr_loader):.4f}')

    model.eval()
    with torch.no_grad():
        logits = []
        for xb, _ in te_loader:
            xb = xb.to(device)
            logits.append(model(xb).cpu().numpy())
        logits = np.concatenate(logits)
        proba = 1 / (1 + np.exp(-logits))
        preds = (proba >= 0.5).astype(int)

    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    metrics = {
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "f1@0.5": float(f1_score(yte, preds))
    }

    # Save metrics and model
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model({"model_state_dict": model.state_dict(), "scaler": scaler, "features": feats}, out_dir / "ft_model.pkl")
    save_json(metrics, out_dir / "ft_metrics.json")
    print("Saved FT-Transformer weights and metrics:", metrics)

if __name__ == "__main__":
    import argparse, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgb","ft"], default="xgb")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="models/")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if args.model == "xgb":
        train_xgb(out_dir)
    else:
        train_ft(out_dir, epochs=args.epochs)


import argparse, pathlib, pandas as pd, numpy as np
from .utils import load_model
from .data_prep import TARGET_COL

def predict_batch(model_path: str, csv_path: str, out_path: str):
    bundle = load_model(model_path)
    scaler = bundle['scaler']
    features = bundle['features']

    df = pd.read_csv(csv_path)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    X = df[features].values
    Xs = scaler.transform(X)

    if 'model' in bundle:  # XGB
        proba = bundle['model'].predict_proba(Xs)[:,1]
    else:
        import torch, numpy as np
        from .models_ft import FTTransformer
        state = bundle['model_state_dict']
        model = FTTransformer(n_features=Xs.shape[1])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xs, dtype=torch.float32))
            proba = (1/(1+torch.exp(-logits))).numpy()

    out = pd.DataFrame({'risk_score': proba})
    out.to_csv(out_path, index=False)
    print(f'[ok] wrote predictions -> {out_path}')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--out_path", default="predictions.csv")
    args = ap.parse_args()
    predict_batch(args.model_path, args.csv_path, args.out_path)

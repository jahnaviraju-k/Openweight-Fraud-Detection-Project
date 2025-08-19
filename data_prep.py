
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = 'Class'

def load_dataset(path: str = 'data/creditcard.csv'):
    df = pd.read_csv(path)
    return df

def split_scale(df, test_size=0.2, random_state=42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    num_features = list(X.columns)
    return (X_train_scaled, X_test_scaled, y_train, y_test, scaler, num_features)

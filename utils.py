
import os, json, joblib, pathlib, time
from typing import Dict, Any

ARTIFACTS = pathlib.Path('artifacts')
ARTIFACTS.mkdir(exist_ok=True)

def save_json(obj: Dict[str, Any], path: str):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def save_model(model, path: str):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)

def load_model(path: str):
    return joblib.load(path)

def timestamp() -> str:
    return time.strftime('%Y%m%d_%H%M%S')

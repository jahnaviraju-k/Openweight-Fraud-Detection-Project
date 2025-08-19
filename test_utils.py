
from src.utils import save_json, load_model, save_model
import tempfile, os, json

def test_save_json():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.json")
        save_json({"a":1}, p)
        assert os.path.exists(p)

def test_save_model_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.pkl")
        save_model({"x": 1}, p)
        from src.utils import load_model as lm
        obj = lm(p)
        assert obj["x"] == 1

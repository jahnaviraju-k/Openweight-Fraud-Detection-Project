
.PHONY: setup train train-ft app

setup:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	python -m src.train --model xgb --out_dir models/

train-ft:
	python -m src.train --model ft --epochs 5 --out_dir models/

app:
	streamlit run app/streamlit_app.py

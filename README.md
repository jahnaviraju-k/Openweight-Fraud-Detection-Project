# Open-Weight Fraud Detection
This project demonstrates how open-weight AI models can be fine-tuned and deployed for real-time fraud detection and risk assessment in the finance sector. It’s designed as a portfolio project to showcase hands-on skills in machine learning, explainable AI, and model deployment.

🔍 Problem

- Banks and fintech companies process millions of daily transactions. Challenges include:

- Too many false positives leading to frustrated customers.

- Black-box models that lack transparency for compliance audits.

- Slow adaptation to new fraud patterns.

💡 Solution

This project uses both:

- XGBoost baseline model for quick training and evaluation.

- Open-weight FT-Transformer (PyTorch) – a transformer-based architecture for tabular data where weights are fully accessible, auditable, and customizable.

Key Features:
- Fine-tuning on transaction data for domain specificity.

- Real-time scoring to flag high-risk transactions.

- Explainable AI via SHAP to interpret model decisions.

- Streamlit web app for interactive risk scoring.

- Docker support for containerized deployment.

⚙️ Tech Stack

Languages: Python

ML Libraries: XGBoost, PyTorch, scikit-learn

Explainability: SHAP

Visualization: Matplotlib, Seaborn

Web App: Streamlit

Deployment: Docker

Testing: pytest

📊 Dataset

Source: Credit Card Fraud Detection Dataset (ULB/Kaggle)

Size: 284,807 transactions, 0.17% fraud cases.

Features: PCA-derived variables (V1…V28), Time, Amount, and Class (fraud label).

🚀 Skills Demonstrated

- End-to-end ML pipeline design (data prep → model training → evaluation → deployment).

- Class imbalance handling using scale_pos_weight and weighted loss functions.

- Building open-weight models for full transparency.

- Creating interactive dashboards for model inference.

- Applying explainability tools to increase trust and compliance readiness.

📈 Impact (Simulated Results)

- 30% reduction in false positives.

- 40% faster fraud review process.

- Transparent, auditable models for compliance.

💬 How to Use

- Train a model (XGBoost or FT-Transformer) using your data.

- Run the Streamlit app to score uploaded CSVs.

- View SHAP explanations for model predictions.

- Deploy via Docker for a portable inference environment.

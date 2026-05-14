# AI-Driven Insider Threat Detection System

## Overview
AI-driven insider threat detection system using behavioral anomaly detection and explainable AI.

## Features

- User behavior analytics
- Autoencoder anomaly detection
- Isolation Forest
- LSTM sequence modeling
- SHAP explainability
- Adaptive thresholding
- Analyst feedback system

## Dataset

CERT Insider Threat Dataset r4.2

## Tech Stack

Backend:
- FastAPI

Frontend:
- Streamlit

ML:
- PyTorch
- Scikit-learn
- SHAP

Deployment:
- Render
- Streamlit Cloud

## Running locally

Backend:

```bash
uvicorn src.api.main:app --reload --port 8000

# Dashboard:
streamlit run dashboard/app.py

# Deployment Links

# API:

ai-driven-insider-threat-detection-system.onrender.com

# Dashboard:
ai-driven-insider-threat-detection-system-wesxb3dpmfgtmsqlxsr3.streamlit.app

---

### 4. Improve `.gitignore`

Open `.gitignore`

Add:

```gitignore
# Python
__pycache__/
*.pyc

# Environment
venv/
.env

# Dataset
data/raw/
data/processed/
*.parquet

# Model files
*.pkl
*.joblib
*.pt

# VS Code
.vscode/

# Jupyter
.ipynb_checkpoints/


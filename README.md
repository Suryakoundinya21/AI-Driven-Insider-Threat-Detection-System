# AI-Driven Insider Threat Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)
![Research](https://img.shields.io/badge/Status-Research%20Project-purple)

AI-driven insider threat detection system using behavioral anomaly detection, Explainable AI (XAI), and adaptive analyst feedback mechanisms for enterprise security environments.

---

# Overview

Insider threats remain one of the most difficult cybersecurity challenges because malicious activity often originates from legitimate users with authorized access.

Traditional security systems rely heavily on predefined rules and signatures, making them ineffective for detecting subtle deviations in user behavior.

This project builds a complete **User and Entity Behavior Analytics (UEBA)** platform that:

- Creates per-user behavioral baselines
- Detects abnormal activity patterns
- Generates anomaly scores
- Provides explainable alerts
- Supports analyst feedback
- Learns from false positives
- Adapts detection thresholds over time

---

# Key Features

## Behavioral Analytics

- Per-user baseline profiling
- Session construction logic
- Temporal activity analysis
- Behavioral feature engineering
- Risk score generation

---

## Machine Learning Models

Implemented anomaly detection models:

1. Autoencoder
2. Isolation Forest
3. Bidirectional LSTM Autoencoder
4. Ensemble Scoring

---

## Explainable AI (SHAP)

Provides:

- Global SHAP feature importance
- Alert-level explanations
- Baseline vs observed comparisons
- Human-readable alert reasoning

### Example Alert Explanation

```text
HIGH ALERT — GKO0078

USB Activity:
Observed: 8
Baseline: 1.9
Deviation: 4.1×

After-hours Activity:
Observed: 94%
Baseline: 11%

Reason:
User activity significantly exceeded personal behavior baseline.
```

---

## Analyst Feedback System

Analysts can:

- Confirm real threats
- Mark false positives
- Flag alerts for investigation

### Adaptive Threshold Logic

```text
1 False Positive → +5%

2 False Positives → +10%

3 False Positives → +15%
```

This reduces alert fatigue and continuously improves detection quality.

---

# Dataset

## Dataset Used

**CERT Insider Threat Dataset R4.2**

The CERT dataset provides realistic enterprise activity logs and insider threat scenarios.

### Log Sources Used

- Logon activity
- Device activity
- Email activity
- File access activity
- HTTP browsing logs
- User session data

### Dataset Time Period

2010–2011

---

# Feature Engineering

The project uses engineered features rather than directly feeding raw logs into machine learning models.

## Temporal Features

- First login hour
- Last login hour
- Day of week
- Time since previous activity
- After-hours activity ratio

## Behavioral Features

- Device event count
- Email count
- External email count
- File access count
- URL visit count
- Activity entropy
- Session event count

## Statistical Features

- Rolling user averages
- Historical user baselines
- Z-score deviations
- Composite risk scores

---

# System Architecture

```text
CERT Dataset
      ↓

Log Processing
      ↓

Session Construction
      ↓

Feature Engineering
      ↓

Behavior Baseline Modeling
      ↓

Autoencoder
Isolation Forest
LSTM
      ↓

Ensemble Scoring
      ↓

SHAP Explainability Layer
      ↓

FastAPI Backend
      ↓

Streamlit Dashboard
      ↓

Analyst Feedback System
      ↓

Adaptive Threshold Learning
```

---

# Technology Stack

## Backend

- FastAPI
- Uvicorn

## Frontend

- Streamlit

## Machine Learning

- PyTorch
- Scikit-learn
- SHAP
- NumPy
- Pandas

## Visualization

- Plotly
- Matplotlib

## Deployment

- Render
- Streamlit Cloud

## Development Tools

- Git
- GitHub
- VS Code

---

# Dashboard Modules

## Overview

Displays:

- Total alerts
- Risk summaries
- Alert statistics
- User insights

---

## Alert Center

Displays:

- Active alerts
- Alert severity
- Filtering options

---

## User Investigation

Displays:

- User timelines
- Historical behavior
- Activity analysis

---

## Explainability Center

Displays:

- SHAP feature importance
- Alert reasoning
- Baseline comparisons

---

## Model Report

Displays:

- Model comparison
- ROC curves
- Performance metrics

---

## Analyst Feedback

Displays:

- Feedback statistics
- User threshold adjustments
- False positive tracking

---

# Project Structure

```text
AI-Driven-Insider-Threat-Detection-System/

├── dashboard/
│   ├── views/
│   └── app.py
│
├── src/
│   ├── api/
│   ├── explainability/
│   ├── feedback/
│   └── models/
│
├── data/
├── deployment_data/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── docs/
│   └── demo/
│
├── requirements.txt
├── render.yaml
├── README.md
└── .gitignore
```

---

# Model Evaluation

| Model | Purpose |
|---------|----------|
| Autoencoder | Reconstruction-based anomaly detection |
| Isolation Forest | Density-based outlier detection |
| Bidirectional LSTM | Temporal sequence anomaly detection |
| Ensemble | Combined anomaly scoring |

### Evaluation Metrics

- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision
- False Positive Rate

---

# API Endpoints

## Alerts

```http
GET /alerts
GET /alerts/count
GET /alerts/{alert_id}
```

## Users

```http
GET /users/top-risk
GET /users/{user_id}/timeline
GET /users/{user_id}/summary
```

## Statistics

```http
GET /stats/overview
```

## Feedback

```http
POST /feedback/false-positive
POST /feedback/confirm
GET /feedback/stats
GET /feedback/user/{user_id}
```

---

# Run Locally

## Clone Repository

```bash
git clone https://github.com/Suryakoundinya21/AI-Driven-Insider-Threat-Detection-System.git

cd AI-Driven-Insider-Threat-Detection-System
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Backend API

```bash
uvicorn src.api.main:app --reload --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard URL:

```text
http://localhost:8501
```

---

# Live Deployment

## Backend API

```text
https://ai-driven-insider-threat-detection-system.onrender.com
```

## Dashboard

```text
https://ai-driven-insider-threat-detection-system-wesxb3dpmfgtmsqlxsr3.streamlit.app
```

---

# Research Contribution

Novel contributions include:

- Per-user behavioral baseline modeling
- Multi-model ensemble anomaly detection
- Explainable AI integration
- Human-readable alert reasoning
- Adaptive threshold learning from analyst feedback

---

# Future Improvements

- Real-time streaming log ingestion
- Transformer-based sequence models
- Graph Neural Networks
- SIEM integration
- Multi-tenant architecture
- Federated learning support

---

# Author

**Surya Koundinya C**

AI-Driven Insider Threat Detection System  
Using Behavioral Analytics and Explainable AI

---
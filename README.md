# AI-Driven Insider Threat Detection System

AI-driven insider threat detection system using behavioral anomaly detection, explainable AI, and adaptive analyst feedback mechanisms for enterprise security environments.

---

## Overview

Insider threats remain one of the most difficult cybersecurity challenges because malicious or abnormal activity often originates from legitimate users with valid access permissions. Traditional security systems rely heavily on predefined rules and signatures, which can struggle to identify subtle behavioral deviations.

This project addresses the problem by building a per-user behavioral baseline using machine learning techniques and identifying anomalous user sessions through an ensemble anomaly detection framework.

The system provides:

- Behavioral anomaly detection using multiple models
- Explainable AI using SHAP
- Analyst-driven feedback loop
- Adaptive thresholding
- Interactive dashboard for security analysts
- Cloud deployment for real-world accessibility

---

## Features

### User Behavior Analytics
- Per-user baseline behavior profiling
- Temporal and behavioral feature extraction
- Session-level activity analysis

### Machine Learning Models
- Autoencoder-based anomaly detection
- Isolation Forest anomaly detection
- Bidirectional LSTM sequence modeling
- Ensemble anomaly scoring

### Explainability
- SHAP feature importance visualization
- Alert-level reasoning
- Baseline vs observed comparisons
- Human-readable anomaly explanations

### Analyst Feedback System
- Confirm threats
- Mark false positives
- Flag alerts for investigation
- Adaptive threshold adjustment

### Dashboard
- Alert center
- User investigation timeline
- Explainability center
- Model performance report
- Analyst feedback page

### Deployment
- Backend API hosted on Render
- Dashboard deployed on Streamlit Cloud

---

## Dataset

### Dataset Used

**CERT Insider Threat Dataset r4.2**

The CERT dataset provides realistic synthetic enterprise activity logs containing insider threat scenarios.

### Log Sources Used

- Logon events
- Device activity logs
- Email activity
- File access activity
- HTTP browsing logs
- Session information

### Dataset Period

- User activity between **2010–2011**

---

## Feature Engineering

The project uses engineered behavioral features rather than directly feeding raw logs into machine learning models.

### Temporal Features

- First login hour
- Last login hour
- Day of week
- Time since previous activity
- After-hours activity ratio

### Behavioral Features

- Device event count
- Email count
- External email count
- File access count
- URL visit count
- Activity entropy
- Session event count

### Statistical Features

- Per-user historical averages
- Rolling baseline statistics
- Z-score deviations
- Composite risk score

---

## System Architecture

```text
CERT Dataset
      ↓
Log Processing
      ↓
Session Construction
      ↓
Feature Engineering
      ↓
Behavioral Baseline Creation
      ↓
Autoencoder + Isolation Forest + LSTM
      ↓
Ensemble Scoring
      ↓
SHAP Explainability Layer
      ↓
FastAPI Backend
      ↓
Streamlit Dashboard
      ↓
Analyst Feedback
      ↓
Adaptive Threshold Calibration
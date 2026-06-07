# Heart Disease Prediction Pipeline (MLOps Lifecycle)

This repository contains an end-to-end Machine Learning Operations (MLOps) pipeline for predicting heart disease using the UCI clinical dataset. It tracks baseline metrics, executes automated hyperparameter tuning sweeps, manages model versions, provides feature explainability, and monitors performance drops under simulated data drift.

---

## 🛑 Step 0: Start the MLflow Tracking Server

Before running any of the pipeline scripts, you must boot up the MLflow background tracking server. This initialization command sets up the SQLite backend database hub (`mlflow.db`) and allocates the local artifact folder structure (`./mlruns`).

Run this command in your terminal window:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

Once running, you can access the interactive dashboard interface at any time here:
👉 **MLflow UI Dashboard:** [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📋 File Execution Order Guide

Keep your MLflow server running in its own terminal window, open a second terminal window, and execute the Python pipeline files sequentially in this exact order:

### Step 1: Clean and Prepare the Data
```bash
python data_prep.py
```
* **What it does:** Ingests the raw clinical dataset, fixes formatting anomalies, handles missing indicators, and saves the file `heart_disease_cleaned.csv` to act as our unified data source.

### Step 2: Establish the Baseline Benchmark
```bash
python train.py
```
* **What it does:** Trains an out-of-the-box, untuned baseline Random Forest classifier to act as our control group. It automatically registers it as Version 2 of `HeartDiseaseClassifier` and programmatically tags it with the `"baseline"` registry alias.

### Step 3: Run Multi-Model Parameter Tuning Sweep
```bash
python tune.py
```
* **What it does:** Triggers an automated 40-iteration search using Hyperopt. It acts as an algorithmic showdown, testing complex tree distributions for both **Random Forest** and **XGBoost** to discover which configuration tracks the highest validation accuracy.

### Step 4: Assemble & Register the Champion Model
```bash
python register.py
```
* **What it does:** Connects to the tracking server, finds the winning run from Step 3, reads its optimal configuration, and trains the final champion asset. It maps human-readable medical feature labels, attaches a high-resolution **SHAP beeswarm chart** to the metrics dashboard, adds the asset to the Model Registry, and programmatically moves the `"production"` tag over to it.

### Step 5: Execute Comparative Drift Monitoring
```bash
python monitor.py
```
* **What it does:** Loads both our operational registered models (`baseline` and `production`) out of the registry and evaluates them side-by-side against an identical patient matrix corrupted with synthetic sensor noise. It prints a **Degradation Matrix** directly to the terminal to check for structural overfitting.

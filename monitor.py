import pandas as pd
import numpy as np
import logging
import warnings
import mlflow
from sklearn.metrics import accuracy_score, precision_score
from mlflow.client import MlflowClient

# Silence serialization and model loading noise
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)


def simulate_comparative_drift():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Heart_Disease_Monitoring")
    client = MlflowClient()

    print("Querying MLflow Registry for both operational models...")

    # 1. Fetch Baseline Model Data
    base_version = client.get_model_version_by_alias("HeartDiseaseClassifier", "baseline")
    base_run = client.get_run(base_version.run_id)
    base_orig_acc = base_run.data.metrics.get("accuracy", 0.0)
    base_orig_prec = base_run.data.metrics.get("precision", 0.0)

    # 2. Fetch Champion Model Data
    champ_version = client.get_model_version_by_alias("HeartDiseaseChampion", "production")
    champ_run = client.get_run(champ_version.run_id)
    champ_orig_acc = champ_run.data.metrics.get("accuracy", 0.0)
    champ_orig_prec = champ_run.data.metrics.get("precision", 0.0)

    print("Downloading architecture assets into memory...")
    base_model = mlflow.sklearn.load_model("models:/HeartDiseaseClassifier@baseline")
    champ_model = mlflow.sklearn.load_model("models:/HeartDiseaseChampion@production")

    print("Loading evaluation dataset and generating hardware noise...")
    df = pd.read_csv("heart_disease_cleaned.csv")

    # Matrix 1: For the Baseline Model
    X_base = df.iloc[:, :-1]
    y = (df.iloc[:, -1] > 0).astype(int)

    # Generate identical noise for a fair comparison
    rng = np.random.default_rng(seed=67)
    noise = rng.normal(loc=0.0, scale=2.0, size=X_base.shape)
    X_base_drifted = X_base + noise

    # Matrix 2: For the Champion Model (Translated Names)
    feature_rename_map = {
        'age': 'Age (Years)', 'sex': 'Biological Sex', 'cp': 'Chest Pain Severity',
        'trestbps': 'Resting Blood Pressure', 'chol': 'Serum Cholesterol',
        'fbs': 'Fasting Blood Sugar > 120mg/dl', 'restecg': 'Resting ECG Results',
        'thalach': 'Max Heart Rate Achieved', 'exang': 'Exercise-Induced Angina',
        'oldpeak': 'ST Depression (Exercise Metric)', 'slope': 'Peak ST Segment Slope',
        'ca': 'Number of Major Blood Vessels', 'thal': 'Thalassemia Type'
    }
    X_champ_drifted = X_base_drifted.rename(columns=feature_rename_map)

    print("Evaluating models against drifted clinical data...")
    with mlflow.start_run(run_name="Comparative_Drift_Analysis"):
        # Predict using Baseline
        base_preds = base_model.predict(X_base_drifted)
        base_drift_acc = accuracy_score(y, base_preds)
        base_drift_prec = precision_score(y, base_preds, zero_division=0)

        # Predict using Champion
        champ_preds = champ_model.predict(X_champ_drifted)
        champ_drift_acc = accuracy_score(y, champ_preds)
        champ_drift_prec = precision_score(y, champ_preds, zero_division=0)

        # Log metrics dynamically
        mlflow.log_metrics({
            "base_drifted_acc": base_drift_acc,
            "base_drifted_prec": base_drift_prec,
            "champ_drifted_acc": champ_drift_acc,
            "champ_drifted_prec": champ_drift_prec
        })

        # Format handling for un-tracked baseline precision
        base_prec_str = f"{base_orig_prec:.4f}" if base_orig_prec > 0 else "N/A"
        base_prec_drop = f"{(base_orig_prec - base_drift_prec):.4f}" if base_orig_prec > 0 else "N/A"

        # Print the comparative UI to the terminal
        print("\n=========================================================================")
        print("📊 MLOPS DEGRADATION MATRIX (NOISE VARIANCE: 2.0)")
        print("=========================================================================")
        print(f"{'MODEL TYPE':<20} | {'ORIGINAL ACC':<14} | {'DRIFTED ACC':<13} | {'TOTAL DROP'}")
        print("-" * 73)
        print(
            f"{'Untuned Baseline':<20} | {base_orig_acc:<14.4f} | {base_drift_acc:<13.4f} | {(base_orig_acc - base_drift_acc):.4f}")
        print(
            f"{'Tuned Champion':<20} | {champ_orig_acc:<14.4f} | {champ_drift_acc:<13.4f} | {(champ_orig_acc - champ_drift_acc):.4f}")
        print("\n" + "-" * 73)
        print(f"{'MODEL TYPE':<20} | {'ORIGINAL PREC':<14} | {'DRIFTED PREC':<13} | {'TOTAL DROP'}")
        print("-" * 73)
        print(f"{'Untuned Baseline':<20} | {base_prec_str:<14} | {base_drift_prec:<13.4f} | {base_prec_drop}")
        print(
            f"{'Tuned Champion':<20} | {champ_orig_prec:<14.4f} | {champ_drift_prec:<13.4f} | {(champ_orig_prec - champ_drift_prec):.4f}")
        print("=========================================================================\n")


if __name__ == "__main__":
    simulate_comparative_drift()
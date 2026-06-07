import pandas as pd
import logging
import warnings
import mlflow
import mlflow.sklearn
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from mlflow.client import MlflowClient

# Mute backend serialization noise
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)


def register_champion_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Heart_Disease_Prediction")

    print("Querying MLflow tracking server for the active session champion...")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Heart_Disease_Prediction")

    # Find the absolute newest parent tuning session execution window
    latest_parent_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'Advanced_Showdown_Tuning'",
        order_by=["attributes.start_time DESC"],  # Newest first
        max_results=1
    )

    if not latest_parent_runs:
        print("No advanced tuning sessions found.")
        return

    latest_parent_id = latest_parent_runs[0].info.run_id

    # Scan only children within this specific session parameters
    best_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{latest_parent_id}'",
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not best_runs:
        print("No valid child runs located.")
        return

    best_run = best_runs[0]
    best_params = best_run.data.params
    best_accuracy = best_run.data.metrics["accuracy"]
    best_precision = best_run.data.metrics.get("precision", 0.0)

    model_type = best_params.get("model_type")
    print(f"\nFound Target Champion! Architecture: {model_type} | Accuracy: {best_accuracy:.4f}")

    # Load and map human-readable clinical data frame
    df = pd.read_csv("heart_disease_cleaned.csv")
    X = df.iloc[:, :-1]
    y = (df.iloc[:, -1] > 0).astype(int)

    feature_rename_map = {
        'age': 'Age (Years)', 'sex': 'Biological Sex', 'cp': 'Chest Pain Severity',
        'trestbps': 'Resting Blood Pressure', 'chol': 'Serum Cholesterol',
        'fbs': 'Fasting Blood Sugar > 120mg/dl', 'restecg': 'Resting ECG Results',
        'thalach': 'Max Heart Rate Achieved', 'exang': 'Exercise-Induced Angina',
        'oldpeak': 'ST Depression (Exercise Metric)', 'slope': 'Peak ST Segment Slope',
        'ca': 'Number of Major Blood Vessels', 'thal': 'Thalassemia Type'
    }
    X = X.rename(columns=feature_rename_map)

    # Dynamically build the configuration tracking vectors
    if model_type == "RandomForest":
        n_estimators = int(float(best_params["n_estimators"]))
        max_depth = int(float(best_params["max_depth"]))
        min_samples_split = int(float(best_params["min_samples_split"]))
        min_samples_leaf = int(float(best_params["min_samples_leaf"]))

        final_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    elif model_type == "XGBoost":
        n_estimators = int(float(best_params["n_estimators"]))
        max_depth = int(float(best_params["max_depth"]))
        learning_rate = float(best_params["learning_rate"])
        subsample = float(best_params["subsample"])
        colsample_bytree = float(best_params["colsample_bytree"])

        final_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42
        )

    print("Fitting final production asset layer...")
    final_model.fit(X, y)

    # Package and update Registry details
    with mlflow.start_run(run_name=f"Final_Production_{model_type}"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", best_accuracy)
        mlflow.log_metric("precision", best_precision)

        # Build clean SHAP summary
        print("Generating standard SHAP beeswarm summary graphics...")
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            correct_shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            correct_shap_values = shap_values[:, :, 1]
        else:
            correct_shap_values = shap_values

        plt.clf()
        shap.summary_plot(correct_shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png", bbox_inches='tight', dpi=300)
        plt.close()

        mlflow.log_artifact("shap_summary.png")

        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            registered_model_name="HeartDiseaseChampion"
        )

        # Inject detailed Markdown tracking notes directly into the UI dashboard
        description_markdown = f"""
        ### Heart Disease Champion Classifier
        * **Architecture Type:** {model_type}
        * **Tuning Validation Accuracy:** {best_accuracy:.4f}
        * **Tuning Validation Precision:** {best_precision:.4f}
        * **Operational Status:** Production-Ready

        **Model Parameters:** {chr(10).join([f'* **{k}:** {v}' for k, v in best_params.items() if k != 'model_type'])}
        """

        client.update_model_version(
            name="HeartDiseaseChampion",
            version=model_info.registered_model_version,
            description=description_markdown
        )

    print("\nModel pipeline execution complete. System successfully updated!")


if __name__ == "__main__":
    register_champion_model()
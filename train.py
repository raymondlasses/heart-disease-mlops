import pandas as pd
import warnings
import logging
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from mlflow.client import MlflowClient

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)


def train_baseline():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Heart_Disease_Prediction")

    print("Loading baseline clinical dataset...")
    df = pd.read_csv("heart_disease_cleaned.csv")

    # We leave the names raw here because the baseline model was originally trained on raw names
    X = df.iloc[:, :-1]
    y = (df.iloc[:, -1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training out-of-the-box Baseline Random Forest...")
    with mlflow.start_run(run_name="Baseline_Model_V2"):
        # 1. Train untuned model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # 2. Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)

        # 3. Log metrics specifically so monitor.py can find them
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)

        # 4. Register to the original baseline catalog
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="HeartDiseaseClassifier"
        )

        # 5. Automatically move the "baseline" alias to this new V2 model
        client = MlflowClient()
        client.set_registered_model_alias(
            name="HeartDiseaseClassifier",
            alias="baseline",
            version=model_info.registered_model_version
        )

    print(f"   Baseline successfully registered!")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")


if __name__ == "__main__":
    train_baseline()
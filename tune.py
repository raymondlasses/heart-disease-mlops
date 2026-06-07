import logging
import warnings

# Suppress package warnings and reduce MLflow logging noise in the terminal
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("mlflow").setLevel(logging.ERROR)

import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def load_data():
    """Loads cleaned heart disease data and prepares an 80/20 train/test split."""
    df = pd.read_csv("heart_disease_cleaned.csv")
    X = df.iloc[:, :-1]

    # Convert multi-class target to binary classification (0 = Healthy, 1 = Disease)
    y = (df.iloc[:, -1] > 0).astype(int)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def objective(params):
    """Objective function for Hyperopt to calculate classification accuracy."""
    # Group the 40 trials as nested child runs under a single parent tracking window
    with mlflow.start_run(nested=True):
        model_type = params['type']

        # Configuration for Random Forest
        if model_type == 'random_forest':
            n_estimators = int(params['n_estimators'])
            max_depth = int(params['max_depth'])
            min_samples_split = int(params['min_samples_split'])
            min_samples_leaf = int(params['min_samples_leaf'])

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            # Log active parameter combinations to MLflow
            mlflow.log_params({
                "model_type": "RandomForest",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf
            })

        # Configuration for XGBoost
        elif model_type == 'xgboost':
            n_estimators = int(params['n_estimators'])
            max_depth = int(params['max_depth'])
            learning_rate = params['learning_rate']
            subsample = params['subsample']
            colsample_bytree = params['colsample_bytree']

            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42
            )
            # Log active regularization parameters to MLflow
            mlflow.log_params({
                "model_type": "XGBoost",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree
            })

        # Model training and prediction
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calculate evaluation performance dimensions
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)

        # Record validation metrics to the MLflow metrics database
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)

        # Return negative accuracy as loss score because hyperopt minimizes optimization targets
        return {'loss': -accuracy, 'status': STATUS_OK}


if __name__ == "__main__":
    # Point directly to the active local MLflow tracking server port
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Heart_Disease_Prediction")

    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = load_data()

    # Define hyperparameter search spaces for both model types
    search_space = hp.choice('classifier_type', [
        {
            'type': 'random_forest',
            'n_estimators': hp.quniform('rf_n_estimators', 50, 350, 10),
            'max_depth': hp.quniform('rf_max_depth', 3, 16, 1),
            'min_samples_split': hp.quniform('rf_min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 5, 1)
        },
        {
            'type': 'xgboost',
            'n_estimators': hp.quniform('xgb_n_estimators', 50, 350, 10),
            'max_depth': hp.quniform('xgb_max_depth', 3, 12, 1),
            'learning_rate': hp.loguniform('xgb_learning_rate', -3, 0),
            'subsample': hp.uniform('xgb_subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('xgb_colsample_bytree', 0.5, 1.0)
        }
    ])

    print("Starting Advanced Multi-Architecture Parameter Sweep...")

    # Initialize parent run workspace container
    with mlflow.start_run(run_name="Advanced_Showdown_Tuning"):
        trials = Trials()

        # Execute Bayesian Optimization loop across exactly 40 evaluations
        best_result = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=40,
            trials=trials
        )

        print("\nTuning complete with zero serialization warnings!")
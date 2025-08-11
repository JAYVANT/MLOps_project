# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
import logging

# Setup logger
log = logging.getLogger("train")
log.setLevel(logging.INFO)
handler = logging.FileHandler("train_log.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

def eval_metrics(actual, pred):
    """Calculates and returns model evaluation metrics."""
    log.info("Executing eval_metrics")
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    log.info(f"Metrics calculated: RMSE={rmse}, R2={r2}")
    return rmse, r2

def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test):
    """Trains a model and logs its parameters, metrics, and artifact with MLflow."""
    log.info(f"Executing train_and_log_model for {model_name}")
    with mlflow.start_run(run_name=model_name) as run:
        log.info(f"Started MLflow run: {run.info.run_id}")
        print(f"--- Training {model_name} ---")

        # Train model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        (rmse, r2) = eval_metrics(y_test, predictions)

        # Log parameters (for Decision Tree)
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        print(f"  RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
        log.info(f"Logged metrics for {model_name}: RMSE={rmse}, R2={r2}")
        # Log the model artifact
        mlflow.sklearn.log_model(model, "model")
        log.info(f"Logged model artifact for {model_name}")
        return run.info.run_id

if __name__ == "__main__":
    # Use a relative path for the tracking URI to ensure portability
    log.info("train.py main execution started")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("California Housing Prediction")

    # Load and split data
    df = pd.read_csv("data/raw/housing.csv")
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models and get their run IDs
    lr_run_id = train_and_log_model('LinearRegression', LinearRegression(), X_train, y_train, X_test, y_test)
    dt_run_id = train_and_log_model('DecisionTree', DecisionTreeRegressor(random_state=42), X_train, y_train, X_test, y_test)

    # --- Select and Register the Best Model ---
    client = mlflow.tracking.MlflowClient()
    lr_r2 = client.get_metric_history(lr_run_id, "r2")[0].value
    dt_r2 = client.get_metric_history(dt_run_id, "r2")[0].value

    best_run_id = dt_run_id if dt_r2 > lr_r2 else lr_run_id
    best_model_name = "DecisionTree" if dt_r2 > lr_r2 else "LinearRegression"
    log.info(f"Best model selected: {best_model_name} (run_id={best_run_id})")
    print(f"\nBest model is '{best_model_name}' with R2 score.")

    # Register the best model
    model_uri = f"runs:/{best_run_id}/model"
    model_name = "california-housing-regressor"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
    log.info(f"Model '{model_name}' registered as Version {registered_model.version}")
    print(f"Model '{model_name}' has been registered as Version {registered_model.version}.")
    log.info("train.py main execution finished")
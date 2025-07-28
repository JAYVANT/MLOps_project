# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np

def eval_metrics(actual, pred):
    """Calculates and returns model evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    return rmse, r2

def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test):
    """Trains a model and logs its parameters, metrics, and artifact with MLflow."""
    with mlflow.start_run(run_name=model_name) as run:
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

        # Log the model artifact
        mlflow.sklearn.log_model(model, "model")

        return run.info.run_id

if __name__ == "__main__":
    # Use a relative path for the tracking URI to ensure portability
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

    print(f"\nBest model is '{best_model_name}' with R2 score.")

    # Register the best model
    model_uri = f"runs:/{best_run_id}/model"
    model_name = "california-housing-regressor"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"Model '{model_name}' has been registered as Version {registered_model.version}.")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
import os

def eval_metrics(actual, pred):
    """Calculates and returns model evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_model(model_name, model, X_train, y_train, X_test, y_test):
    """Trains a model and logs it with MLflow."""
    # Start a new MLflow run
    with mlflow.start_run(run_name=model_name):
        print(f"--- Training {model_name} ---")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        (rmse, mae, r2) = eval_metrics(y_test, predictions)

        # Log model parameters
        # For Decision Tree, we can log its parameters like max_depth
        if model_name == 'DecisionTree':
            params = model.get_params()
            mlflow.log_params(params)

        # Log metrics
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2 Score: {r2}")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log the trained model as an artifact
        mlflow.sklearn.log_model(model, "model")
        # Register the model in the Model Registry
        model_uri = mlflow.get_artifact_uri("model")
        mlflow.register_model(model_uri, "california-housing-regressor")
        print(f"--- {model_name} Run Finished ---")


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")   # Set the MLflow tracking URI") 
    # Load the dataset
    try:
        df = pd.read_csv("data/raw/housing.csv")
    except FileNotFoundError:
        print("Error: housing.csv not found. Please run src/preprocess.py first.")
        exit()

    # Define features (X) and target (y)
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set the MLflow experiment
    # If the experiment does not exist, MLflow creates it.
    mlflow.set_experiment("California Housing Prediction")

    # Train Linear Regression
    lr_model = LinearRegression()
    train_model('LinearRegression', lr_model, X_train, y_train, X_test, y_test)

    # Train Decision Tree Regressor
    # We can try a specific hyperparameter, e.g., max_depth
    dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
    train_model('DecisionTree', dt_model, X_train, y_train, X_test, y_test)

    print("\nAll models trained and logged to MLflow.")
    print("Run 'mlflow ui' in your terminal to see the results.")
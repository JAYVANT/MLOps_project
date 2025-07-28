import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Define the path to the MLflow tracking server.
# Since we are running locally, it's the 'mlruns' directory.
# In a production setup, this would be a remote URI (e.g., http://, postgresql://).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Define the model name and stage we want to load
MODEL_NAME =  "california-housing-regressor"  # This should match the name used in MLflow
MODEL_VERSION = "1" # This is the version of the model we want to load
MODEL_STAGE = "None" # We didn't assign a stage, so we use 'None'

# Create the FastAPI app object
app = FastAPI(
    title="California Housing Price Prediction API",
    description="An API to predict housing prices in California.",
    version="0.1.0"
)

# Load the model from the MLflow Model Registry
try:
    print(f"Loading model '{MODEL_NAME}' version {MODEL_VERSION} from stage '{MODEL_STAGE}'...")
    # The model URI format is "models:/<model_name>/<stage_or_version>"
    # model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model_uri = "mlruns/<895107721900356651>/<2c894360b0f14aadb767146c9cdb4236>/artifacts/model"
    model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
except Exception as e:
    # If the model fails to load, we raise an error to prevent the app from starting.
    # This is a critical failure.
    print(f"Error loading model: {e}")
    model = None # Set model to None to indicate failure
    # In a real app, you might want to exit or have more robust error handling.


# Define the input data schema using Pydantic
# These are the features our model expects.
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

    class Config:
        # Provides an example for the auto-generated documentation
        schema_extra = {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.98412698,
                "AveBedrms": 1.02380952,
                "Population": 322.0,
                "AveOccup": 2.55555556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }


@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the ML Model Prediction API!"}


@app.post("/predict/")
def predict_price(features: HouseFeatures):
    """
    Accepts housing features in a POST request and returns the predicted price.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    try:
        # Convert the input data to a pandas DataFrame
        # The model expects a DataFrame with specific column names.
        input_df = pd.DataFrame([features.dict()])

        # Make a prediction
        prediction = model.predict(input_df)

        # The prediction is usually a numpy array, so we extract the first element.
        predicted_price = prediction[0]

        return {"predicted_median_house_value": predicted_price}

    except Exception as e:
        # Handle any errors during prediction
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
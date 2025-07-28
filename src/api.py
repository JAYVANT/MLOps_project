# src/api.py
import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from pythonjsonlogger import jsonlogger
from prometheus_fastapi_instrumentator import Instrumentator

# --- 1. Setup Logging & Monitoring ---
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logHandler = logging.FileHandler("api_log.log")
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
log.addHandler(logHandler)

# --- 2. Create FastAPI App and Instrument for Prometheus ---
app = FastAPI(title="MLOps Prediction API")
Instrumentator().instrument(app).expose(app)

# --- 3. Load Registered Model from MLflow ---
MODEL_NAME = "california-housing-regressor"
MODEL_VERSION = 1 # We use the version we just registered
model = None
try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.pyfunc.load_model(model_uri)
    log.info(f"Successfully loaded model '{MODEL_NAME}' version {MODEL_VERSION}")
except Exception as e:
    log.error(f"Error loading model: {e}", exc_info=True)

# --- 4. Define API Input Schema ---
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# --- 5. Define Prediction Endpoint ---
@app.post("/predict/")
def predict_price(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        log.info("Received prediction request", extra={'input': features.dict()})
        input_df = pd.DataFrame([features.dict()])
        prediction = model.predict(input_df)[0]
        log.info("Prediction successful", extra={'prediction': prediction})
        return {"predicted_median_house_value": prediction}
    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")
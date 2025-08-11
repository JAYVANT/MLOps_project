# src/api.py
import os
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from pythonjsonlogger import jsonlogger
from prometheus_fastapi_instrumentator import Instrumentator
import time
from prometheus_client import Counter, Histogram
 
INFERENCE_SECONDS = Histogram(
    "inference_seconds", "Model inference duration (s)",
    buckets=(0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5)
)
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Prediction requests", ["status", "model_version"]
)
PREDICTION_VALUE = Histogram(
    "prediction_value", "Predicted median house value ($100k units)",
    buckets=(0.1,0.25,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5.0,6.0)
)
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
    # model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    # model_uri = "mlruns/453282261435004634/e5cc911fc51d41b59a5882795832d89f/models/m-b96964a56177484bbdbf9eb2e2262584/artifacts"
    log.info("Executing model loading")
    model_uri= "mlruns/453282261435004634/models/m-b96964a56177484bbdbf9eb2e2262584/artifacts"
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
    log.info("Executing predict_price endpoint")
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
# def predict_price(features: HouseFeatures):
#     if model is None:
#         # model not loaded
#         PREDICTIONS_TOTAL.labels(status="unavailable", model_version=str(MODEL_VERSION)).inc()
#         raise HTTPException(status_code=503, detail="Model not available")
 
#     try:
#         log.info("Received prediction request", extra={'input': features.dict()})
#         input_df = pd.DataFrame([features.dict()])
 
#         # measure inference time
#         t0 = time.perf_counter()
#         prediction = float(model.predict(input_df)[0])
#         duration = time.perf_counter() - t0
#         INFERENCE_SECONDS.observe(duration)
 
#         # record success + value distribution
#         PREDICTIONS_TOTAL.labels(status="ok", model_version=str(MODEL_VERSION)).inc()
#         PREDICTION_VALUE.observe(prediction)
 
#         log.info("Prediction successful", extra={'prediction': prediction, 'latency_s': duration})
#         return {"predicted_median_house_value": prediction}
#     except Exception as e:
#         # record failure
#         PREDICTIONS_TOTAL.labels(status="error", model_version=str(MODEL_VERSION)).inc()
#         log.error(f"Prediction error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Prediction failed")
import requests
import json

# The URL of the API running inside the Docker container
# Note: We use port 8001 because we mapped it from the container's port 8000
URL = "http://localhost:8001/predict/"

# Sample data for prediction, matching the Pydantic model in api.py
# This is an example of a high-value coastal area in California
input_data = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
}

print("🚀 Sending request to API...")
print("Data:", json.dumps(input_data, indent=2))

try:
    # Send a POST request with the JSON data
    response = requests.post(URL, json=input_data)

    # Raise an exception if the request was unsuccessful (e.g., 4xx or 5xx errors)
    response.raise_for_status() 

    print("\n✅ Success!")
    print("Status Code:", response.status_code)
    print("Prediction:", response.json())

except requests.exceptions.RequestException as e:
    print(f"\n❌ An error occurred: {e}")
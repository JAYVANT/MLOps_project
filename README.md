# End-to-End MLOps Pipeline for California Housing Prediction

This repository contains a complete, end-to-end MLOps pipeline for training, versioning, containerizing, deploying, and monitoring a regression model for the California Housing dataset.

# Architectural Summary
This project demonstrates a modern MLOps workflow, taking a model from experimentation to an automated, observable production service.

# Architectural Flow
Developer         GitHub & DVC          GitHub Actions (CI/CD)      Docker Hub           Production Server
+---------+       +-------------+       +------------------------+  +------------+       +-------------------+
|         |       |             |       |                        |  |            |       |                   |
| git push|------>| Code & Data |------>| 1. Test & Lint (CI)    |  |            |       | ./deploy.sh pulls |
|         |       | Versioning  |       | 2. Build Docker Image  |->| Image      |<------| & runs container  |
+---------+       +-------------+       | 3. Push to Registry(CD)|  | Registry   |       |                   |
                                        +------------------------+  +------------+       +-------------------+
                                                                                             |     ^
     ^                                                                                       |     |
     |                                                                                       |     |
     |    MLflow (Tracks Experiments & Registers Models)                                     |  Prometheus
     +---------------------------------------------------------------------------------------+  (Scrapes
                                                                                                /metrics)

# Stages of the Pipeline
1. Foundation & Experimentation (Reproducibility)
Code Versioning (Git): All source code is tracked using Git.

Data Versioning (DVC): The raw dataset is versioned with DVC, keeping the Git repository lightweight while ensuring data reproducibility.

Experiment Tracking (MLflow): Every model training run logs parameters, metrics, and model artifacts to MLflow.

Model Registry (MLflow): The best-performing model is programmatically identified and registered in the MLflow Model Registry, giving it an official version for production use.

2. Application Packaging & Observability (Production Readiness)

API Service (FastAPI): A robust API is built to serve predictions from the registered model.

Containerization (Docker): The API, model artifacts, and all dependencies are packaged into a self-contained Docker image for ultimate portability.

Logging & Monitoring: The API is instrumented with structured JSON logging for every request and a /metrics endpoint for real-time monitoring with tools like Prometheus.

3. CI/CD & Deployment (Automation)

Automation (GitHub Actions): A CI/CD pipeline automatically tests, lints, builds, and pushes the Docker image to a registry (Docker Hub) on every git push to the main branch.

Deployment: A simple deployment script pulls the latest container image from the registry and runs it on a target server, completing the automated lifecycle.

# Step-by-Step Instructions to Build and Deploy
Follow these commands to replicate the entire pipeline on your local machine.

Part 0: Prerequisites
1. Clone the Repository:

git clone <your-repository-url>
cd <repository-name>

2. Install Tools: Ensure you have Python 3.9+, Git, and Docker Desktop installed and running.

3. Create a Virtual Environment:

# Create the environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

4. Install Dependencies:

pip install --upgrade pip
pip install -r requirements.txt

5. Part 1: Data Versioning
Fetch and Preprocess Data: This script downloads the dataset and saves it.

python src/preprocess.py

6. Initialize DVC and Track Data:

# Initialize DVC in the repository
dvc init

# Add the raw data file to DVC tracking
dvc add data/raw/housing.csv

# Commit the DVC pointer file to Git
git add data/raw/housing.csv.dvc .gitignore
git commit -m "feat: track raw data with DVC"

7. Part 2: Model Training & Registration
Run the Training Script: This will train two models, compare them, and register the best one in MLflow.

#### python src/train.py

View the Experiments (Optional):

#### mlflow ui

Open http://127.0.0.1:5000 in your browser. Go to the Models tab to see your registered california-housing-regressor.

8. Part 3 & 5: API, Docker, and Monitoring
#### Build the Docker Image:

docker build -t housing-api .

#### Run the Docker Container:

docker run -d -p 8001:8000 --name housing-predictor housing-api

#### Test the Running Service:

API Prediction:

curl -X 'POST' \
  'http://localhost:8001/predict/' \
  -H 'Content-Type: application/json' \
  -d '{"MedInc": 8.3, "HouseAge": 41, "AveRooms": 7, "AveBedrms": 1, "Population": 322, "AveOccup": 2.5, "Latitude": 37.88, "Longitude": -122.23}'

#### Check Logs:

docker exec housing-predictor cat api_log.log

View Metrics: Open http://localhost:8001/metrics in your browser.

9. Part 4: CI/CD Setup and Deployment
#### Set Up GitHub & Docker Hub:

Create a public repository on Docker Hub.

In your GitHub repository, go to Settings > Secrets and variables > Actions.

Create two repository secrets: DOCKER_USERNAME (your Docker Hub username) and DOCKER_PASSWORD (your Docker Hub password or access token).

Trigger the CI/CD Pipeline: Commit all your code and push it to GitHub. This will automatically trigger the workflow defined in .github/workflows/.

git add .
git commit -m "feat: complete initial pipeline setup"
git push origin main

Go to the Actions tab on your GitHub repository to watch the pipeline run.

Deploy the Latest Version: Once the pipeline succeeds, run the deployment script to pull the new image from Docker Hub and run it.

Important: Edit deploy.sh and replace "your-dockerhub-username" with your actual username.

Make the script executable:

chmod +x deploy.sh

Run the deployment:

./deploy.sh

Your updated service is now running and available at http://localhost:8001.
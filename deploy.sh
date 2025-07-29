#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# CHANGE THIS to your Docker Hub username
DOCKER_USERNAME="jayvantsolomon"
IMAGE_NAME="california-housing-api"
CONTAINER_NAME="housing-predictor"

echo "--- Deploying new version of $IMAGE_NAME ---"

# Stop and remove existing container to avoid conflicts
echo "1. Stopping and removing old container..."
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

# Pull the latest version of the image from the registry
echo "2. Pulling latest image from Docker Hub..."
docker pull $DOCKER_USERNAME/$IMAGE_NAME:latest

# Run the new container
echo "3. Starting new container..."
docker run -d -p 8001:8000 --name $CONTAINER_NAME $DOCKER_USERNAME/$IMAGE_NAME:latest

echo "✅ Deployment complete. Service is running."
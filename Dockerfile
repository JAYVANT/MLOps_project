# Stage 1: Use an official Python runtime as a parent image
# We use a specific version for reproducibility.
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
# We copy and install requirements first to leverage Docker's layer caching.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Reduces the image size by not storing the cache
# --upgrade pip: Ensures we have the latest pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "fastapi[all]"

# Copy the rest of the application's code into the container
# This includes our 'src' directory and the 'mlruns' directory.
# We need 'mlruns' so the API can load the model from it.
COPY src/ ./src/
COPY mlruns/ ./mlruns/

# Expose port 8000 to the outside world. This is the port our app will run on.
EXPOSE 8000

# Define the command to run the application.
# This command is executed when the container starts.
# We use 0.0.0.0 as the host to make it accessible from outside the container.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

def get_data():
    """Fetches the California Housing dataset and saves it to a CSV file."""
    print("Fetching dataset...")
    # Fetch the dataset
    housing = fetch_california_housing(as_frame=True)

    # The data is in a Bunch object, we'll use the frame attribute which is a pandas DataFrame
    df = housing.frame

    print("Dataset fetched successfully.")

    # Define the path to save the raw data
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "housing.csv")

    # Save the dataframe to a CSV file
    df.to_csv(output_path, index=False)

    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    get_data()
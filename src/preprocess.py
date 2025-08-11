import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
# This script fetches the California Housing dataset and saves it to a CSV file.

# def get_data():
#     """Fetches the California Housing dataset and saves it to a CSV file."""
#     print("Fetching dataset...")
#     # Fetch the dataset
#     housing = fetch_california_housing(as_frame=True)

#     # The data is in a Bunch object, we'll use the frame attribute which is a pandas DataFrame
#     df = housing.frame

#     print("Dataset fetched successfully.")

#     # Define the path to save the raw data
#     output_dir = "data/raw"
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, "housing.csv")

#     # Save the dataframe to a CSV file
#     df.to_csv(output_path, index=False)

#     print(f"Data saved to {output_path}")

# if __name__ == "__main__":
#     get_data()

# This function performs quick validation on the DataFrame and generates a report.
# It checks for basic schema compliance and sensible value ranges, then writes a summary report.

def _quick_validate_and_report(df, out_dir="artifacts/validation/raw"):
    import os, json, datetime, shutil
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
 
    # Basic checks for CA Housing schema & sensible ranges
    checks = {
        "MedInc":      (df["MedInc"] >= 0),
        "HouseAge":    (df["HouseAge"].between(0, 100, inclusive="both")),
        "AveRooms":    (df["AveRooms"] > 0),
        "AveBedrms":   (df["AveBedrms"] > 0),
        "Population":  (df["Population"] >= 0),
        "AveOccup":    (df["AveOccup"] > 0),
        "Latitude":    (df["Latitude"].between(32, 43, inclusive="both")),
        "Longitude":   (df["Longitude"].between(-125, -114, inclusive="both")),
        # target can exist pre-split; require non-negative if present
        **({"MedHouseVal": (df["MedHouseVal"] >= 0)} if "MedHouseVal" in df.columns else {})
    }
 
    null_counts = df.isna().sum().to_dict()
    errors = []
    for col, mask in checks.items():
        bad = int((~mask).sum())
        if bad > 0:
            errors.append({"column": col, "invalid_rows": bad})
 
    status = "passed" if not errors else "failed"
    summary = {
        "status": status,
        "rows": int(len(df)),
        "null_counts": null_counts,
        "n_errors": len(errors),
        "errors": errors,
        "columns": list(df.columns),
    }
 
    # Write machine-readable summary
    sum_path = os.path.join(out_dir, f"{ts}_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
 
    # Write tiny human HTML report
    html_path = os.path.join(out_dir, f"{ts}_report.html")
    with open(html_path, "w") as f:
        f.write(f"<h2>Data Validation: {status}</h2>")
        f.write(f"<p>Rows: {summary['rows']}</p>")
        f.write("<h3>Null counts</h3><pre>")
        f.write(json.dumps(null_counts, indent=2))
        f.write("</pre>")
        if errors:
            f.write("<h3>Errors</h3><pre>")
            f.write(json.dumps(errors, indent=2))
            f.write("</pre>")
 
    # Stable pointers for easy viewing
    try:
        shutil.copyfile(sum_path, os.path.join(out_dir, "summary.json"))
        shutil.copyfile(html_path, os.path.join(out_dir, "latest_report.html"))
    except Exception:
        pass
 
    return {"status": status, "summary_path": sum_path, "html_path": html_path}
 
# This script fetches the California Housing dataset and saves it to a CSV file.
def get_data():
    """Fetches the California Housing dataset and saves it to a CSV file."""
    print("Fetching dataset...")
    # Fetch the dataset
    housing = fetch_california_housing(as_frame=True)
 
    # The data is in a Bunch object, we'll use the frame attribute which is a pandas DataFrame
    df = housing.frame
 
    print("Dataset fetched successfully.")
 
    # Minimal validation (writes artifacts/validation/raw/*)
    val = _quick_validate_and_report(df)
    print(f"Data validation {val['status']}. Report: {val['html_path']}")
 
    # Define the path to save the raw data
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "housing.csv")
 
    # Save the dataframe to a CSV file
    df.to_csv(output_path, index=False)
 
    print(f"Data saved to {output_path}")
 
if __name__ == "__main__":
    get_data()
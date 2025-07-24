import json
import pandas as pd
from pathlib import Path
import numpy as np # Used for np.nan

#############################
### CONFIGURATION ###
#############################

# Base directory for data and figures, relative to the location of this script (COCA_PROJECT/figures).
BASE_PROJECT_DIR = Path("../")

# Paths to the JSON files for each dataset
# Updated paths to reflect the new structure relative to COCA_PROJECT/figures/
JSON_PATHS = {
    "1st_Dataset": BASE_PROJECT_DIR / "analysis" / "CULTIVATED2ND" / "02_trained_models_Cultivated1st" / "metrics" /
"Cultivated1st_ECT_Mask_2Channel_CNN_Ensemble_Improved_classification_report.json",
    "2nd_Dataset": BASE_PROJECT_DIR / "analysis" / "CULTIVATED2ND" / "02_trained_models_Cultivated2nd" / "metrics" / "Cultivated2nd_ECT_Mask_2Channel_CNN_Ensemble_Improved_classification_report.json",
    # CORRECTED: Combined Dataset Path with the right filename
    "Combined_Dataset": BASE_PROJECT_DIR / "analysis" / "CULTIVATED2ND" / "03_trained_models_both_combined" / "metrics" /
"Combined_ECT_Mask_2Channel_CNN_Ensemble_classification_report.json",
}

# Output directory for the performance table
OUTPUT_DIR = Path(".") # Save output in the same directory as the script (COCA_PROJECT/figures/)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure the figures directory exists
OUTPUT_TABLE_FILENAME = "Table2.csv" # Output filename is "Table2.csv"

# Define the 16 shared classes alphabetically
SHARED_CLASSES = [
    'amazona', 'boliviana blanca', 'boliviana roja', 'chiparra', 'chirosa',
    'crespa', 'dulce', 'gigante', 'guayaba roja', 'patirroja',
    'peruana roja', 'tingo maria', 'tingo pajarita', 'tingo pajarita caucana',
    'tingo peruana', 'trujillense caucana'
]

# Define the order of aggregate rows
AGGREGATE_ROWS = ["macro avg", "weighted avg"]

#############################
### GENERATE PERFORMANCE TABLE ###
#############################

print("--- Generating Cultigen Model Performance Table ---")

# Dictionary to hold data for easy DataFrame construction
data_for_df = []

# Load data for each dataset
dataset_reports = {}
for dataset_name, json_path in JSON_PATHS.items():
    print(f"\nProcessing data for {dataset_name} from: {json_path}")

    if not json_path.exists():
        print(f"Warning: JSON file not found at {json_path}. Skipping {dataset_name}.")
        dataset_reports[dataset_name] = {} # Store empty dict if not found
        continue

    try:
        with open(json_path, 'r') as f:
            report_data = json.load(f)
            dataset_reports[dataset_name] = report_data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}. Skipping {dataset_name}.")
        dataset_reports[dataset_name] = {}
    except Exception as e:
        print(f"An unexpected error occurred while processing {dataset_name}: {e}")

# Prepare the data for the combined table
# Start with the 16 shared classes
for class_name in SHARED_CLASSES:
    row_data = {"Class": class_name}
    # Iterate through each dataset type (1st, 2nd, Combined)
    for dataset_prefix, ds_name in [("1st", "1st_Dataset"), ("2nd", "2nd_Dataset"), ("Combined", "Combined_Dataset")]:
        report = dataset_reports.get(ds_name, {})
        metrics = report.get(class_name, {})
        row_data[f"Precision {dataset_prefix}"] = metrics.get("precision", np.nan)
        row_data[f"Recall {dataset_prefix}"] = metrics.get("recall", np.nan)
        row_data[f"F1 {dataset_prefix}"] = metrics.get("f1-score", np.nan)
    data_for_df.append(row_data)

# Add "macro avg" and "weighted avg" rows
for agg_type_raw in AGGREGATE_ROWS:
    agg_type_display = agg_type_raw.replace("_", " ") + "." # Format "macro avg." etc.
    row_data = {"Class": agg_type_display}
    # Iterate through each dataset type (1st, 2nd, Combined)
    for dataset_prefix, ds_name in [("1st", "1st_Dataset"), ("2nd", "2nd_Dataset"), ("Combined", "Combined_Dataset")]:
        report = dataset_reports.get(ds_name, {})
        metrics = report.get(agg_type_raw, {})
        row_data[f"Precision {dataset_prefix}"] = metrics.get("precision", np.nan)
        row_data[f"Recall {dataset_prefix}"] = metrics.get("recall", np.nan)
        row_data[f"F1 {dataset_prefix}"] = metrics.get("f1-score", np.nan)
    data_for_df.append(row_data)

if not data_for_df:
    print("No data was processed to generate the table. Please check JSON paths and content.")
    exit(0)

# Create DataFrame from the prepared data
performance_df = pd.DataFrame(data_for_df)

# Define the desired column order for the final CSV (Added Combined columns)
COLUMNS_ORDER = [
    "Class",
    "Precision 1st", "Recall 1st", "F1 1st",
    "Precision 2nd", "Recall 2nd", "F1 2nd",
    "Precision Combined", "Recall Combined", "F1 Combined"
]
performance_df = performance_df[COLUMNS_ORDER]

# Format numerical columns to 4 decimal places
# Use a lambda function with f-string for consistent formatting of NaNs as empty strings
for col in COLUMNS_ORDER:
    if "Precision" in col or "Recall" in col or "F1" in col:
        performance_df[col] = performance_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '')

# Print the table to console in Markdown format
print("\n--- Cultigen Model Performance Summary Table ---")
print(performance_df.to_markdown(index=False))

# Save the DataFrame to CSV
output_csv_path = OUTPUT_DIR / OUTPUT_TABLE_FILENAME
performance_df.to_csv(output_csv_path, index=False)
print(f"\nPerformance table saved to: {output_csv_path}")

print("\n--- Script finished ---")
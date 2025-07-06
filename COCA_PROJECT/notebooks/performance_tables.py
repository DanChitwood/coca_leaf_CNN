#######################
### LOAD IN MODULES ###
#######################

import json
import pandas as pd
from pathlib import Path
import sys
import numpy as np # Used for np.nan

#############################
### CONFIGURATION ###
#############################

# Base directory for data and figures, relative to the location of this script.
# If this script is in 'COCA_PROJECT/notebooks/', then BASE_PROJECT_DIR points to 'COCA_PROJECT/'
BASE_PROJECT_DIR = Path("../")

# Filename for the JSON classification reports
JSON_FILENAME = "ECT_Mask_2Channel_CNN_Ensemble_Improved_classification_report_Leaf_Class.json"

# Paths to the JSON files for each dataset
JSON_PATHS = {
    "Plowman": BASE_PROJECT_DIR / "data" / "PLOWMAN" / "trained_models" / "metrics_output" / JSON_FILENAME,
    "Cultigens (1st)": BASE_PROJECT_DIR / "data" / "CULTIVATED1ST" / "trained_models" / "metrics_output" / JSON_FILENAME,
    "Wildspecies": BASE_PROJECT_DIR / "data" / "WILDSPECIES" / "trained_models" / "metrics_output" / JSON_FILENAME,
}

# Output directory for the performance table
OUTPUT_DIR = BASE_PROJECT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure the figures directory exists
OUTPUT_TABLE_FILENAME = "overall_model_performance_summary.csv"

# Define the order of datasets for the table
DATASET_ORDER = ["Plowman", "Cultigens (1st)", "Wildspecies"]

# Define the order of classes for the table (full list)
# Only classes present in a given dataset's JSON will be included for that dataset
CLASS_ORDER = [
    "coca", "ipadu", "novogranatense", "truxillense",
    "cataractarum", "foetidum", "gracilipes", "lineolatum"
]

#############################
### GENERATE PERFORMANCE TABLE ###
#############################

print("--- Generating Overall Model Performance Table ---")

table_data = []

for dataset_display_name in DATASET_ORDER:
    json_path = JSON_PATHS[dataset_display_name]

    print(f"\nProcessing data for {dataset_display_name} from: {json_path}")

    if not json_path.exists():
        print(f"Warning: JSON file not found at {json_path}. Skipping {dataset_display_name}.")
        continue

    try:
        with open(json_path, 'r') as f:
            report_data = json.load(f)

        # Add "macro avg" row
        macro_avg = report_data.get("macro avg", {})
        table_data.append({
            "Dataset": dataset_display_name,
            "Class": "macro avg.",
            "Precision": macro_avg.get("precision", np.nan),
            "Recall": macro_avg.get("recall", np.nan),
            "F1": macro_avg.get("f1-score", np.nan)
        })

        # Add "weighted avg" row
        weighted_avg = report_data.get("weighted avg", {})
        table_data.append({
            "Dataset": dataset_display_name,
            "Class": "weighted avg.",
            "Precision": weighted_avg.get("precision", np.nan),
            "Recall": weighted_avg.get("recall", np.nan),
            "F1": weighted_avg.get("f1-score", np.nan)
        })

        # Add individual class rows in specified order
        for class_name in CLASS_ORDER:
            # Check if the class exists in the current report data and is a dictionary (not 'accuracy')
            if class_name in report_data and isinstance(report_data[class_name], dict):
                class_metrics = report_data[class_name]
                table_data.append({
                    "Dataset": dataset_display_name,
                    "Class": class_name,
                    "Precision": class_metrics.get("precision", np.nan),
                    "Recall": class_metrics.get("recall", np.nan),
                    "F1": class_metrics.get("f1-score", np.nan)
                })
            # If a class from CLASS_ORDER is not present in the current dataset's report, it's skipped for that dataset.

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}. Skipping {dataset_display_name}.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {dataset_display_name}: {e}")


if not table_data:
    print("No data was processed to generate the table. Please check JSON paths and content.")
    sys.exit(0)

# Create DataFrame
performance_df = pd.DataFrame(table_data)

# Format numerical columns to 4 decimal places for display and CSV
# Using .round() for numerical precision and then .astype(str) for consistent string formatting
# This prevents potential issues with to_csv/to_markdown and float precision
numeric_cols = ["Precision", "Recall", "F1"]
for col in numeric_cols:
    # Round to 4 decimal places where not NaN, then format to string
    performance_df[col] = performance_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '')


# Print the table to console in Markdown format
print("\nOverall Model Performance Summary Table:")
print(performance_df.to_markdown(index=False))

# Save the DataFrame to CSV
output_csv_path = OUTPUT_DIR / OUTPUT_TABLE_FILENAME
performance_df.to_csv(output_csv_path, index=False)
print(f"\nPerformance table saved to: {output_csv_path}")

print("\n--- Script finished ---")
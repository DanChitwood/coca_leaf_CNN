#######################
### LOAD IN MODULES ###
#######################

import pandas as pd
from pathlib import Path
import os # For creating directories

#############################
### CONFIGURATION ###
#############################

# Define the base project directory.
# Since the script runs from COCA_PROJECT/data/PLANTPREDICT,
# and the project root is COCA_PROJECT, we go up two levels.
BASE_PROJECT_DIR = Path("../../")

# Input file path relative to BASE_PROJECT_DIR
INPUT_CSV_PATH = BASE_PROJECT_DIR / "data" / "CULTIVATED2ND" / "01_cultivated2nd_landmarks.csv"

# Output directory relative to BASE_PROJECT_DIR
OUTPUT_DIR = BASE_PROJECT_DIR / "data" / "PLANTPREDICT"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists

# Output file path relative to OUTPUT_DIR
OUTPUT_CSV_FILENAME = "01_cultivated2nd_landmarks.csv"
OUTPUT_CSV_PATH = OUTPUT_DIR / OUTPUT_CSV_FILENAME

#############################
### SCRIPT LOGIC ###
#############################

print("--- Starting 00_metadata.py script ---")

print(f"Reading input CSV from: {INPUT_CSV_PATH}")

try:
    # Load the CSV file
    df = pd.read_csv(INPUT_CSV_PATH)

    # Create the new 'plantID' column by concatenating 'variety' and 'plant'
    # Check if 'variety' and 'plant' columns exist
    if 'variety' in df.columns and 'plant' in df.columns:
        df['plantID'] = df['variety'].astype(str) + '_' + df['plant'].astype(str)
        print("Successfully created 'plantID' column.")
    else:
        print("Warning: 'variety' or 'plant' column not found. 'plantID' column not created.")

    # Save the modified DataFrame to the new location
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Modified CSV saved to: {OUTPUT_CSV_PATH}")

except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_CSV_PATH}. Please ensure the path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("--- 00_metadata.py script finished ---")
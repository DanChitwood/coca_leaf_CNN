import pandas as pd
from pathlib import Path

# --- Configuration ---
# Input paths for original landmark files
# Corrected paths: One '..' to go up from SYNTHETICPLANT to data/, then into the respective folders.
CULTIVATED1ST_LANDMARKS_PATH = Path("../CULTIVATED1ST/01_cultivated1st_landmarks.csv")
CULTIVATED2ND_LANDMARKS_PATH = Path("../CULTIVATED2ND/01_cultivated2nd_landmarks.csv")

# Output path for the combined plant landmarks metadata
# This assumes the script is run from 'COCA_PROJECT/data/SYNTHETICPLANT'
OUTPUT_DIR = Path(".") # This means the current directory where the script is run
OUTPUT_METADATA_FILE = OUTPUT_DIR / "01_plant_landmarks.csv"
PLANT_ID_COUNTS_FILE = OUTPUT_DIR / "plant_id_counts_by_full_name.txt"

# Ensure output directory exists (it should be the current directory, but good practice)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
print("Loading cultivated1st_landmarks.csv...")
df1 = pd.read_csv(CULTIVATED1ST_LANDMARKS_PATH)
df1['dataset'] = 'first' # Add 'dataset' column

print("Loading cultivated2nd_landmarks.csv...")
df2 = pd.read_csv(CULTIVATED2ND_LANDMARKS_PATH)
df2['dataset'] = 'second' # Add 'dataset' column

# --- Select and Concatenate Relevant Columns ---
# Columns to retain - NOW INCLUDING base_x, base_y, tip_x, tip_y
RELEVANT_COLUMNS = ["file", "variety", "full_name", "plant", "leaf", "dataset", "base_x", "base_y", "tip_x", "tip_y"]

df_combined = pd.concat([df1[RELEVANT_COLUMNS], df2[RELEVANT_COLUMNS]], ignore_index=True)

print(f"Combined data has {len(df_combined)} rows.")

# --- Standardize 'full_name' entries in the second dataset ---
# Create a mapping for standardization in df2
name_standardization_map = {
    "BON": "boliviana negra",
    "DES": "desconocido",
    "POM": "pomarosa"
}

# Apply standardization specifically to rows from the 'second' dataset
# This ensures we don't accidentally rename varieties in the 'first' dataset if they had these abbreviations.
# Using .loc for safe modification
mask_second_dataset = df_combined['dataset'] == 'second'
df_combined.loc[mask_second_dataset, 'full_name'] = df_combined.loc[mask_second_dataset, 'full_name'].replace(name_standardization_map)

print("Standardized 'full_name' entries for 'second' dataset where applicable.")

# --- Create 'plantID' Column ---
# Replace spaces with underscores in 'full_name' for plantID creation
df_combined['full_name_clean'] = df_combined['full_name'].str.replace(' ', '_')
df_combined['plantID'] = df_combined['full_name_clean'] + "_" + df_combined['dataset'] + "_" + df_combined['plant'].astype(str)

# Drop the temporary cleaned full_name column if you don't need it
df_combined = df_combined.drop(columns=['full_name_clean'])

print("Created 'plantID' column.")

# --- Save Combined Metadata ---
df_combined.to_csv(OUTPUT_METADATA_FILE, index=False)
print(f"Combined plant landmarks saved to: {OUTPUT_METADATA_FILE}")

# --- Print and Save Unique PlantID Counts per Full Name ---
print("\n--- Unique PlantID Counts per 'full_name' ---")
# Count unique plantIDs for each full_name
unique_plant_counts = df_combined.groupby('full_name')['plantID'].nunique().sort_index()

print(unique_plant_counts)

# Save to file
with open(PLANT_ID_COUNTS_FILE, 'w') as f:
    f.write("Unique PlantID Counts per 'full_name':\n")
    f.write(unique_plant_counts.to_string())

print(f"\nUnique PlantID counts saved to: {PLANT_ID_COUNTS_FILE}")

print("\nScript finished successfully.")
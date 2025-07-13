#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

############################
### CONFIGURATION (Relevant parts from 06_train_CNN.py) ###
############################

# Define the path to your COCA_PROJECT directory (adjust if different)
COCA_PROJECT_ROOT = Path("/Users/chitwoo9/Desktop/COCA_PROJECT")

# Paths to the consolidated data for CNN training
DATASET_PATHS = {
    "cultivated1st": COCA_PROJECT_ROOT / "data" / "CULTIVATED1ST" / "05_synthetic_leaf_data_cultivated1st",
    "cultivated2nd": COCA_PROJECT_ROOT / "data" / "CULTIVATED2ND" / "05_synthetic_leaf_data_cultivated2nd",
}

# --- Image Parameters (for verification, taken from 04_synthetic_data_generation.py) ---
IMAGE_SIZE = (256, 256) # Expected image size

###########################
### DEBUGGING SCRIPT ###
###########################

def debug_dataset_loading(dataset_name: str, dataset_base_path: Path):
    """
    Loads a dataset, performs label encoding, and prints detailed debugging information.
    """
    print(f"\n{'='*10} DEBUGGING DATA LOADING FOR: {dataset_name.upper()} {'='*10}")

    final_prepared_data_file = dataset_base_path / "final_cnn_dataset.pkl"

    if not final_prepared_data_file.exists():
        print(f"Error: Data file not found for {dataset_name} at {final_prepared_data_file}")
        return

    try:
        with open(final_prepared_data_file, 'rb') as f:
            data = pickle.load(f)

        X = data['X_images']
        y_encoded = data['y_labels_encoded']
        class_names = data['class_names']
        is_real_flags = data['is_real_flags']
        
        # Verify image_size and num_channels
        loaded_image_size = data.get('image_size', None)
        loaded_num_channels = data.get('num_channels', None)

        print(f"Loaded X_images shape: {X.shape}")
        print(f"Loaded y_encoded shape: {y_encoded.shape}")
        print(f"Number of samples (total): {X.shape[0]}")
        
        print(f"Expected Image Size (from config): {IMAGE_SIZE}")
        print(f"Loaded Image Size (from pkl): {loaded_image_size}")
        print(f"Loaded Number of Channels (from pkl): {loaded_num_channels}")
        print(f"Actual X_images channels: {X.shape[-1]}")


        # --- Label Encoding and Class Information ---
        # Note: y_encoded and class_names are already from the LabelEncoder during data generation
        # in 04_synthetic_data_generation.py, but we'll re-verify it's consistent.
        
        print(f"\n--- Class Information for {dataset_name} ---")
        print(f"Encoded class names (used by model): {class_names}")
        print(f"Number of encoded classes: {len(class_names)}")
        
        print("\nEncoded Label to Class Name Mapping:")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")

        # Check distribution of encoded labels
        unique_encoded, counts_encoded = np.unique(y_encoded, return_counts=True)
        print(f"\nDistribution of Encoded Labels for {dataset_name}:")
        # Ensure that unique_encoded and counts_encoded are iterated in tandem
        for enc_val, count in zip(unique_encoded, counts_encoded):
            # Guard against index out of bounds if class_names is somehow malformed
            if enc_val < len(class_names):
                print(f"  Label {enc_val} ({class_names[enc_val]}): {count} samples")
            else:
                print(f"  WARNING: Label {enc_val} exists but no corresponding class name found.")


        # --- Data Channel Inspection (Mask and ECT) ---
        if X.shape[-1] == 2: # Assuming 2 channels (Mask and ECT)
            mask_channel_data = X[:, :, :, 0] # Channel 0: Mask
            ect_channel_data = X[:, :, :, 1] # Channel 1: ECT

            print(f"\n--- Mask Channel (Channel 0) Pixel Value Statistics for {dataset_name} ---")
            print(f"  Min pixel value: {mask_channel_data.min():.4f}")
            print(f"  Max pixel value: {mask_channel_data.max():.4f}")
            print(f"  Mean pixel value: {mask_channel_data.mean():.4f}")
            print(f"  Std Dev pixel value: {mask_channel_data.std():.4f}")

            print(f"\n--- ECT Channel (Channel 1) Pixel Value Statistics for {dataset_name} ---")
            print(f"  Min pixel value: {ect_channel_data.min():.4f}")
            print(f"  Max pixel value: {ect_channel_data.max():.4f}")
            print(f"  Mean pixel value: {ect_channel_data.mean():.4f}")
            print(f"  Std Dev pixel value: {ect_channel_data.std():.4f}")
            
            # Check for near-zero standard deviation (indicates flat/uniform image)
            if ect_channel_data.std() < 1e-5:
                print(f"  WARNING: ECT channel standard deviation is extremely low, suggesting uniform/empty images.")
        else:
            print(f"\nWARNING: Expected 2 channels, but found {X.shape[-1]} channels in X_images.")


        # --- Real vs. Synthetic Samples ---
        if is_real_flags is not None:
            real_counts = np.bincount(is_real_flags) # False (0) and True (1)
            print(f"\n--- Real vs. Synthetic Sample Counts for {dataset_name} ---")
            if len(real_counts) > 0:
                print(f"  Synthetic Samples (is_real=False): {real_counts[0] if len(real_counts) > 0 else 0} (approx. {real_counts[0]/X.shape[0]*100:.2f}%)")
            if len(real_counts) > 1:
                print(f"  Real Samples (is_real=True): {real_counts[1] if len(real_counts) > 1 else 0} (approx. {real_counts[1]/X.shape[0]*100:.2f}%)")
            else:
                print("  is_real_flags array is missing or malformed.")


    except Exception as e:
        print(f"An error occurred while debugging {dataset_name}: {e}")

if __name__ == "__main__":
    # You can choose which dataset to debug
    # debug_dataset_loading("cultivated1st", DATASET_PATHS["cultivated1st"])
    debug_dataset_loading("cultivated2nd", DATASET_PATHS["cultivated2nd"])

    print("\nDebugging complete.")
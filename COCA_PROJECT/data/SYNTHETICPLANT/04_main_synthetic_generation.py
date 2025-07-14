#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import Path
import h5py
import sys
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import disk
import cv2
from scipy.interpolate import griddata
import time
from collections import Counter

############################
### CONFIGURATION (ALL PARAMETERS UP FRONT) ###
############################

# --- General Configuration ---
# Use pathlib for robust path handling
BASE_DIR = Path("./") # Current directory for the script
OUTPUT_DIR = BASE_DIR / "05_synthetic_leaf_data"

# --- Data Input Configuration ---
PCA_MODEL_PARAMS_FILE = BASE_DIR / "03_morphometrics_output_combined" / "leaf_pca_model_parameters_combined.h5"
ORIGINAL_PCA_SCORES_LABELS_FILE = BASE_DIR / "03_morphometrics_output_combined" / "original_pca_scores_and_class_labels_combined.h5"

# --- Synthetic Data Generation Parameters ---
NUM_LEAVES_PER_PLANTID_COLLECTION = 10 # Number of leaves (ECT/Mask pairs) per plant ID
TARGET_PLANTIDS_PER_FULL_NAME = 360 # Target number of unique plant IDs per full_name (variety)
NUM_NEAREST_NEIGHBORS = 5 # For finding similar ECTs for synthetic generation
SYNTHETIC_NOISE_STD_DEV = 0.05 # Standard deviation for adding noise to synthetic ECT values

# --- Image Processing Parameters (for reconstructing ECT/Mask images) ---
IMAGE_SIZE_FOR_SYNTHESIS = (256, 256) # (Height, Width)
ECT_IMAGE_BG_VALUE = 0.0 # Background value for ECT images
DILATION_KERNEL_SIZE_FOR_MASK = 5 # Size of disk kernel for mask dilation
EROSION_KERNEL_SIZE_FOR_MASK = 3 # Size of disk kernel for mask erosion
GRADIENT_SMOOTHING_KERNEL_SIZE = 5 # Kernel size for Gaussian smoothing on ECT gradient

# --- Output File Names ---
FINAL_PREPARED_DATA_FILE = OUTPUT_DIR / "final_cnn_dataset_plantID_level.pkl"

#############################
### ECT & MASK PROCESSING ###
#############################

# Helper class to store ECT and mask data
class ECTResult:
    def __init__(self, mask, ect_values):
        self.mask = mask
        self.ect_values = ect_values

def process_and_save_leaf_data(mask_image, ect_image, image_size, global_min_ect, global_max_ect):
    """
    Processes a single mask and ECT image pair, normalizes ECT,
    and returns a 2-channel numpy array (mask, normalized ECT).
    Does NOT save images to disk in this version, just returns the array.
    """
    if mask_image.dtype == bool:
        mask_image = mask_image.astype(np.uint8) * 255 # Convert boolean mask to uint8 (0 or 255)

    # Ensure mask is binary (0 or 255)
    mask_image_binary = (mask_image > 0).astype(np.uint8) * 255

    # --- Pre-processing for Mask (optional, based on observed data quality) ---
    # Convert to boolean for morphological operations
    mask_bool = mask_image_binary.astype(bool)

    # Dilate and then erode to close small holes and smooth edges (optional, adjust kernel sizes)
    dilated_mask = binary_dilation(mask_bool, structure=disk(DILATION_KERNEL_SIZE_FOR_MASK))
    processed_mask_bool = binary_erosion(dilated_mask, structure=disk(EROSION_KERNEL_SIZE_FOR_MASK))

    # Convert back to uint8 (0 or 255)
    processed_mask_uint8 = processed_mask_bool.astype(np.uint8) * 255

    # --- ECT Processing ---
    # Apply mask to ECT values to ensure background is ECT_IMAGE_BG_VALUE (e.g., 0)
    ect_image_masked = ect_image * processed_mask_bool

    # Normalize ECT values globally based on the real data's min/max
    # This is crucial for consistent scaling across real and synthetic data
    if global_max_ect - global_min_ect > 0:
        normalized_ect = (ect_image_masked - global_min_ect) / (global_max_ect - global_min_ect)
    else: # Handle cases where there's no variance (e.g., all same ECT value)
        normalized_ect = np.zeros_like(ect_image_masked)

    # Ensure normalization clamps values between 0 and 1
    normalized_ect = np.clip(normalized_ect, 0, 1)

    # Stack mask and normalized ECT into a 2-channel array
    # Ensure both are float32 for consistency with model input
    # Mask should be 0-1 for CNN input
    final_mask_channel = (processed_mask_uint8 / 255.0).astype(np.float32)
    final_ect_channel = normalized_ect.astype(np.float32)

    # Ensure consistent sizing (should already be IMAGE_SIZE_FOR_SYNTHESIS)
    if final_mask_channel.shape != image_size or final_ect_channel.shape != image_size:
        final_mask_channel = cv2.resize(final_mask_channel, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        final_ect_channel = cv2.resize(final_ect_channel, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)

    # Stack as (H, W, 2)
    two_channel_image = np.stack([final_mask_channel, final_ect_channel], axis=-1)

    return two_channel_image

def reconstruct_ect_from_pca(pca_scores, pca_model, scaler_ect, original_ect_shape, mask_for_reconstruction):
    """
    Reconstructs an ECT image from PCA scores.
    Args:
        pca_scores (np.array): 1D array of PCA scores for a single ECT.
        pca_model (PCA): Fitted PCA model.
        scaler_ect (StandardScaler): Fitted StandardScaler for ECT data.
        original_ect_shape (tuple): (height, width) of the original flattened ECT images.
        mask_for_reconstruction (np.array): Binary mask for the leaf (bool or uint8).
    Returns:
        np.array: Reconstructed ECT image, original shape, with background zeroed.
    """
    # Inverse transform PCA scores to get back original (flattened) features
    reconstructed_flattened = pca_model.inverse_transform(pca_scores.reshape(1, -1))
    # Inverse transform the scaler to get back original ECT value scale
    reconstructed_ect_unscaled = scaler_ect.inverse_transform(reconstructed_flattened)
    
    # Reshape back to 2D image
    reconstructed_ect_2d = reconstructed_ect_unscaled.reshape(original_ect_shape)

    # Apply the mask to zero out background pixels
    if mask_for_reconstruction.dtype == bool:
        masked_ect = reconstructed_ect_2d * mask_for_reconstruction
    else: # Assume uint8 (0 or 255)
        masked_ect = reconstructed_ect_2d * (mask_for_reconstruction > 0)
    
    return masked_ect

# This function is currently not used in the main generation loop,
# but keeping it for completeness if a future use case arises.
def generate_synthetic_ect_mask_pair_unused(original_mask, original_ect_coords, ect_pca_model, ect_scaler, original_ect_img_shape, global_min_ect, global_max_ect, noise_std_dev=0.05):
    pass # This function is not currently used in the main generation loop.

def load_pca_model_data(pca_model_params_file, original_pca_scores_labels_file):
    """
    Loads PCA model components, scaler, and original PCA scores and labels.
    """
    print(f"Loaded PCA model parameters from {pca_model_params_file}.")
    with h5py.File(pca_model_params_file, 'r') as f:
        pca_components = f['pca_components'][()]
        pca_mean = f['pca_mean'][()]
        scaler_mean = f['scaler_mean'][()]
        scaler_scale = f['scaler_scale'][()]
        original_ect_image_shape = tuple(f['original_ect_image_shape'][()])

    # Reconstruct PCA model and scaler
    pca_model = PCA(n_components=pca_components.shape[0])
    pca_model.components_ = pca_components
    pca_model.mean_ = pca_mean

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    scaler.var_ = scaler_scale**2 # StandardScaler uses var_ internally for inverse_transform

    print(f"Loaded original PCA scores, plantID labels, and full_name labels from {original_pca_scores_labels_file}.")
    with h5py.File(original_pca_scores_labels_file, 'r') as f:
        original_pca_scores = f['pca_scores'][()]
        plantid_labels = f['plantid_labels'][()].astype(str) # Ensure string type
        full_name_labels = f['full_name_labels'][()].astype(str) # Ensure string type
        original_ect_masks = f['ect_masks'][()] # The actual 2-channel ECT/Mask images

    # Global min/max for ECT values from original data for consistent normalization
    global_ect_min = original_ect_masks[:, :, :, 1].min() # Min from ECT channel (index 1)
    global_ect_max = original_ect_masks[:, :, :, 1].max() # Max from ECT channel (index 1)

    print(f"Global ECT Min: {global_ect_min:.4f}, Max: {global_ect_max:.4f}")

    return pca_model, scaler, original_pca_scores, plantid_labels, full_name_labels, original_ect_image_shape, original_ect_masks, global_ect_min, global_ect_max


def generate_targeted_synthetic_pca_samples(
    current_full_name_original_pca_scores,
    target_count,
    num_leaves_per_plantid,
    ect_pca_model,
    ect_scaler,
    original_ect_image_shape,
    original_ect_masks_data, # Renamed to avoid confusion with global variable
    global_min_ect,
    global_max_ect,
    noise_std_dev,
    all_original_plant_ids, # All unique original plant IDs
    all_original_full_names # All unique original full_names (for mask sampling)
):
    """
    Generates synthetic PCA scores for new plant IDs, ensuring each target 'full_name'
    reaches `target_count` unique plant IDs.
    For each new synthetic plant ID, `num_leaves_per_plantid` synthetic leaves are generated.

    Args:
        current_full_name_original_pca_scores (pd.DataFrame): PCA scores for original plant IDs
                                                               belonging to the current full_name.
                                                               Must contain 'plantID' and PC columns.
        target_count (int): The desired number of plant IDs for this full_name.
        num_leaves_per_plantid (int): Number of synthetic leaves to generate per synthetic plant ID.
        ect_pca_model (PCA): Fitted PCA model for ECT.
        ect_scaler (StandardScaler): Fitted scaler for ECT data.
        original_ect_image_shape (tuple): (H, W) of original ECT images.
        original_ect_masks_data (np.array): All original 2-channel ECT/Mask images.
        global_min_ect (float): Global min ECT for normalization.
        global_max_ect (float): Global max ECT for normalization.
        noise_std_dev (float): Standard deviation for adding noise to synthetic ECT values.
        all_original_plant_ids (np.array): All unique original plant IDs.
        all_original_full_names (np.array): All unique original full names.

    Returns:
        list: A list of dictionaries, each representing a synthetic leaf for a synthetic plant ID.
              Each dict contains: 'leaf_image_2channel', 'pca_scores', 'plantID', 'full_name'.
    """
    synthetic_plantid_data = []
    
    # Get the specific full_name this DataFrame belongs to
    if not current_full_name_original_pca_scores.empty:
        current_full_name = current_full_name_original_pca_scores['full_name'].iloc[0]
    else:
        # This case should ideally not happen if loop correctly filters, but handle defensively
        return []

    # Get the unique plant IDs already present for this full_name
    existing_plant_ids_for_full_name = current_full_name_original_pca_scores['plantID'].unique()
    num_existing_plant_ids = len(existing_plant_ids_for_full_name)

    num_plantids_to_generate = target_count - num_existing_plant_ids

    if num_plantids_to_generate <= 0:
        print(f"  '{current_full_name}': Already has {num_existing_plant_ids} plant IDs, no new plant IDs needed.")
        return []

    print(f"  '{current_full_name}': Generating {num_plantids_to_generate} new synthetic plant IDs to reach {target_count}...")

    # Identify PCA columns
    pca_cols = [col for col in current_full_name_original_pca_scores.columns if 'PC' in col]
    original_pca_scores_subset = current_full_name_original_pca_scores[pca_cols].values

    # Find the k-nearest neighbors in PCA space among existing plant IDs of this full_name
    # This helps ensure synthetic plant IDs are 'similar' to existing ones in their group.
    if len(original_pca_scores_subset) < NUM_NEAREST_NEIGHBORS:
        # If not enough samples, use all available as neighbors
        nn_model = NearestNeighbors(n_neighbors=len(original_pca_scores_subset) - 1 if len(original_pca_scores_subset) > 1 else 1)
    else:
        nn_model = NearestNeighbors(n_neighbors=NUM_NEAREST_NEIGHBORS)

    if len(original_pca_scores_subset) > 0:
        nn_model.fit(original_pca_scores_subset)

    for i in range(num_plantids_to_generate):
        # Generate a unique synthetic plant ID string
        # Combine full_name and a unique counter to form a new synthetic plantID
        synthetic_plant_id = f"{current_full_name.replace(' ', '_')}_syn_plant_{i+1:04d}"

        # Select a base PCA score for perturbation
        if len(original_pca_scores_subset) > 0:
            # Pick a random existing PCA score from this variety to base the synthetic one on
            random_original_idx_in_subset = np.random.randint(0, len(original_pca_scores_subset))
            base_pca_score = original_pca_scores_subset[random_original_idx_in_subset]

            # Optionally find its nearest neighbors and average them to get a smoother 'base'
            distances, indices = nn_model.kneighbors(base_pca_score.reshape(1, -1))
            averaged_neighbor_pca_score = np.mean(original_pca_scores_subset[indices[0]], axis=0)
            synthetic_pca_score_for_plantid = averaged_neighbor_pca_score
        else:
            # If no original samples exist for this full_name (unlikely but defensive)
            # Create a synthetic PCA score from a random point in the overall PCA space
            print(f"  Warning: No original samples for '{current_full_name}'. Generating random PCA scores for synthetic plant ID.")
            synthetic_pca_score_for_plantid = np.random.uniform(
                ect_pca_model.components_.min(),
                ect_pca_model.components_.max(),
                size=ect_pca_model.n_components_
            )


        # Generate NUM_LEAVES_PER_PLANTID_COLLECTION leaves for this synthetic plant ID
        for leaf_idx in range(num_leaves_per_plantid):
            # Perturb the plant-level synthetic PCA score slightly for each leaf
            # This introduces intra-plant variation for synthetic leaves
            perturbed_pca_score = synthetic_pca_score_for_plantid + np.random.normal(0, SYNTHETIC_NOISE_STD_DEV, size=ect_pca_model.n_components_)

            # Reconstruct ECT values from the perturbed PCA scores
            reconstructed_ect_values = ect_pca_model.inverse_transform(perturbed_pca_score.reshape(1, -1))
            reconstructed_ect_values_unscaled = ect_scaler.inverse_transform(reconstructed_ect_values).flatten()
            
            # Reshape ECT values back to image shape
            synthetic_ect_image = reconstructed_ect_values_unscaled.reshape(original_ect_image_shape)

            # Select an existing mask to pair with this synthetic ECT
            # Find a mask from an original plant ID of the same full_name
            # If no original leaves for this full_name, pick a random mask from overall dataset
            original_masks_for_full_name_indices = [
                idx for idx, fname in enumerate(all_original_full_names)
                if fname == current_full_name
            ]

            if original_masks_for_full_name_indices:
                random_original_mask_idx = np.random.choice(original_masks_for_full_name_indices)
                synthetic_mask = original_ect_masks_data[random_original_mask_idx, :, :, 0] # Get only the mask channel
            else:
                # Fallback: pick a random mask from the entire original dataset
                random_original_mask_idx = np.random.randint(0, len(original_ect_masks_data))
                synthetic_mask = original_ect_masks_data[random_original_mask_idx, :, :, 0]

            # Create an ECTResult object for consistency
            synthetic_ect_result = ECTResult(mask=synthetic_mask, ect_values=synthetic_ect_image)

            # Process and store the synthetic leaf data as a 2-channel image
            leaf_image_2channel = process_and_save_leaf_data(
                mask_image=synthetic_ect_result.mask,
                ect_image=synthetic_ect_result.ect_values,
                image_size=original_ect_image_shape,
                global_min_ect=global_min_ect,
                global_max_ect=global_max_ect,
            )

            synthetic_plantid_data.append({
                'leaf_image_2channel': leaf_image_2channel,
                'pca_scores': perturbed_pca_score,
                'plantID': synthetic_plant_id,
                'full_name': current_full_name,
                'is_real': False # Flag for synthetic data
            })
    return synthetic_plantid_data


#######################
### MAIN FUNCTION ###
#######################

def main():
    print("--- Starting Synthetic Data Generation ---")

    # --- 1. Set up output directories ---
    print("\n--- Setting up output directories ---")
    if OUTPUT_DIR.exists():
        print(f"Cleaned existing directory: {OUTPUT_DIR}")
        import shutil
        shutil.rmtree(OUTPUT_DIR) # Remove existing directory and contents
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Output directories created.")

    # --- 2. Load PCA model and original data ---
    pca_model, scaler_ect, original_pca_scores, plantid_labels_raw, full_name_labels_raw, \
    original_ect_image_shape, original_ect_masks, global_ect_min, global_ect_max = \
        load_pca_model_data(PCA_MODEL_PARAMS_FILE, ORIGINAL_PCA_SCORES_LABELS_FILE)

    # Create a DataFrame for easier querying of original data
    original_df = pd.DataFrame(original_pca_scores, columns=[f'PC{i+1}' for i in range(original_pca_scores.shape[1])])
    original_df['plantID'] = plantid_labels_raw
    original_df['full_name'] = full_name_labels_raw
    original_df['original_index'] = original_df.index.values # Store original index for image lookup

    # Group original data by plantID
    plantid_to_full_name = original_df.set_index('plantID')['full_name'].to_dict()
    unique_original_plant_ids = original_df['plantID'].unique()
    unique_original_full_names = original_df['full_name'].unique()

    plant_id_pca_scores = {}
    plant_id_full_names = {}
    plant_id_leaf_images = {} # To store the 2-channel (mask, ECT) images per leaf for each plantID

    print(f"\nFound {len(unique_original_plant_ids)} unique plant IDs.")

    # Process original data, collect into plantID-level collections
    print("\n--- Processing original real leaf data into PlantID collections ---")
    for plant_id in unique_original_plant_ids:
        leaves_for_this_plant = original_df[original_df['plantID'] == plant_id]
        
        # Collect PCA scores and 2-channel images for each leaf of this plantID
        current_plant_leaves_pca_scores = []
        current_plant_leaves_images = []

        for original_idx in leaves_for_this_plant['original_index'].values:
            pca_score_for_leaf = original_pca_scores[original_idx]
            ect_mask_image = original_ect_masks[original_idx] # (H, W, 2) <- Access original_ect_masks directly
            
            # Process real leaf image (normalize ECT, convert mask to 0-1)
            processed_image = process_and_save_leaf_data(
                mask_image=ect_mask_image[:, :, 0],
                ect_image=ect_mask_image[:, :, 1],
                image_size=original_ect_image_shape,
                global_min_ect=global_ect_min,
                global_max_ect=global_ect_max,
            )
            
            current_plant_leaves_pca_scores.append(pca_score_for_leaf)
            current_plant_leaves_images.append(processed_image)
        
        plant_id_pca_scores[plant_id] = np.array(current_plant_leaves_pca_scores)
        plant_id_full_names[plant_id] = plant_id_to_full_name[plant_id] # Get the full_name for this plantID
        plant_id_leaf_images[plant_id] = np.array(current_plant_leaves_images) # (num_leaves, H, W, 2)

    # --- Determine target counts for each full_name ---
    # Count how many original plant IDs exist per full_name
    original_plantid_counts_per_full_name = original_df.groupby('full_name')['plantID'].nunique().sort_values(ascending=False)
    print("\nOriginal count of unique plantIDs per full_name (variety):")
    print(original_plantid_counts_per_full_name)

    # Calculate target number of plant IDs for each full_name
    # Target maximum is the largest existing count, or the specified TARGET_PLANTIDS_PER_FULL_NAME
    max_original_plantids = original_plantid_counts_per_full_name.max()
    final_target_per_full_name = max(max_original_plantids, TARGET_PLANTIDS_PER_FULL_NAME)
    print(f"\nTargeting {final_target_per_full_name} plant IDs per full_name (variety) for synthetic generation.")

    # --- Generate Synthetic Plant IDs (each with NUM_LEAVES_PER_PLANTID_COLLECTION leaves) ---
    all_synthetic_plantid_leaves_data = []

    print("\n--- Generating synthetic leaves for each plant ID ---")
    for full_name in unique_original_full_names:
        print(f"\nProcessing full_name: '{full_name}'")
        
        # Get original PCA scores for plant IDs belonging to this full_name
        current_full_name_original_df = original_df[original_df['full_name'] == full_name]
        
        synthetic_leaves_for_this_full_name = generate_targeted_synthetic_pca_samples(
            current_full_name_original_pca_scores=current_full_name_original_df,
            target_count=final_target_per_full_name,
            num_leaves_per_plantid=NUM_LEAVES_PER_PLANTID_COLLECTION,
            ect_pca_model=pca_model,
            ect_scaler=scaler_ect,
            original_ect_image_shape=original_ect_image_shape,
            original_ect_masks_data=original_ect_masks, # Pass original_ect_masks directly
            global_min_ect=global_min_ect,
            global_max_ect=global_max_ect,
            noise_std_dev=SYNTHETIC_NOISE_STD_DEV,
            all_original_plant_ids=unique_original_plant_ids,
            all_original_full_names=full_name_labels_raw # Pass all raw full_name labels for mask selection
        )
        all_synthetic_plantid_leaves_data.extend(synthetic_leaves_for_this_full_name)

    print("\n--- Combining Real and Synthetic PlantID Collections ---")

    # Prepare real plantID collections
    X_plantid_collections_list = [] # List to hold (H, W, 2*NUM_LEAVES) arrays
    y_plantid_labels_raw = [] # List to hold string full_name labels
    plantid_is_fully_real_collection = [] # List of booleans

    # Add real plantID collections
    for plant_id in unique_original_plant_ids:
        leaves_images = plant_id_leaf_images[plant_id] # (num_leaves_actual, H, W, 2)
        
        # If a plantID has fewer than NUM_LEAVES_PER_PLANTID_COLLECTION, pad with duplicates
        if leaves_images.shape[0] < NUM_LEAVES_PER_PLANTID_COLLECTION:
            num_to_pad = NUM_LEAVES_PER_PLANTID_COLLECTION - leaves_images.shape[0]
            # Randomly sample with replacement from existing leaves
            padding_indices = np.random.choice(leaves_images.shape[0], num_to_pad, replace=True)
            padded_leaves = leaves_images[padding_indices]
            leaves_images_for_collection = np.concatenate((leaves_images, padded_leaves), axis=0)
            print(f"  PlantID '{plant_id}' had {leaves_images.shape[0]} leaves, padded to {NUM_LEAVES_PER_PLANTID_COLLECTION}.")
        else: # If a plantID has more, randomly sample
            sampled_indices = np.random.choice(leaves_images.shape[0], NUM_LEAVES_PER_PLANTID_COLLECTION, replace=False)
            leaves_images_for_collection = leaves_images[sampled_indices]
        
        # Flatten the leaves into a single (H, W, 2*NUM_LEAVES) tensor
        # This stacks the (mask, ECT) pairs along the channel dimension
        plantid_collection_image = leaves_images_for_collection.reshape(
            original_ect_image_shape[0], # H
            original_ect_image_shape[1], # W
            -1 # 2 * NUM_LEAVES_PER_PLANTID_COLLECTION channels
        )
        X_plantid_collections_list.append(plantid_collection_image)
        y_plantid_labels_raw.append(plant_id_full_names[plant_id]) # Use full_name as label
        plantid_is_fully_real_collection.append(True)

    # Add synthetic plantID collections
    # Group synthetic leaves by their synthetic plantID
    synthetic_plant_ids = {leaf['plantID'] for leaf in all_synthetic_plantid_leaves_data}

    for syn_plant_id in synthetic_plant_ids:
        leaves_for_this_syn_plant = [leaf for leaf in all_synthetic_plantid_leaves_data if leaf['plantID'] == syn_plant_id]
        
        # Collect the 2-channel images for each synthetic leaf of this plantID
        current_syn_plant_leaves_images = [leaf['leaf_image_2channel'] for leaf in leaves_for_this_syn_plant]
        current_syn_plant_leaves_images_np = np.array(current_syn_plant_leaves_images) # (num_leaves, H, W, 2)
        
        # Ensure exactly NUM_LEAVES_PER_PLANTID_COLLECTION leaves
        if current_syn_plant_leaves_images_np.shape[0] != NUM_LEAVES_PER_PLANTID_COLLECTION:
            # This should ideally not happen if generation is correct, but defensively pad/sample
            print(f"  Warning: Synthetic PlantID '{syn_plant_id}' has {current_syn_plant_leaves_images_np.shape[0]} leaves, expected {NUM_LEAVES_PER_PLANTID_COLLECTION}. Adjusting.")
            if current_syn_plant_leaves_images_np.shape[0] < NUM_LEAVES_PER_PLANTID_COLLECTION:
                num_to_pad = NUM_LEAVES_PER_PLANTID_COLLECTION - current_syn_plant_leaves_images_np.shape[0]
                padding_indices = np.random.choice(current_syn_plant_leaves_images_np.shape[0], num_to_pad, replace=True)
                padded_leaves = current_syn_plant_leaves_images_np[padding_indices]
                leaves_images_for_collection = np.concatenate((current_syn_plant_leaves_images_np, padded_leaves), axis=0)
            else:
                sampled_indices = np.random.choice(current_syn_plant_leaves_images_np.shape[0], NUM_LEAVES_PER_PLANTID_COLLECTION, replace=False)
                leaves_images_for_collection = current_syn_plant_leaves_images_np[sampled_indices]
        else:
            leaves_images_for_collection = current_syn_plant_leaves_images_np
        
        plantid_collection_image = leaves_images_for_collection.reshape(
            original_ect_image_shape[0], # H
            original_ect_image_shape[1], # W
            -1 # 2 * NUM_LEAVES_PER_PLANTID_COLLECTION channels
        )
        X_plantid_collections_list.append(plantid_collection_image)
        
        # Extract full_name from the synthetic plant ID string (e.g., 'chiparra_syn_plant_0001' -> 'chiparra')
        # Assuming the format 'full_name_syn_plant_XXXX'
        inferred_full_name = '_'.join(syn_plant_id.split('_')[:-2])
        y_plantid_labels_raw.append(inferred_full_name.replace('_', ' ')) # Convert back to original full_name format
        plantid_is_fully_real_collection.append(False)

    # Convert lists to numpy arrays
    X_plantid_collections = np.array(X_plantid_collections_list) # (Num_PlantIDs, H, W, 2*NUM_LEAVES)
    plantid_is_fully_real_collection_np = np.array(plantid_is_fully_real_collection)

    # Encode full_name labels to numerical values
    label_encoder = LabelEncoder()
    y_plantid_labels_encoded = label_encoder.fit_transform(y_plantid_labels_raw)
    plantid_class_names = label_encoder.classes_.tolist() # Get the string names of classes

    print(f"\nTotal PlantID collections (real + synthetic): {X_plantid_collections.shape[0]}")
    print(f"Number of fully real PlantID collections: {np.sum(plantid_is_fully_real_collection_np)}")
    print(f"Number of synthetic PlantID collections: {np.sum(~plantid_is_fully_real_collection_np)}")
    print(f"Shape of X_plantid_collections: {X_plantid_collections.shape}")
    print(f"Shape of y_plantid_labels_encoded: {y_plantid_labels_encoded.shape}")
    print(f"Number of unique full_name classes: {len(plantid_class_names)}")
    print(f"Full_name class names: {plantid_class_names}")

    # Final data dictionary for CNN training
    final_cnn_dataset = {
        'X_plantid_collections': X_plantid_collections,
        'y_plantid_labels_encoded': y_plantid_labels_encoded,
        'plantid_is_fully_real_collection': plantid_is_fully_real_collection_np,
        'plantid_class_names': plantid_class_names, # These are the full_name strings
        'image_size_per_leaf': original_ect_image_shape,
        'num_leaves_per_plantid': NUM_LEAVES_PER_PLANTID_COLLECTION,
        'total_channels_per_plantid': 2 * NUM_LEAVES_PER_PLANTID_COLLECTION
    }

    # Save the final dataset
    print(f"\n--- Saving final PlantID-level CNN dataset to {FINAL_PREPARED_DATA_FILE} ---")
    with open(FINAL_PREPARED_DATA_FILE, 'wb') as f:
        pickle.dump(final_cnn_dataset, f)
    print("Synthetic data generation complete and dataset saved.")

if __name__ == "__main__":
    main()
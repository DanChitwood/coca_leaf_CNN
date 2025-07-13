import numpy as np
import cv2
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import random

# Assume ECT and other necessary functions/constants are imported or defined here
# from ect import ECT # Assuming ect.py is available
# from _03_morphometric_data_processing_library import ( # Assuming this handles PCA loading/inverse
#    inverse_transform_pca, generate_synthetic_pca_samples, load_pca_model_data
# )

# Dummy imports/definitions for functions/classes from other parts if not explicitly provided
# In a real setup, these would be proper imports from their respective files.
class ECT:
    def __init__(self, num_dirs, thresholds, bound_radius):
        self.num_dirs = num_dirs
        self.thresholds = thresholds
        self.bound_radius = bound_radius

    def calculate_ect(self, outline_coords):
        # Dummy ECT calculation for demonstration
        # In a real scenario, this would compute the ECT values.
        # For a simple placeholder, let's return a simple radial gradient or flat value.
        # This needs to match the expected output of your actual ECT calculator.
        if outline_coords is None or outline_coords.size == 0:
            return np.zeros((256, 256), dtype=np.float32) # Return empty if no coords

        # Simple bounding box for min/max
        min_x, min_y = outline_coords.min(axis=0)
        max_x, max_y = outline_coords.max(axis=0)
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

        # Create a dummy ECT map - for a real ECT, this would involve ray casting etc.
        ect_map = np.zeros(IMAGE_SIZE, dtype=np.float32)
        for y in range(IMAGE_SIZE[1]):
            for x in range(IMAGE_SIZE[0]):
                dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                # Simple inverse distance mapping, normalized
                ect_map[y, x] = 1.0 - (dist_to_center / (max(IMAGE_SIZE) / 2)) # Normalize to [0,1]

        # Apply a simple mask based on outline for demonstration
        mask = np.zeros(IMAGE_SIZE, dtype=np.uint8)
        # Rescale outline coords to the IMAGE_SIZE if they aren't already
        # Assuming outline_coords are already in pixel space appropriate for IMAGE_SIZE
        if outline_coords.size > 0:
             cv2.fillPoly(mask, [outline_coords.astype(np.int32)], 255)
        ect_map = ect_map * (mask / 255.0) # Apply mask

        return ect_map

def inverse_transform_pca(synthetic_X_pca, components, mean):
    # Dummy inverse PCA transform for demonstration
    # In a real scenario, this would reconstruct the original feature vector.
    # For now, let's just return a random array of the expected shape for flattened coords
    # which is N_SAMPLES x (NUM_LANDMARKS * 2)
    num_landmarks_times_2 = 120 # Example, depends on your actual landmark data
    if synthetic_X_pca.shape[0] == 0:
        return np.array([])
    return np.random.rand(synthetic_X_pca.shape[0], num_landmarks_times_2) * 100 # Example coordinates

def generate_synthetic_pca_samples(pca_data, target_samples, k_neighbors):
    # Dummy synthetic PCA sample generation for demonstration
    # In a real scenario, this would use SMOTE.
    original_scores = pca_data['original_pca_scores']
    original_labels = pca_data['original_class_labels']

    if len(original_scores) == 0:
        return np.array([]), np.array([])

    # Simulate SMOTE by just picking random original samples
    # and slightly perturbing them.
    num_to_generate = max(0, target_samples - len(original_scores))
    if num_to_generate == 0:
        return original_scores, original_labels # No new samples needed

    synthetic_scores = []
    synthetic_labels = []

    for _ in range(num_to_generate):
        # Randomly select an existing sample
        idx = np.random.randint(0, len(original_scores))
        score = original_scores[idx]
        label = original_labels[idx]

        # Perturb the score slightly
        perturbed_score = score + np.random.normal(0, 0.1, score.shape)
        synthetic_scores.append(perturbed_score)
        synthetic_labels.append(label)

    # Combine original and synthetic for the 'return all' logic of SMOTE
    # This function is used by calculate_ect_min_max_for_dataset for its 'test' synthetic samples.
    # It should reflect the real SMOTE behavior of returning (original + synthetic).
    combined_scores = np.vstack([original_scores, np.array(synthetic_scores)])
    combined_labels = np.concatenate([original_labels, np.array(synthetic_labels)])

    return combined_scores, combined_labels


def load_pca_model_data(pca_params_file_path, pca_scores_labels_file_path, class_column, exclude_classes):
    # Dummy PCA data loading for demonstration
    # In a real scenario, this would load actual PCA components, mean, and scores.
    print(f"Loading dummy PCA data from {pca_params_file_path} and {pca_scores_labels_file_path}")
    # Simulate loading real data. For this example, let's create some dummy data.
    num_samples = 100
    num_components = 5
    num_landmarks_times_2 = 120 # Example: 60 landmarks * 2 coordinates

    # Dummy original PCA scores
    original_pca_scores = np.random.rand(num_samples, num_components) * 10 - 5

    # Dummy original class labels (e.g., 'ClassA', 'ClassB')
    unique_classes = ['ClassA', 'ClassB', 'ClassC', 'ClassD']
    original_class_labels = np.random.choice(unique_classes, num_samples)

    # Exclude some dummy classes for testing
    if exclude_classes:
        valid_indices = ~np.isin(original_class_labels, exclude_classes)
        original_pca_scores = original_pca_scores[valid_indices]
        original_class_labels = original_class_labels[valid_indices]

    # Dummy PCA components and mean
    components = np.random.rand(num_components, num_landmarks_times_2)
    mean = np.random.rand(num_landmarks_times_2)

    # Dummy original flattened coordinates (needed for ECT min/max calculation)
    # These would be the actual landmark coordinates before PCA
    original_flattened_coords = np.random.rand(original_pca_scores.shape[0], num_landmarks_times_2) * 100

    return {
        'components': components,
        'mean': mean,
        'original_pca_scores': original_pca_scores,
        'original_class_labels': original_class_labels,
        'original_flattened_coords': original_flattened_coords
    }

def calculate_ect_min_max_for_dataset(all_flattened_coords, ect_calculator, apply_random_rotation, dataset_name=""):
    """
    Calculates the global min and max ECT values across a dataset of flattened coordinates.
    This function processes the ECT maps for each coordinate set and finds the absolute min/max.
    """
    print(f"Calculating global ECT min/max for {len(all_flattened_coords)} samples in {dataset_name}...")
    min_vals = []
    max_vals = []
    
    if all_flattened_coords.size == 0:
        print("No flattened coordinates provided for ECT min/max calculation. Returning default (0, 1).")
        return 0.0, 1.0

    for i, flat_coords in enumerate(all_flattened_coords):
        # Reshape flat_coords (e.g., 60 landmarks * 2 coords = 120 elements) to (60, 2)
        # Assuming flat_coords are already in the correct format for ECT calculation,
        # often (N_landmarks, 2). If it's a flattened (N_landmarks * 2) array, reshape it.
        if flat_coords.size > 0 and flat_coords.shape[0] > 0 and flat_coords.shape[0] % 2 == 0:
            coords_2d = flat_coords.reshape(-1, 2)
            
            # Apply dummy random rotation if enabled (not used for global min/max calc usually)
            if apply_random_rotation:
                angle = random.uniform(0, 360)
                M = cv2.getRotationMatrix2D((coords_2d[:,0].mean(), coords_2d[:,1].mean()), angle, 1)
                coords_2d = cv2.transform(coords_2d.reshape(-1, 1, 2), M).reshape(-1, 2)

            ect_map = ect_calculator.calculate_ect(coords_2d)
            if ect_map.size > 0:
                min_vals.append(ect_map.min())
                max_vals.append(ect_map.max())
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(all_flattened_coords)} for ECT min/max...")

    if not min_vals or not max_vals:
        print("No valid ECT maps were generated for min/max calculation. Returning default (0, 1).")
        return 0.0, 1.0

    return np.min(min_vals), np.max(max_vals)

def generate_single_ect_shape_pair(
    flat_coords: np.ndarray,
    ect_calculator: ECT,
    ect_min_val: float,
    ect_max_val: float,
    image_size: tuple
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Generates a single 2-channel (mask, ECT) image from flattened coordinates.
    Also calculates length-to-width ratio and returns pixel coordinates.
    """
    shape_mask = np.zeros(image_size, dtype=np.float32)
    ect_mask = np.zeros(image_size, dtype=np.float32)
    lw_ratio = 0.0
    pixel_coords = np.array([])

    if flat_coords.size == 0 or flat_coords.shape[0] == 0:
        return shape_mask, ect_mask, lw_ratio, pixel_coords

    # Reshape to (N_landmarks, 2)
    coords_2d = flat_coords.reshape(-1, 2).astype(np.float32)

    # Center and scale coordinates to fit into image_size
    # This is a simplified scaling. For real data, you might need more robust normalization.
    min_x, min_y = np.min(coords_2d, axis=0)
    max_x, max_y = np.max(coords_2d, axis=0)
    
    range_x = max_x - min_x
    range_y = max_y - min_y

    if range_x == 0 or range_y == 0: # Degenerate shape
        return shape_mask, ect_mask, lw_ratio, pixel_coords

    # Determine scaling factor to fit within IMAGE_SIZE with some margin
    scale = min((image_size[0] - 20) / range_x, (image_size[1] - 20) / range_y) # 20 px margin
    
    scaled_coords = (coords_2d - np.array([min_x, min_y])) * scale

    # Translate to center in the image
    offset_x = (image_size[0] - (range_x * scale)) / 2
    offset_y = (image_size[1] - (range_y * scale)) / 2
    
    scaled_coords[:, 0] += offset_x
    scaled_coords[:, 1] += offset_y
    
    pixel_coords = scaled_coords.astype(np.int32) # For cv2.fillPoly

    if pixel_coords.shape[0] < 3: # Need at least 3 points for a polygon
        return shape_mask, ect_mask, lw_ratio, pixel_coords

    # Create shape mask
    cv2.fillPoly(shape_mask, [pixel_coords], 1.0) # Mask is 0 or 1

    # Calculate ECT map
    ect_raw = ect_calculator.calculate_ect(pixel_coords) # Use pixel_coords for ECT calc

    # Normalize ECT map using global min/max
    if ect_max_val == ect_min_val:
        ect_mask = np.zeros_like(ect_raw, dtype=np.float32)
    else:
        ect_mask = (ect_raw - ect_min_val) / (ect_max_val - ect_min_val)
    ect_mask = np.clip(ect_mask, 0.0, 1.0) # Ensure values are within [0, 1]

    # Calculate length-to-width ratio (simplified for this dummy example)
    # This would typically come from actual morphometric measurements
    lw_ratio = (range_y * scale) / (range_x * scale) if range_x > 0 else 0.0

    return shape_mask, ect_mask, lw_ratio, pixel_coords

def visualize_combined_ect_and_outline(ect_channel_image, outline_pixel_coords, title, ax, image_size):
    """
    Visualizes the ECT channel image with an outline overlay.
    Adjusted to accept single ECT channel and a single (N, 2) outline array.
    """
    ax.imshow(ect_channel_image, cmap='magma', origin='upper', vmin=0, vmax=1, aspect='auto')
    if outline_pixel_coords.size > 0:
        # Plot each segment of the outline
        # Ensure outline_pixel_coords are in the correct format for plotting (N, 2)
        ax.plot(outline_pixel_coords[:, 0], outline_pixel_coords[:, 1], color='cyan', linewidth=1)
    ax.set_title(title)
    ax.axis('off')


# --- Constants from the original script 04 ---
NUM_ECT_DIRECTIONS = 180
ECT_THRESHOLDS = np.linspace(0, 1, NUM_ECT_DIRECTIONS) # Example thresholds
BOUND_RADIUS = 1.0

# GLOBAL CONSTANTS FOR IMAGE AND PADDING (will be imported by script 07)
IMAGE_SIZE = (256, 256) # Width, Height for individual leaf image
PADDING_BETWEEN_LEAVES = 10 # Pixels of padding between individual leaf images
COMBINED_IMAGE_HEIGHT = IMAGE_SIZE[1] # For simplicity, combined image has same height as individual leaf

# Define SAMPLES_PER_CLASS_TARGET and DATASET_CONFIGS for script 07 to import
SAMPLES_PER_CLASS_TARGET = 50 # Example target for SMOTE
APPLY_RANDOM_ROTATION = False # Flag for whether to apply random rotation during ECT calculation (usually false for aligned shapes)

# Dummy DATASET_CONFIGS for demonstration
DATASET_CONFIGS = {
    "plant_predict": {
        "DATASET_FULL_NAME": "Plant Predict Collection",
        "SAVED_MODEL_DIR": Path("./03_morphometrics_output_combined"), # Dummy path for testing
        "PCA_PARAMS_FILE_NAME": "pca_model_params.pkl",
        "PCA_SCORES_LABELS_FILE_NAME": "pca_scores_labels.pkl",
        "CLASS_COLUMN": "plantID", # Assuming this is the class label column in your data
        "EXCLUDE_CLASSES": ["unlabeled_class"] # Example classes to exclude
    }
}


def generate_n_ranked_leaves_for_cnn(
    n: int,
    class_label: str, # This will be the `plantID` (decoded string)
    pca_data: dict,
    ect_calculator: ECT,
    ect_min_val: float,
    ect_max_val: float,
    real_sample_pct: float = 0.5, # Percentage of real samples to use if available
    image_size: tuple = IMAGE_SIZE,
    padding: int = PADDING_BETWEEN_LEAVES,
    verbose: bool = False, # ADDED VERBOSE PARAMETER HERE
    pre_selected_real_indices: list = None # New: Optional pre-selected indices for real samples
) -> tuple[np.ndarray, list]: # Now also returns list of pixel coords for viz
    """
    Generates a combined 2-channel (mask, ECT) image of 'n' synthetic (or real) leaves for a given class,
    ranked by their length-to-width ratio.
    Also returns the pixel coordinates of the outlines for visualization.

    Args:
        n (int): Number of leaves to combine.
        class_label (str): The plantID for which to generate leaves.
        pca_data (dict): Dictionary containing PCA model components, mean, and original PCA scores/labels.
        ect_calculator (ECT): Initialized ECT calculator instance.
        ect_min_val (float): Minimum ECT value for consistent scaling.
        ect_max_val (float): Maximum ECT value for consistent scaling.
        real_sample_pct (float): Proportion of 'n' samples that should be real if available.
        image_size (tuple): Tuple (width, height) for individual leaf images.
        padding (int): Pixels of padding between individual leaf images.
        verbose (bool): If True, print detailed generation messages.
        pre_selected_real_indices (list): If provided, use these specific indices from
                                           original_class_flattened_coords for real samples.
                                           This is crucial for reproducibility and k-fold splits.

    Returns:
        tuple[np.ndarray, list]: A tuple containing:
            - np.ndarray: A 2-channel (mask, ECT) NumPy array representing the combined n leaves,
                          ranked by length-to-width ratio, and normalized to [0, 1].
                          Shape: (height, (n * width) + (n-1)*padding, 2)
            - list: A list of pixel coordinate arrays for each leaf's outline, scaled to its
                    position within the combined image.
    """
    if n <= 0:
        if verbose:
            print(f"  Warning: n_leaves is 0 for class {class_label}. Returning empty image.")
        return np.zeros((image_size[1], 0, 2), dtype=np.float32), []

    # Filter original data for the specific class_label (plantID)
    original_class_indices = np.where(pca_data['original_class_labels'] == class_label)[0]
    original_class_pca_scores = pca_data['original_pca_scores'][original_class_indices]
    original_class_flattened_coords = pca_data['original_flattened_coords'][original_class_indices]

    leaf_data = [] # Stores (lw_ratio, shape_mask_array, ect_array, pixel_coords_for_shape_mask)

    # 1. Use Real Samples (if available and needed)
    num_real_samples_to_use = 0
    if pre_selected_real_indices is not None:
        # Use pre-selected real samples for this specific call (e.g., for training/validation sets)
        num_real_samples_to_use = len(pre_selected_real_indices)
        if verbose:
            print(f"  Using {num_real_samples_to_use} pre-selected real samples for class '{class_label}'.")
        
        for idx in pre_selected_real_indices:
            # Check if idx is within bounds of original_class_flattened_coords
            if idx < len(original_class_flattened_coords):
                flat_coords = original_class_flattened_coords[idx]
                shape_mask, ect_mask, lw_ratio, pixel_coords = generate_single_ect_shape_pair(
                    flat_coords, ect_calculator, ect_min_val, ect_max_val, image_size
                )
                if lw_ratio > 0 and pixel_coords.size > 0: # Only add valid shapes
                    leaf_data.append((lw_ratio, shape_mask, ect_mask, pixel_coords))
                elif verbose:
                    print(f"    Skipping degenerate real sample for class '{class_label}' at index {idx}.")
            elif verbose:
                print(f"    Warning: Pre-selected index {idx} out of bounds for original_class_flattened_coords (size {len(original_class_flattened_coords)}). Skipping.")
    else:
        # If no pre-selected indices, generate a mix of real/synthetic as before
        num_real_samples_to_use = min(int(n * real_sample_pct), len(original_class_flattened_coords))
        if num_real_samples_to_use > 0:
            if verbose:
                print(f"  Selecting {num_real_samples_to_use} random real samples for class '{class_label}'.")
            sampled_real_indices = np.random.choice(len(original_class_flattened_coords), num_real_samples_to_use, replace=False)
            for idx in sampled_real_indices:
                flat_coords = original_class_flattened_coords[idx]
                shape_mask, ect_mask, lw_ratio, pixel_coords = generate_single_ect_shape_pair(
                    flat_coords, ect_calculator, ect_min_val, ect_max_val, image_size
                )
                if lw_ratio > 0 and pixel_coords.size > 0:
                    leaf_data.append((lw_ratio, shape_mask, ect_mask, pixel_coords))
                elif verbose:
                    print(f"    Skipping degenerate real sample for class '{class_label}'.")

    # 2. Generate Synthetic Samples (to make up the rest or if no real samples were requested/available)
    # This logic assumes 'n' is the target total. If pre_selected_real_indices were provided,
    # then num_synthetic_samples_to_generate will be n - num_real_samples_used_from_pre_selected.
    num_synthetic_samples_to_generate = n - len(leaf_data) # Remaining needed to reach 'n'

    if num_synthetic_samples_to_generate > 0:
        if verbose:
            print(f"  Generating {num_synthetic_samples_to_generate} synthetic samples for class '{class_label}'.")
        
        # Ensure enough samples for SMOTE if it's used; otherwise, use random uniform within bounds.
        K_NEIGHBORS_SMOTE = max(1, len(original_class_pca_scores) - 1) if len(original_class_pca_scores) > 1 else 1

        if len(original_class_pca_scores) >= K_NEIGHBORS_SMOTE + 1: # SMOTE requires k_neighbors + 1 samples
            temp_pca_data_for_smote = {
                'original_pca_scores': original_class_pca_scores,
                'original_class_labels': np.array([class_label]*len(original_class_pca_scores))
            }
            # Generate more than needed, then filter for only new synthetic samples
            synthetic_X_pca, _ = generate_synthetic_pca_samples(
                temp_pca_data_for_smote,
                num_synthetic_samples_to_generate + len(original_class_pca_scores), # Target total samples
                K_NEIGHBORS_SMOTE
            )
            # Take only the newly generated ones (assuming SMOTE appends them)
            synthetic_X_pca = synthetic_X_pca[len(original_class_pca_scores):]
            
            # If SMOTE couldn't generate enough, log a warning
            if len(synthetic_X_pca) < num_synthetic_samples_to_generate:
                if verbose:
                    print(f"    Warning: SMOTE generated only {len(synthetic_X_pca)} synthetic samples, less than requested {num_synthetic_samples_to_generate}, for class '{class_label}'.")
                num_synthetic_samples_to_generate = len(synthetic_X_pca) # Adjust target
            
            synthetic_flattened_coords = inverse_transform_pca(
                synthetic_X_pca, pca_data['components'], pca_data['mean']
            )

        else:
            if len(original_class_pca_scores) == 0:
                if verbose:
                    print(f"    Warning: Class '{class_label}' has no real samples to generate synthetic from. Cannot generate synthetic data.")
                synthetic_flattened_coords = np.array([])
            else:
                if verbose:
                    print(f"    Warning: Class '{class_label}' has insufficient samples ({len(original_class_pca_scores)}) for meaningful NearestNeighbors calculation (k={K_NEIGHBORS_SMOTE}). Generating samples randomly within PCA bounds.")
                min_pc_vals = np.min(original_class_pca_scores, axis=0)
                max_pc_vals = np.max(original_class_pca_scores, axis=0)
                synthetic_X_pca = np.random.uniform(min_pc_vals, max_pc_vals, (num_synthetic_samples_to_generate, pca_data['components'].shape[1]))
                synthetic_flattened_coords = inverse_transform_pca(
                    synthetic_X_pca, pca_data['components'], pca_data['mean']
                )

        # Process the generated synthetic flattened coordinates
        if len(synthetic_flattened_coords) > 0:
            for flat_coords in synthetic_flattened_coords:
                shape_mask, ect_mask, lw_ratio, pixel_coords = generate_single_ect_shape_pair(
                    flat_coords, ect_calculator, ect_min_val, ect_max_val, image_size
                )
                if lw_ratio > 0 and pixel_coords.size > 0:
                    leaf_data.append((lw_ratio, shape_mask, ect_mask, pixel_coords))
                elif verbose:
                    print(f"    Skipping degenerate synthetic sample for class '{class_label}'.")
    
    # 3. Sort leaves by length-to-width ratio
    if verbose:
        print(f"  Sorting {len(leaf_data)} generated leaves by length-to-width ratio.")
    leaf_data.sort(key=lambda x: x[0])

    # Pad if not enough valid leaves were generated to reach 'n'
    initial_leaf_data_count = len(leaf_data)
    while len(leaf_data) < n:
        leaf_data.append((0.0, np.zeros(image_size, dtype=np.float32), np.zeros(image_size, dtype=np.float32), np.array([])))
    if initial_leaf_data_count < n and verbose:
        print(f"  Padded with {n - initial_leaf_data_count} blank leaves to reach target of {n} leaves.")

    # Take only the first 'n' leaves (after padding to ensure exactly 'n' elements)
    leaf_data = leaf_data[:n]

    # 4. Combine the 'n' individual leaf images into a single 2-channel array
    combined_width = n * image_size[0] + (n - 1) * padding
    combined_image_mask = np.zeros((COMBINED_IMAGE_HEIGHT, combined_width), dtype=np.float32)
    combined_image_ect = np.zeros((COMBINED_IMAGE_HEIGHT, combined_width), dtype=np.float32)
    
    combined_outline_pixel_coords = [] # Store pixel coords for the combined visualization

    current_x_offset = 0
    if verbose:
        print(f"  Combining {len(leaf_data)} individual leaf images into a single array (width: {combined_width}).")
    for i, (_, shape_mask, ect_mask, individual_pixel_coords) in enumerate(leaf_data):
        if shape_mask.shape != image_size or ect_mask.shape != image_size:
            if verbose:
                print(f"    Resizing individual image {i+1} due to shape mismatch: {shape_mask.shape} vs {image_size}.")
            shape_mask = cv2.resize(shape_mask, image_size, interpolation=cv2.INTER_AREA)
            ect_mask = cv2.resize(ect_mask, image_size, interpolation=cv2.INTER_AREA)

        # Handle cases where current_x_offset + image_size[0] might exceed combined_width
        end_x = min(current_x_offset + image_size[0], combined_width)
        
        # Ensure the slice of the source image is also correct
        src_width = end_x - current_x_offset
        if src_width > 0:
            combined_image_mask[:, current_x_offset : end_x] = shape_mask[:, :src_width]
            combined_image_ect[:, current_x_offset : end_x] = ect_mask[:, :src_width]
        
        # Adjust individual pixel coordinates by the current offset for combined viz
        if individual_pixel_coords.size > 0:
            offset_coords = individual_pixel_coords.copy()
            offset_coords[:, 0] += current_x_offset
            combined_outline_pixel_coords.append(offset_coords)

        current_x_offset += (image_size[0] + padding)

    final_combined_2_channel_image = np.stack([combined_image_mask, combined_image_ect], axis=-1)
    if verbose:
        print(f"  Combined image shape: {final_combined_2_channel_image.shape}")

    return final_combined_2_channel_image, combined_outline_pixel_coords

# --- Main execution block for testing/demonstration ---
if __name__ == "__main__":
    print("--- Running Synthetic Data Generation Library (Demonstration Mode) ---")

    # Initialize a single ECT calculator instance
    ect_calculator_instance = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)

    dataset_name = "plant_predict"
    config = DATASET_CONFIGS[dataset_name]

    # 1. Load PCA Data (essential for both real and synthetic data generation)
    pca_params_file_path = config['SAVED_MODEL_DIR'] / config['PCA_PARAMS_FILE_NAME']
    pca_scores_labels_file_path = config['SAVED_MODEL_DIR'] / config['PCA_SCORES_LABELS_FILE_NAME']

    # Ensure PCA output directory exists before attempting to load
    # For dummy data, we just create a dummy path
    if not config['SAVED_MODEL_DIR'].exists():
        print(f"Dummy PCA model directory not found at {config['SAVED_MODEL_DIR']}. Creating it for demonstration.")
        config['SAVED_MODEL_DIR'].mkdir(parents=True, exist_ok=True)
        # sys.exit(1) # Don't exit for dummy demo

    pca_data_for_generation = load_pca_model_data(
        pca_params_file_path, pca_scores_labels_file_path, config['CLASS_COLUMN'], config['EXCLUDE_CLASSES']
    )

    if pca_data_for_generation['original_flattened_coords'] is None:
        print(f"Error: 'original_flattened_coords' not found in PCA data. Cannot generate leaves.")
        sys.exit(1)

    # 2. Calculate global ECT Min/Max (used for consistent scaling)
    # Combine real and a set of preliminary synthetic data for robust min/max calculation
    synthetic_X_pca_for_minmax, synthetic_y_labels_for_minmax = generate_synthetic_pca_samples(
        pca_data_for_generation, SAMPLES_PER_CLASS_TARGET, 1 # k_neighbors=1 for dummy
    )
    synthetic_flattened_coords_for_minmax = inverse_transform_pca(
        synthetic_X_pca_for_minmax,
        pca_data_for_generation['components'],
        pca_data_for_generation['mean']
    )
    
    # Ensure original_flattened_coords is not empty before vstack
    if pca_data_for_generation['original_flattened_coords'].size > 0 and synthetic_flattened_coords_for_minmax.size > 0:
        combined_flattened_coords_for_minmax_calc = np.vstack([
            pca_data_for_generation['original_flattened_coords'],
            synthetic_flattened_coords_for_minmax
        ])
    elif pca_data_for_generation['original_flattened_coords'].size > 0:
        combined_flattened_coords_for_minmax_calc = pca_data_for_generation['original_flattened_coords']
    elif synthetic_flattened_coords_for_minmax.size > 0:
        combined_flattened_coords_for_minmax_calc = synthetic_flattened_coords_for_minmax
    else:
        combined_flattened_coords_for_minmax_calc = np.array([])


    dataset_ect_min, dataset_ect_max = calculate_ect_min_max_for_dataset(
        combined_flattened_coords_for_minmax_calc,
        ect_calculator_instance,
        APPLY_RANDOM_ROTATION, # Use current setting for global min/max calc
        dataset_name
    )

    # --- DEMONSTRATION OF ON-THE-FLY GENERATION ---
    print("\n--- Demonstrating on-the-fly N-leaf generation ---")

    # Pick a few example plantIDs for demonstration
    unique_plant_ids = np.unique(pca_data_for_generation['original_class_labels'])
    if len(unique_plant_ids) == 0:
        print("No unique plant IDs found for demonstration.")
        sys.exit(0)

    example_plant_id = unique_plant_ids[0] # Use the first available plantID

    # Create a directory to save demonstration images
    demo_output_dir = Path("./06_demonstration_on_the_fly_output/")
    demo_output_dir.mkdir(parents=True, exist_ok=True)

    for n_leaves in [1, 2, 5]: # Test with different numbers of leaves, keep N small for dummy data
        print(f"\nGenerating {n_leaves} leaves for plantID '{example_plant_id}'...")
        combined_image_array, combined_outline_pixel_coords = generate_n_ranked_leaves_for_cnn(
            n=n_leaves,
            class_label=example_plant_id,
            pca_data=pca_data_for_generation,
            ect_calculator=ect_calculator_instance,
            ect_min_val=dataset_ect_min,
            ect_max_val=dataset_ect_max,
            real_sample_pct=0.5, # Example: mix of real and synthetic
            verbose=True # Set verbose to True for demonstration mode
        )

        # Visualize the combined 2-channel image with outline overlay
        if combined_image_array.shape[1] > 0: # Check if width > 0 (height is fixed by COMBINED_IMAGE_HEIGHT)
            fig, ax = plt.subplots(1, 1, figsize=(combined_image_array.shape[1] / 100, combined_image_array.shape[0] / 100))

            # Stack the combined_outline_pixel_coords from the list of arrays
            all_outlines = np.vstack(combined_outline_pixel_coords) if combined_outline_pixel_coords else np.array([])

            visualize_combined_ect_and_outline(
                combined_image_array[:, :, 1], # Pass the ECT channel
                all_outlines, # Pass all outlines
                f'Combined ECT and Outline (PlantID: {example_plant_id}, N={n_leaves})',
                ax,
                image_size=(combined_image_array.shape[1], combined_image_array.shape[0]) # Use combined image dimensions
            )

            plt.tight_layout()
            demo_img_path = demo_output_dir / f"combined_viz_n{n_leaves}_plantID_{example_plant_id.replace(' ', '_')}.png"
            plt.savefig(demo_img_path)
            plt.close(fig)
            print(f"Saved combined visualization to {demo_img_path}")
        else:
            print(f"No valid combined image generated for n={n_leaves}.")

    print("\nDemonstration completed. Check the '06_demonstration_on_the_fly_output' folder for new visualizations.")
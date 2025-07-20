import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import shutil
import cv2
import os
import matplotlib.cm as cm
import h5py
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Ensure the ect library is installed and accessible
try:
    from ect import ECT, EmbeddedGraph
except ImportError:
    print("Error: The 'ect' library is not found. Please ensure it's installed and accessible.")
    print("Add its directory to PYTHONPATH or optionally install it using pip:")
    print("pip install ect-morphology")
    sys.exit(1)

############################
### CONFIGURATION (ALL PARAMETERS UP FRONT) ###
############################

# --- Input Data Configuration (from previous script's output) ---
SAVED_MODEL_DIR = Path("./03_morphometrics_output_combined/") # CORRECTED DIRECTORY
PCA_PARAMS_FILE = SAVED_MODEL_DIR / "leaf_pca_model_parameters_combined.h5" # CORRECTED FILENAME
PCA_SCORES_LABELS_FILE = SAVED_MODEL_DIR / "original_pca_scores_and_class_labels_combined.h5" # CORRECTED FILENAME

# --- Shape Information (from previous script's `NUM_LANDMARKS`) ---
# The 'original_flattened_coords' has shape (3565, 198), meaning 198 coordinates.
# Since it's flattened, this means 198 / 2 = 99 (x,y) pseudo-landmarks.
PREVIOUS_NUM_PSEUDO_LANDMARKS = 99
TOTAL_CONTOUR_COORDS = PREVIOUS_NUM_PSEUDO_LANDMARKS
FLATTENED_COORD_DIM = TOTAL_CONTOUR_COORDS * 2 # This will be 198

# --- ECT (Elliptic Contour Transform) Parameters ---
BOUND_RADIUS = 1
NUM_ECT_DIRECTIONS = 180
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS)

# --- Output Image Parameters ---
IMAGE_SIZE = (256, 256) # Output size for all generated images (masks, ECT, combined viz)
BACKGROUND_PIXEL = 0
SHAPE_PIXEL = 1
MASK_BACKGROUND_GRAY = 0
MASK_SHAPE_GRAY = 255

# --- Combined Visualization Parameters ---
OUTLINE_LINE_WIDTH = 2

# --- Augmentation and Balancing Parameters ---
# NEW: Target number of leaves for EACH plantID
TARGET_NUM_LEAVES_PER_PLANTID = 10 # Example: Aim for 10 leaves per plantID collection
K_NEIGHBORS_SMOTE = 5 # Number of nearest neighbors for SMOTE-like interpolation

# --- Random Rotation for Data Augmentation (for Procrustes-aligned data, usually False) ---
APPLY_RANDOM_ROTATION = False # <<< KEEP THIS TO FALSE FOR PROCRUSTES-ALIGNED DATA >>>
RANDOM_ROTATION_RANGE_DEG = (-180, 180) # Range of random rotation (in degrees)

# --- Output Directory Structure for Synthetic Samples ---
SYNTHETIC_DATA_OUTPUT_DIR = Path("./05_synthetic_leaf_data/")
SYNTHETIC_SHAPE_MASK_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "shape_masks"
SYNTHETIC_SHAPE_ECT_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "shape_ects"
SYNTHETIC_COMBINED_VIZ_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "combined_viz"
SYNTHETIC_METADATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"

# --- New: Consolidated Data Output for CNN Training ---
FINAL_PREPARED_DATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "final_cnn_dataset.pkl"

# Global ECT min/max will be calculated dynamically
GLOBAL_ECT_MIN = None
GLOBAL_ECT_MAX = None

###########################
### HELPER FUNCTIONS ###
###########################

def apply_transformation_with_affine_matrix(points: np.ndarray, affine_matrix: np.ndarray):
    """
    Applies a 3x3 affine matrix to a set of 2D points (N, 2) or a single point (2,).
    Returns the transformed points.
    """
    if points.size == 0:
        return np.array([])
        
    original_shape = points.shape
    if points.ndim == 1 and points.shape[0] == 2:
        points = points.reshape(1, 2) # Handle single 2D point

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Input 'points' must be a (N, 2) array or a (2,) array. Got shape: {original_shape}")

    if affine_matrix.shape != (3, 3):
        raise ValueError(f"Input 'affine_matrix' must be (3, 3). Got shape: {affine_matrix.shape}")

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    transformed_homogeneous = points_homogeneous @ affine_matrix.T
    
    # Return to original shape if single point was input
    if original_shape == (2,):
        return transformed_homogeneous[0, :2]
    return transformed_homogeneous[:, :2]

def find_robust_affine_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray):
    """
    Finds a robust affine transformation matrix between source and destination points.
    It attempts to find 3 non-collinear points for cv2.getAffineTransform.
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        if len(src_points) == 0:
            return np.eye(3) # Return identity for empty input
        raise ValueError(f"Need at least 3 points to compute affine transformation. Got {len(src_points)}.")

    chosen_src_pts = []
    chosen_dst_pts = []
    
    indices = np.arange(len(src_points))
    num_attempts = min(len(src_points) * (len(src_points) - 1) * (len(src_points) - 2) // 6, 1000) 

    for _ in range(num_attempts):
        selected_indices = np.random.choice(indices, 3, replace=False)
        p1_src, p2_src, p3_src = src_points[selected_indices]
        p1_dst, p2_dst, p3_dst = dst_points[selected_indices]
        
        area_val = (p1_src[0] * (p2_src[1] - p3_src[1]) +
                    p2_src[0] * (p3_src[1] - p1_src[1]) +
                    p3_src[0] * (p1_src[1] - p2_src[1]))
        
        if np.abs(area_val) > 1e-6: # Check if points are not collinear
            chosen_src_pts = np.float32([p1_src, p2_src, p3_src])
            chosen_dst_pts = np.float32([p1_dst, p2_dst, p3_dst])
            break
    
    if len(chosen_src_pts) < 3:
        raise ValueError("Could not find 3 non-collinear points for affine transformation. Shape is likely degenerate or a line.")

    M_2x3 = cv2.getAffineTransform(chosen_src_pts, chosen_dst_pts)
    
    if M_2x3.shape != (2, 3):
        raise ValueError(f"cv2.getAffineTransform returned a non-(2,3) matrix: {M_2x3.shape}")

    affine_matrix_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    
    return affine_matrix_3x3

def ect_coords_to_pixels(coords_ect: np.ndarray, image_size: tuple, bound_radius: float):
    """
    Transforms coordinates from ECT space (mathematical, Y-up, origin center, range [-R, R])
    to image pixel space (Y-down, origin top-left, range [0, IMAGE_SIZE]).
    """
    if len(coords_ect) == 0:
        return np.array([])
        
    display_x_conceptual = coords_ect[:, 1]
    display_y_conceptual = coords_ect[:, 0]

    scale_factor = image_size[0] / (2 * bound_radius)
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2 
    
    pixel_x = (display_x_conceptual * scale_factor + offset_x).astype(int)
    pixel_y = (-display_y_conceptual * scale_factor + offset_y).astype(int)
    
    return np.column_stack((pixel_x, pixel_y))

def save_grayscale_shape_mask(transformed_coords: np.ndarray, save_path: Path):
    """
    Saves a grayscale image representing a transformed contour/pixel set.
    The input transformed_coords are expected to be in ECT-normalized space [-R, R].
    """
    img = Image.new("L", IMAGE_SIZE, MASK_BACKGROUND_GRAY) # "L" for 8-bit pixels, black background
    draw = ImageDraw.Draw(img)

    if transformed_coords is not None and transformed_coords.size > 0:
        pixel_coords = ect_coords_to_pixels(transformed_coords, IMAGE_SIZE, BOUND_RADIUS)
        
        pixel_coords = np.clip(pixel_coords, [0, 0], [IMAGE_SIZE[0] - 1, IMAGE_SIZE[1] - 1])

        if len(pixel_coords) >= 3:
            polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
            draw.polygon(polygon_points, fill=MASK_SHAPE_GRAY)
        elif len(pixel_coords) > 0:
            for x, y in pixel_coords:
                draw.point((x, y), fill=MASK_SHAPE_GRAY)
    
    img.save(save_path)

def save_radial_ect_image(ect_result, save_path: Path, cmap_name: str = "gray", vmin: float = None, vmax: float = None):
    """
    Saves the radial ECT plot as an image with the specified colormap.
    Accepts optional vmin/vmax for consistent scaling.
    """
    if ect_result is None:
        Image.new("L", IMAGE_SIZE, 0).save(save_path)
        return

    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap=cmap_name, vmin=vmin, vmax=vmax)
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

def create_combined_viz_from_images(ect_image_path: Path, overlay_coords: np.ndarray,
                                     save_path: Path, overlay_color: tuple, overlay_alpha: float,
                                     overlay_type: str = "points", line_width: int = 1):
    """
    Creates a combined visualization by overlaying transformed elements (e.g., leaf shape)
    onto the ECT image. Overlayed elements are transformed to pixel space.
    """
    try:
        ect_img = Image.open(ect_image_path).convert("RGBA")
        img_width, img_height = ect_img.size

        composite_overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw_composite = ImageDraw.Draw(composite_overlay)

        if overlay_coords is not None and overlay_coords.size > 0:
            pixel_coords = ect_coords_to_pixels(overlay_coords, IMAGE_SIZE, BOUND_RADIUS)
            
            pixel_coords = np.clip(pixel_coords, [0, 0], [img_width - 1, img_height - 1])

            fill_color_with_alpha = (overlay_color[0], overlay_color[1], overlay_color[2], int(255 * overlay_alpha))

            if overlay_type == "mask_pixels":
                if len(pixel_coords) >= 3:
                    polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
                    draw_composite.polygon(polygon_points, outline=fill_color_with_alpha, width=line_width)
                elif len(pixel_coords) > 0:
                    for x, y in pixel_coords:
                        draw_composite.point((x, y), fill=fill_color_with_alpha)

            elif overlay_type == "points":
                point_radius = 2
                for x, y in pixel_coords:
                    draw_composite.ellipse([x - point_radius, y - point_radius,
                                            x + point_radius, y + point_radius],
                                            fill=fill_color_with_alpha)
            
        final_combined_img = Image.alpha_composite(ect_img, composite_overlay).convert("RGB")
        final_combined_img.save(save_path)

    except FileNotFoundError:
        print(f"Error: ECT image file not found at {ect_image_path}. Skipping combined visualization.")
    except Exception as e:
        print(f"Error creating combined visualization for {ect_image_path.stem}: {e}")

def rotate_coords_2d(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates 2D coordinates (Nx2 array) around the origin (0,0).
    """
    if coords.size == 0:
        return np.array([])
        
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    
    rotated_coords = coords @ rot_matrix.T
    return rotated_coords

def calculate_length_width_ratio(transformed_coords: np.ndarray) -> float:
    """
    Calculates the length-to-width ratio of a leaf from its transformed contour coordinates.
    Length: extent of 255 pixels (leaf) in the mask left-to-right.
    Width: extent of 255 pixels (leaf) in the mask bottom-to-top.
    Assumes coords are in a normalized space, like ECT space.
    """
    if transformed_coords.size == 0:
        return 0.0 # Return 0 for empty or degenerate shapes

    # Ensure coords are within expected bounds or scale them if necessary for robust calculation
    # For ECT space, they are typically centered around (0,0) and scaled to [-BOUND_RADIUS, BOUND_RADIUS]
    # We need to map them to pixel-like coordinates for bounding box calculation
    
    # Directly use min/max of the transformed coordinates
    min_x, max_x = transformed_coords[:, 0].min(), transformed_coords[:, 0].max()
    min_y, max_y = transformed_coords[:, 1].min(), transformed_coords[:, 1].max()

    length = max_x - min_x
    width = max_y - min_y

    if width == 0:
        return 0.0 # Avoid division by zero for flat shapes

    return length / width

##############################
### CORE LOGIC FUNCTIONS ###
##############################

def load_pca_model_data(pca_params_file: Path, pca_scores_labels_file: Path):
    """
    Loads PCA model parameters and original PCA scores/labels.
    Now also loads full_name_labels and plantid_labels.
    """
    pca_data = {}
    with h5py.File(pca_params_file, 'r') as f:
        pca_data['components'] = f['components'][:]
        pca_data['mean'] = f['mean'][:]
        pca_data['explained_variance'] = f['explained_variance'][:]
        pca_data['n_components'] = f.attrs['n_components']
        
    with h5py.File(pca_scores_labels_file, 'r') as f:
        pca_data['original_pca_scores'] = f['pca_scores'][:]
        
        # CORRECTED: Use 'full_name_labels' for class labels
        pca_data['original_class_labels'] = np.array([s.decode('utf-8') for s in f['full_name_labels'][:]])
        
        # CORRECTED: Use 'plantid_labels' for plant IDs
        pca_data['original_plantid_labels'] = np.array([s.decode('utf-8') for s in f['plantid_labels'][:]])

        if 'original_flattened_coords' in f:
            pca_data['original_flattened_coords'] = f['original_flattened_coords'][:]
        else:
            print("Warning: 'original_flattened_coords' not found in PCA_SCORES_LABELS_FILE. "
                  "Real samples cannot be processed directly from this file. "
                  "Ensure your 03_morphometrics_output script saves this data.")
            pca_data['original_flattened_coords'] = None
            
    print(f"Loaded PCA model parameters from {pca_params_file}.")
    print(f"Loaded original PCA scores, full_name_labels, plantid_labels, and flattened coords from {pca_scores_labels_file}.")
    return pca_data

def inverse_transform_pca(pca_scores: np.ndarray, pca_components: np.ndarray, pca_mean: np.ndarray):
    """
    Inverse transforms PCA scores back to the original flattened coordinate space.
    Assumes pca_components are (n_components, n_features) and pca_mean is (n_features,).
    """
    reconstructed_data = np.dot(pca_scores, pca_components) + pca_mean
    return reconstructed_data

def generate_synthetic_pca_samples_for_plantid(
    plant_id_pca_scores: np.ndarray,
    num_to_generate: int,
    k_neighbors: int,
    fallback_pca_scores: np.ndarray = None # For when a plantID has very few samples
) -> np.ndarray:
    """
    Generates synthetic PCA samples for a specific plantID using a SMOTE-like approach.
    Prioritizes neighbors within the same plantID. If not enough, falls back to broader set.
    """
    synthetic_samples = []
    
    if len(plant_id_pca_scores) < 2 and fallback_pca_scores is None:
        print(f"  Warning: Insufficient samples ({len(plant_id_pca_scores)}) for SMOTE-like augmentation for this plant ID. Cannot generate synthetic samples.")
        return np.array([])
    
    # Determine the set of samples to use for neighbor search
    if len(plant_id_pca_scores) >= 2:
        nn_data = plant_id_pca_scores
    elif fallback_pca_scores is not None:
        nn_data = fallback_pca_scores
        print(f"  Note: Using fallback PCA scores for neighbor search due to insufficient plant ID samples ({len(plant_id_pca_scores)}).")
    else: # Should not happen if previous checks are sound, but as a safeguard
        return np.array([])

    n_neighbors_for_nn = min(len(nn_data) - 1, k_neighbors)
    if n_neighbors_for_nn < 1:
        print(f"  Warning: Not enough distinct samples in NN data ({len(nn_data)}) to find neighbors. Cannot generate synthetic samples.")
        return np.array([])
    
    nn = NearestNeighbors(n_neighbors=n_neighbors_for_nn + 1).fit(nn_data)

    generated_count = 0
    attempts = 0
    max_attempts = num_to_generate * 5 # Prevent infinite loops for problematic cases

    while generated_count < num_to_generate and attempts < max_attempts:
        attempts += 1

        # Select a random sample from the current plant ID's leaves or fallback if very few
        if len(plant_id_pca_scores) > 0:
            idx_in_plant_samples = np.random.randint(0, len(plant_id_pca_scores))
            sample = plant_id_pca_scores[idx_in_plant_samples]
        elif len(fallback_pca_scores) > 0: # Should only be reached if plant_id_pca_scores is empty
            idx_in_fallback_samples = np.random.randint(0, len(fallback_pca_scores))
            sample = fallback_pca_scores[idx_in_fallback_samples]
        else: # No samples to base generation on
            break
        
        # Find neighbors in the `nn_data` set
        distances, indices = nn.kneighbors(sample.reshape(1, -1))
        
        # Select a random neighbor (excluding the sample itself, which is at index 0)
        # Ensure we pick from available neighbors.
        available_neighbors_indices = indices[0][1:]
        
        if len(available_neighbors_indices) == 0:
            continue # No distinct neighbors found, try another base sample

        neighbor_idx = np.random.choice(available_neighbors_indices)
        neighbor = nn_data[neighbor_idx]
        
        alpha = np.random.rand() # Random value between 0 and 1 for interpolation
        synthetic_pca_sample = sample + alpha * (neighbor - sample)
        
        synthetic_samples.append(synthetic_pca_sample)
        generated_count += 1
            
    if generated_count < num_to_generate:
        print(f"  Warning: Could only generate {generated_count} synthetic samples for this plant ID instead of {num_to_generate}.")

    return np.array(synthetic_samples)

def process_leaf_for_cnn_output(
    unique_leaf_id: str, # This is the ID for the individual leaf (e.g., real_plantID_leafIdx or synth_plantID_leafIdx)
    full_name: str, # The class label for CNN
    plant_id: str, # The plantID this leaf belongs to (e.g., 'plant001')
    flat_coords: np.ndarray,
    ect_calculator: ECT,
    output_dirs: dict,
    metadata_records: list,
    is_real_sample: bool = False,
    apply_random_rotation: bool = False,
    global_ect_min: float = None,
    global_ect_max: float = None
):
    """
    Processes a single leaf's flattened coordinates to produce masks, ECTs, and combined viz.
    Calculates and returns the length-to-width ratio for sorting.
    """
    current_metadata = {
        "unique_leaf_id": unique_leaf_id,
        "full_name": full_name,
        "plant_id": plant_id,
        "is_real": is_real_sample,
        "length_width_ratio": 0.0, # Will be calculated
        "is_processed_valid": False,
        "reason_skipped": "",
        "num_contour_coords": 0,
        "file_shape_mask": "",
        "file_shape_ect": "",
        "file_combined_viz": "",
    }

    temp_ect_inferno_path = output_dirs['combined_viz'] / f"{unique_leaf_id}_ect_inferno_temp.png"

    try:
        raw_contour_coords = flat_coords.reshape(TOTAL_CONTOUR_COORDS, 2)
        current_metadata["num_contour_coords"] = len(raw_contour_coords)

        if apply_random_rotation:
            random_angle_deg = np.random.uniform(*RANDOM_ROTATION_RANGE_DEG)
            processed_contour_coords = rotate_coords_2d(raw_contour_coords, random_angle_deg)
        else:
            processed_contour_coords = raw_contour_coords.copy()

        if len(np.unique(processed_contour_coords, axis=0)) < 3:
            raise ValueError(f"Leaf '{unique_leaf_id}' has too few distinct contour points ({len(np.unique(processed_contour_coords, axis=0))}) for ECT calculation.")
            
        G_contour = EmbeddedGraph()
        G_contour.add_cycle(processed_contour_coords)

        original_G_contour_coord_matrix = G_contour.coord_matrix.copy()

        G_contour.center_coordinates(center_type="origin")
        G_contour.transform_coordinates()
        G_contour.scale_coordinates(BOUND_RADIUS)

        if G_contour.coord_matrix.shape[0] < 3 or np.all(G_contour.coord_matrix == 0):
            raise ValueError(f"Degenerate contour shape for '{unique_leaf_id}' after ECT transformation.")

        ect_affine_matrix = find_robust_affine_transformation_matrix(original_G_contour_coord_matrix, G_contour.coord_matrix)
        
        transformed_contour_for_mask = apply_transformation_with_affine_matrix(processed_contour_coords, ect_affine_matrix)
        
        ect_result = ect_calculator.calculate(G_contour)

        # Calculate Length-to-Width Ratio
        current_metadata["length_width_ratio"] = calculate_length_width_ratio(transformed_contour_for_mask)

        shape_mask_path = output_dirs['shape_masks'] / f"{unique_leaf_id}_mask.png"
        shape_ect_path = output_dirs['shape_ects'] / f"{unique_leaf_id}_ect.png"
        combined_viz_path = output_dirs['combined_viz'] / f"{unique_leaf_id}_combined.png"

        save_grayscale_shape_mask(transformed_contour_for_mask, shape_mask_path)
        save_radial_ect_image(ect_result, shape_ect_path, cmap_name="gray", vmin=global_ect_min, vmax=global_ect_max)

        save_radial_ect_image(ect_result, temp_ect_inferno_path, cmap_name="inferno", vmin=global_ect_min, vmax=global_ect_max)
        
        create_combined_viz_from_images(
            temp_ect_inferno_path, transformed_contour_for_mask, combined_viz_path,
            overlay_color=(255, 255, 255), overlay_alpha=1.0,
            overlay_type="mask_pixels", line_width=OUTLINE_LINE_WIDTH
        )

        current_metadata["is_processed_valid"] = True
        current_metadata["file_shape_mask"] = str(shape_mask_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_shape_ect"] = str(shape_ect_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))
        current_metadata["file_combined_viz"] = str(combined_viz_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR))

    except Exception as e:
        current_metadata["reason_skipped"] = f"Processing failed: {e}"
        # print(f"Skipping leaf '{unique_leaf_id}' due to error: {e}") # Mute for less verbose output

    finally:
        metadata_records.append(current_metadata)
        if temp_ect_inferno_path.exists():
            os.remove(temp_ect_inferno_path)

def calculate_global_ect_min_max(all_flattened_coords: np.ndarray, ect_calculator: ECT, apply_random_rotation: bool):
    """
    Calculates the global minimum and maximum ECT values across all (real and synthetic) samples
    to ensure consistent scaling for all generated ECT images.
    """
    print("\n--- Calculating Global ECT Min/Max for consistent visualization ---")
    
    global_min_val = float('inf')
    global_max_val = float('-inf')
    
    num_samples = len(all_flattened_coords)
    
    for i, flat_coords in enumerate(all_flattened_coords):
        # if (i + 1) % 100 == 0 or i == num_samples - 1: # Mute for less verbose output
        #    print(f"  Calculating ECT for sample {i+1}/{num_samples}...")

        try:
            raw_contour_coords = flat_coords.reshape(TOTAL_CONTOUR_COORDS, 2)
            
            if apply_random_rotation:
                random_angle_deg = np.random.uniform(*RANDOM_ROTATION_RANGE_DEG)
                processed_contour_coords = rotate_coords_2d(raw_contour_coords, random_angle_deg)
            else:
                processed_contour_coords = raw_contour_coords.copy()

            if len(np.unique(processed_contour_coords, axis=0)) < 3:
                continue 

            G_contour = EmbeddedGraph()
            G_contour.add_cycle(processed_contour_coords)

            G_contour.center_coordinates(center_type="origin")
            G_contour.transform_coordinates()
            G_contour.scale_coordinates(BOUND_RADIUS)

            if G_contour.coord_matrix.shape[0] < 3 or np.all(G_contour.coord_matrix == 0):
                continue

            ect_result = ect_calculator.calculate(G_contour)
            
            global_min_val = min(global_min_val, ect_result.min())
            global_max_val = max(global_max_val, ect_result.max())

        except Exception as e:
            # print(f"  Error calculating ECT for sample {i} for global min/max: {e}") # Mute for less verbose output
            continue

    if global_min_val == float('inf') or global_max_val == float('-inf'):
        print("  Warning: No valid ECT values found for global min/max calculation. Setting to default [0, 1].")
        global_min_val = 0.0
        global_max_val = 1.0
    elif global_min_val == global_max_val:
        print(f"  Warning: All ECT values are identical ({global_min_val}). Adjusting max to avoid division by zero if plotting.")
        global_max_val = global_min_val + 1e-6

    print(f"  Global ECT Min: {global_min_val:.4f}, Global ECT Max: {global_max_val:.4f}")
    return global_min_val, global_max_val

def extract_unique_plant_identifier(full_name: str, plantid_label: str) -> str:
    """
    Combines the full_name and plantid_label to create a unique identifier for a plant.
    This serves as the 'plant_id' for grouping and balancing.
    Example: 'VarietyA' and 'plant001' -> 'VarietyA_plant001'
    """
    return f"{full_name}_{plantid_label}"

def main_synthetic_generation(clear_existing_data: bool = True):
    """
    Main function to orchestrate synthetic leaf data generation and processing of real data,
    balancing the number of leaves per plant ID.
    """
    print("--- Starting Leaf Data Processing and Augmentation Pipeline ---")

    # --- 1. Setup Output Directories ---
    if clear_existing_data and SYNTHETIC_DATA_OUTPUT_DIR.exists():
        print(f"Clearing existing output directory: {SYNTHETIC_DATA_OUTPUT_DIR}")
        shutil.rmtree(SYNTHETIC_DATA_OUTPUT_DIR)
        
    SYNTHETIC_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_SHAPE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_SHAPE_ECT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_COMBINED_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Created output directories.")

    # --- 2. Load PCA Data (includes original real data) ---
    pca_data = load_pca_model_data(PCA_PARAMS_FILE, PCA_SCORES_LABELS_FILE)
    
    if pca_data['original_flattened_coords'] is None:
        print("Cannot process data as 'original_flattened_coords' was not found. Exiting.")
        return

    # Create a DataFrame for easier manipulation of original data
    original_df = pd.DataFrame({
        'pca_scores': list(pca_data['original_pca_scores']),
        'full_name': pca_data['original_class_labels'],
        'plantid_label': pca_data['original_plantid_labels'], # NEW: added plantid_labels
        'flattened_coords': list(pca_data['original_flattened_coords']),
        'is_real': True
    })
    # Create the unique plant_id for grouping
    original_df['plant_id'] = original_df.apply(lambda row: extract_unique_plant_identifier(row['full_name'], row['plantid_label']), axis=1)


    # --- 3. Balance Leaves Per Plant ID (Upsampling & Downsampling) ---
    print(f"\n--- Balancing leaves to {TARGET_NUM_LEAVES_PER_PLANTID} per plant ID ---")
    
    all_processed_leaves_metadata = []
    all_flattened_coords_for_global_ect = [] # To collect all coords for global ECT min/max calc

    # Group original leaves by plant ID
    plant_id_groups = original_df.groupby('plant_id')

    # Collect synthetic and real leaves to be processed
    leaves_to_process = [] # Will store (pca_score, full_name, plant_id, flattened_coords, is_real)

    synth_leaf_counter = 0

    for plant_id, group_df in plant_id_groups:
        full_name = group_df['full_name'].iloc[0] # Assume same full_name for all leaves in a plantID
        real_leaves_for_plant = group_df.copy().reset_index(drop=True)
        num_real_leaves = len(real_leaves_for_plant)
        
        print(f"  Plant ID '{plant_id}' (Class: {full_name}) has {num_real_leaves} real leaves.")

        if num_real_leaves > TARGET_NUM_LEAVES_PER_PLANTID:
            # Downsample real leaves
            print(f"    Downsampling to {TARGET_NUM_LEAVES_PER_PLANTID} real leaves.")
            sampled_leaves = real_leaves_for_plant.sample(n=TARGET_NUM_LEAVES_PER_PLANTID, random_state=42)
            for _, row in sampled_leaves.iterrows():
                leaves_to_process.append({
                    'pca_score': row['pca_scores'],
                    'full_name': row['full_name'],
                    'plant_id': row['plant_id'],
                    'flattened_coords': row['flattened_coords'],
                    'is_real': True
                })
                all_flattened_coords_for_global_ect.append(row['flattened_coords'])

        elif num_real_leaves < TARGET_NUM_LEAVES_PER_PLANTID:
            # Upsample by adding synthetic leaves
            num_to_generate = TARGET_NUM_LEAVES_PER_PLANTID - num_real_leaves
            print(f"    Upsampling: Generating {num_to_generate} synthetic leaves.")

            # Add all existing real leaves for this plant ID
            for _, row in real_leaves_for_plant.iterrows():
                leaves_to_process.append({
                    'pca_score': row['pca_scores'],
                    'full_name': row['full_name'],
                    'plant_id': row['plant_id'],
                    'flattened_coords': row['flattened_coords'],
                    'is_real': True
                })
                all_flattened_coords_for_global_ect.append(row['flattened_coords'])

            # Generate synthetic leaves
            plant_pca_scores = np.array(list(real_leaves_for_plant['pca_scores']))
            # Fallback to class-wide PCA scores if current plant ID has too few samples for neighbors
            class_pca_scores = pca_data['original_pca_scores'][pca_data['original_class_labels'] == full_name]

            synthetic_pca_scores = generate_synthetic_pca_samples_for_plantid(
                plant_pca_scores,
                num_to_generate,
                K_NEIGHBORS_SMOTE,
                fallback_pca_scores=class_pca_scores if len(plant_pca_scores) < 2 else None
            )

            if synthetic_pca_scores.size > 0:
                synthetic_flattened_coords = inverse_transform_pca(
                    synthetic_pca_scores, pca_data['components'], pca_data['mean']
                )
                for i, synth_coords in enumerate(synthetic_flattened_coords):
                    leaves_to_process.append({
                        'pca_score': synthetic_pca_scores[i],
                        'full_name': full_name,
                        'plant_id': plant_id,
                        'flattened_coords': synth_coords,
                        'is_real': False
                    })
                    all_flattened_coords_for_global_ect.append(synth_coords)
                    synth_leaf_counter += 1

        else: # num_real_leaves == TARGET_NUM_LEAVES_PER_PLANTID
            print(f"    Already has {TARGET_NUM_LEAVES_PER_PLANTID} leaves. No change.")
            for _, row in real_leaves_for_plant.iterrows():
                leaves_to_process.append({
                    'pca_score': row['pca_scores'],
                    'full_name': row['full_name'],
                    'plant_id': row['plant_id'],
                    'flattened_coords': row['flattened_coords'],
                    'is_real': True
                })
                all_flattened_coords_for_global_ect.append(row['flattened_coords'])

    print(f"\nTotal synthetic leaves generated across all plant IDs: {synth_leaf_counter}")
    
    # --- 4. Initialize ECT Calculator ---
    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)
    print("Initialized ECT calculator.")

    # --- 5. Calculate Global ECT Min/Max ---
    global GLOBAL_ECT_MIN, GLOBAL_ECT_MAX
    # Convert list of arrays to single numpy array for calculation
    all_flattened_coords_np = np.array(all_flattened_coords_for_global_ect)
    GLOBAL_ECT_MIN, GLOBAL_ECT_MAX = calculate_global_ect_min_max(all_flattened_coords_np, ect_calculator, APPLY_RANDOM_ROTATION)
    
    # --- 6. Process All Individual Leaves (Real and Synthetic) ---
    print("\n--- Processing all individual leaves (real and synthetic) for image generation ---")
    processed_leaves_metadata = []
    
    # Assign unique IDs to each leaf as they are processed
    # This ensures unique filenames for all individual mask/ECT images
    unique_leaf_processing_id_counter = 0

    for leaf_data in leaves_to_process:
        is_real = leaf_data['is_real']
        id_prefix = "real" if is_real else "synth"
        # Generate a unique ID that includes plant_id for easy lookup
        # Replace non-filename friendly characters in plant_id for safety
        unique_leaf_id = f"{id_prefix}_{leaf_data['plant_id'].replace('_', '-')}_{unique_leaf_processing_id_counter:06d}"
        unique_leaf_processing_id_counter += 1

        # print(f"Processing {id_prefix} leaf ({unique_leaf_id}, Plant ID: {leaf_data['plant_id']}, Class: {leaf_data['full_name']})")
        
        process_leaf_for_cnn_output(
            unique_leaf_id,
            leaf_data['full_name'],
            leaf_data['plant_id'],
            leaf_data['flattened_coords'],
            ect_calculator,
            {
                'shape_masks': SYNTHETIC_SHAPE_MASK_DIR,
                'shape_ects': SYNTHETIC_SHAPE_ECT_DIR,
                'combined_viz': SYNTHETIC_COMBINED_VIZ_DIR,
            },
            processed_leaves_metadata,
            is_real_sample=is_real,
            apply_random_rotation=APPLY_RANDOM_ROTATION,
            global_ect_min=GLOBAL_ECT_MIN,
            global_ect_max=GLOBAL_ECT_MAX
        )

    # Convert to DataFrame for easier manipulation
    final_processed_leaves_df = pd.DataFrame(processed_leaves_metadata)
    final_processed_leaves_df.to_csv(SYNTHETIC_METADATA_FILE, index=False)
    print(f"\nSaved combined real and synthetic leaf metadata to {SYNTHETIC_METADATA_FILE}")

    # Filter for successfully processed samples (important before CNN data prep)
    valid_processed_leaves_df = final_processed_leaves_df[final_processed_leaves_df['is_processed_valid']].copy()
    if valid_processed_leaves_df.empty:
        print("No valid leaves processed to create the final CNN dataset. Exiting.")
        return

    # --- 7. Prepare and Save Consolidated Data for CNN Training (Plant ID Collections) ---
    print("\n--- Consolidating data for CNN training (stacking leaves by plant ID) ---")

    cnn_X_images = [] # This will hold 4D tensors: (Num_PlantIDs, H, W, 2 * num_leaves)
    cnn_y_labels_raw = [] # Full names for each plant ID
    cnn_plant_ids = [] # Store the plant IDs themselves

    # Group valid processed leaves by plant ID
    # Use .copy() to avoid SettingWithCopyWarning if modifying later
    # Sort first by length_width_ratio before grouping to ensure consistent order within each group
    valid_plant_id_groups = valid_processed_leaves_df.sort_values(by='length_width_ratio').groupby('plant_id')

    num_plant_ids_processed_for_cnn = 0

    for plant_id, group_df in valid_plant_id_groups:
        # Ensure each plant ID group has exactly TARGET_NUM_LEAVES_PER_PLANTID leaves
        # This handles cases where some leaves for a plant ID failed processing
        if len(group_df) != TARGET_NUM_LEAVES_PER_PLANTID:
            print(f"  Warning: Plant ID '{plant_id}' has {len(group_df)} valid leaves instead of target {TARGET_NUM_LEAVES_PER_PLANTID}. Skipping this plant ID for CNN dataset.")
            continue
        
        individual_leaf_images_for_stacking = []
        for idx, row in group_df.iterrows():
            mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_shape_mask']
            ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_shape_ect']

            try:
                mask_img = Image.open(mask_path).convert('L')
                mask_array = np.array(mask_img, dtype=np.float32) / 255.0

                ect_img = Image.open(ect_path).convert('L')
                ect_array = np.array(ect_img, dtype=np.float32) / 255.0

                combined_image_2_channel = np.stack([mask_array, ect_array], axis=-1) # (H, W, 2)
                individual_leaf_images_for_stacking.append(combined_image_2_channel)

            except FileNotFoundError:
                print(f"  Critical Error: Image file not found for {row['unique_leaf_id']}. This plant ID will be incomplete. Skipping.")
                individual_leaf_images_for_stacking = [] # Clear and break to skip this plant ID
                break
            except Exception as e:
                print(f"  Error loading or processing images for {row['unique_leaf_id']}: {e}. This plant ID will be incomplete. Skipping.")
                individual_leaf_images_for_stacking = []
                break

        if len(individual_leaf_images_for_stacking) == TARGET_NUM_LEAVES_PER_PLANTID:
            # Stack along the channel dimension: (H, W, 2 * TARGET_NUM_LEAVES_PER_PLANTID)
            stacked_plant_image = np.concatenate(individual_leaf_images_for_stacking, axis=-1)
            cnn_X_images.append(stacked_plant_image)
            cnn_y_labels_raw.append(group_df['full_name'].iloc[0]) # Get full_name for the plant ID
            cnn_plant_ids.append(plant_id)
            num_plant_ids_processed_for_cnn += 1
        else:
            print(f"  Skipped Plant ID '{plant_id}' due to insufficient or problematic leaf images after processing.")


    if not cnn_X_images:
        print("No plant ID collections were successfully loaded and prepared for CNN training. The final dataset will be empty.")
        return

    cnn_X_images_np = np.array(cnn_X_images)
    cnn_y_labels_np = np.array(cnn_y_labels_raw)
    cnn_plant_ids_np = np.array(cnn_plant_ids)

    # Encode class labels to integers
    label_encoder = LabelEncoder()
    cnn_y_encoded = label_encoder.fit_transform(cnn_y_labels_np)
    class_names = label_encoder.classes_

    # Shuffle the dataset consistently
    cnn_X_images_shuffled, cnn_y_encoded_shuffled, cnn_plant_ids_shuffled = shuffle(
        cnn_X_images_np, cnn_y_encoded, cnn_plant_ids_np, random_state=42
    )

    # Save the consolidated data
    final_data = {
        'X_images': cnn_X_images_shuffled, # Shape: (Num_PlantIDs, H, W, 2 * TARGET_NUM_LEAVES_PER_PLANTID)
        'y_labels_encoded': cnn_y_encoded_shuffled,
        'class_names': class_names,
        'plant_ids': cnn_plant_ids_shuffled,
        'image_size': IMAGE_SIZE,
        'num_channels_per_leaf': 2,
        'target_leaves_per_plantid': TARGET_NUM_LEAVES_PER_PLANTID
    }

    with open(FINAL_PREPARED_DATA_FILE, 'wb') as f:
        pickle.dump(final_data, f)
    print(f"Successfully prepared and saved CNN training data to {FINAL_PREPARED_DATA_FILE}")
    print(f"Final CNN Dataset shape: X_images={cnn_X_images_shuffled.shape}, y_labels={cnn_y_encoded_shuffled.shape}")
    print(f"Number of plant ID collections for CNN training: {num_plant_ids_processed_for_cnn}")
    print(f"Class names: {class_names}")

    print("\n--- Leaf Data Processing and Augmentation Pipeline Completed ---")

if __name__ == "__main__":
    main_synthetic_generation(clear_existing_data=True)
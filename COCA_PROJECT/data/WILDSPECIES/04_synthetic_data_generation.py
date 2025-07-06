#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import shutil
import cv2
import os
import matplotlib.cm as cm # For colormaps
import h5py
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder # Added for encoding class labels if needed
from sklearn.utils import shuffle # For consistent shuffling of multiple arrays

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
SAVED_MODEL_DIR = Path("./03_morphometrics_output_streamlined/") # Path to the directory where PCA model and scores were saved
PCA_PARAMS_FILE = SAVED_MODEL_DIR / "leaf_pca_model_parameters_streamlined.h5"
PCA_SCORES_LABELS_FILE = SAVED_MODEL_DIR / "original_pca_scores_and_class_labels_streamlined.h5" # Updated filename to match previous output

# --- Shape Information (from previous script's `NUM_LANDMARKS`) ---
# This defines the number of 2D points that constitute a single leaf shape.
# This should match the `NUM_LANDMARKS` from your previous morphometrics script.
# For example, if FINAL_PSEUDO_LANDMARKS_PER_SIDE was 50, then NUM_LANDMARKS = (50*2)-1 = 99.
PREVIOUS_NUM_PSEUDO_LANDMARKS = 99 # IMPORTANT: Adjust this if your previous script's NUM_LANDMARKS changes!
TOTAL_CONTOUR_COORDS = PREVIOUS_NUM_PSEUDO_LANDMARKS # Represents the number of (x,y) points for the single leaf contour
FLATTENED_COORD_DIM = TOTAL_CONTOUR_COORDS * 2 # Total number of flattened dimensions (e.g., 99 * 2 = 198)

# --- ECT (Elliptic Contour Transform) Parameters ---
BOUND_RADIUS = 1 # The radius of the bounding circle for ECT normalization
NUM_ECT_DIRECTIONS = 180 # Number of radial directions for ECT calculation
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS) # Distance thresholds for ECT calculation

# --- Output Image Parameters ---
IMAGE_SIZE = (256, 256) # Output size for all generated images (masks, ECT, combined viz)

# Pixel values for masks (consistent with user's previous code)
BACKGROUND_PIXEL = 0
SHAPE_PIXEL = 1 # Renamed from BLADE_PIXEL to be more general for the single shape

# Grayscale values for output mask file
MASK_BACKGROUND_GRAY = 0    # Black background
MASK_SHAPE_GRAY = 255       # White shape (e.g., leaf blade)

# --- Combined Visualization Parameters ---
OUTLINE_LINE_WIDTH = 2 # Line width for the leaf outline in combined_viz images

# --- SMOTE-like Augmentation Parameters ---
SAMPLES_PER_CLASS_TARGET = 400 # Desired number of synthetic samples for EACH class
K_NEIGHBORS_SMOTE = 5 # Number of nearest neighbors to consider for SMOTE interpolation

# --- Random Rotation for Data Augmentation ---
# NEW PARAMETER: Set to True to apply random rotation, False to skip it.
APPLY_RANDOM_ROTATION = False # <<< SET THIS TO FALSE FOR PROCRUSTES-ALIGNED DATA >>>
RANDOM_ROTATION_RANGE_DEG = (-180, 180) # Range of random rotation (in degrees) to apply to generated shapes

# --- Output Directory Structure for Synthetic Samples ---
SYNTHETIC_DATA_OUTPUT_DIR = Path("./05_synthetic_leaf_data/")
SYNTHETIC_SHAPE_MASK_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "shape_masks" # Contains binary masks of augmented shapes
SYNTHETIC_SHAPE_ECT_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "shape_ects" # Contains grayscale ECT images of augmented shapes
SYNTHETIC_COMBINED_VIZ_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "combined_viz" # Contains combined visualizations for checking alignment

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
    Splies a 3x3 affine matrix to a set of 2D points (N, 2) or a single point (2,).
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
    # Limit attempts for very large point sets to prevent infinite loops on degenerate shapes
    num_attempts = min(len(src_points) * (len(src_points) - 1) * (len(src_points) - 2) // 6, 1000) 

    for _ in range(num_attempts):
        selected_indices = np.random.choice(indices, 3, replace=False)
        p1_src, p2_src, p3_src = src_points[selected_indices]
        p1_dst, p2_dst, p3_dst = dst_points[selected_indices]
        
        # Check for collinearity by calculating area of triangle formed by points
        # Area formula: 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
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
        
    # In ECT visualization, the x-axis corresponds to the horizontal dimension and y-axis to vertical.
    # When plotting, we typically map ECT-x to image-x, and ECT-y to image-y.
    # However, the ECT visualization usually has the Y-axis increasing upwards (mathematical convention).
    # Image pixel coordinates have Y-axis increasing downwards.
    # The original code's choice was `display_x_conceptual = coords_ect[:, 1]` and `display_y_conceptual = coords_ect[:, 0]`.
    # This implies coords_ect is (y, x) or it's just how the transformation was empirically found to work.
    # We will retain this mapping to ensure consistency with the combined visualization.
    display_x_conceptual = coords_ect[:, 1]
    display_y_conceptual = coords_ect[:, 0]

    scale_factor = image_size[0] / (2 * bound_radius)
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2 # Using IMAGE_SIZE[1] for y-offset

    pixel_x = (display_x_conceptual * scale_factor + offset_x).astype(int)
    pixel_y = (-display_y_conceptual * scale_factor + offset_y).astype(int) # Negate Y for image coordinates (Y-down)
    
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
        
        # Ensure pixel_coords are within image bounds to prevent errors
        pixel_coords = np.clip(pixel_coords, [0, 0], [IMAGE_SIZE[0] - 1, IMAGE_SIZE[1] - 1])

        if len(pixel_coords) >= 3:
            # Draw a filled polygon for closed shapes (leaf outlines)
            polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
            draw.polygon(polygon_points, fill=MASK_SHAPE_GRAY)
        elif len(pixel_coords) > 0:
            # Draw individual points for lines or sparse point sets (e.g., if only 1-2 points are left)
            for x, y in pixel_coords:
                draw.point((x, y), fill=MASK_SHAPE_GRAY)
    
    img.save(save_path)

def save_radial_ect_image(ect_result, save_path: Path, cmap_name: str = "gray", vmin: float = None, vmax: float = None):
    """
    Saves the radial ECT plot as an image with the specified colormap.
    Accepts optional vmin/vmax for consistent scaling.
    """
    if ect_result is None:
        # Create a blank black image if ECT result is None (e.g., degenerate shape)
        Image.new("L", IMAGE_SIZE, 0).save(save_path)
        return

    # Create a figure and polar axes
    # figsize and dpi are set to match IMAGE_SIZE for direct pixel correspondence
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100) # Use 100 DPI to convert inches to pixels directly
    
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    
    # Use pcolormesh to plot the ECT data, applying global vmin/vmax
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap=cmap_name, vmin=vmin, vmax=vmax) # ect_result.T for correct orientation
    
    # Configure polar plot
    ax.set_theta_zero_location("N") # 0 degrees at the top
    ax.set_theta_direction(-1) # Clockwise direction for angles
    ax.set_rlim([0, BOUND_RADIUS]) # Radius limits
    ax.axis('off') # Hide axes for clean image output
    
    # Remove whitespace around the plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100) # Save tightly with no padding
    plt.close(fig) # Close the figure to free memory

def create_combined_viz_from_images(ect_image_path: Path, overlay_coords: np.ndarray,
                                    save_path: Path, overlay_color: tuple, overlay_alpha: float,
                                    overlay_type: str = "points", line_width: int = 1):
    """
    Creates a combined visualization by overlaying transformed elements (e.g., leaf shape)
    onto the ECT image. Overlayed elements are transformed to pixel space.
    """
    try:
        ect_img = Image.open(ect_image_path).convert("RGBA") # Open ECT image and convert to RGBA
        img_width, img_height = ect_img.size

        # Create a transparent overlay image
        composite_overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw_composite = ImageDraw.Draw(composite_overlay)

        if overlay_coords is not None and overlay_coords.size > 0:
            # Transform overlay coordinates to pixel space
            pixel_coords = ect_coords_to_pixels(overlay_coords, IMAGE_SIZE, BOUND_RADIUS)
            
            # Ensure pixel_coords are within image bounds
            pixel_coords = np.clip(pixel_coords, [0, 0], [img_width - 1, img_height - 1])

            # Prepare overlay color with transparency
            fill_color_with_alpha = (overlay_color[0], overlay_color[1], overlay_color[2], int(255 * overlay_alpha))

            if overlay_type == "mask_pixels":
                if len(pixel_coords) >= 3:
                    polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
                    # Draw the outline instead of filling the polygon, using the specified line_width
                    draw_composite.polygon(polygon_points, outline=fill_color_with_alpha, width=line_width)
                elif len(pixel_coords) > 0:
                    for x, y in pixel_coords:
                        draw_composite.point((x, y), fill=fill_color_with_alpha)

            elif overlay_type == "points":
                point_radius = 2 # Fixed radius for individual points
                for x, y in pixel_coords:
                    draw_composite.ellipse([x - point_radius, y - point_radius,
                                            x + point_radius, y + point_radius],
                                            fill=fill_color_with_alpha)
            
        # Composite the ECT image with the overlay
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

##############################
### CORE LOGIC FUNCTIONS ###
##############################

def load_pca_model_data(pca_params_file: Path, pca_scores_labels_file: Path):
    """
    Loads PCA model parameters and original PCA scores/labels.
    """
    pca_data = {}
    with h5py.File(pca_params_file, 'r') as f:
        pca_data['components'] = f['components'][:]
        pca_data['mean'] = f['mean'][:]
        pca_data['explained_variance'] = f['explained_variance'][:]
        pca_data['n_components'] = f.attrs['n_components']
        
    with h5py.File(pca_scores_labels_file, 'r') as f:
        pca_data['original_pca_scores'] = f['pca_scores'][:]
        # Ensure we're loading the 'class_labels' key, consistent with the previous script's output
        pca_data['original_class_labels'] = np.array([s.decode('utf-8') for s in f['class_labels'][:]])
        # Also load the original flattened coordinates if available (they should be)
        if 'original_flattened_coords' in f:
            pca_data['original_flattened_coords'] = f['original_flattened_coords'][:]
        else:
            print("Warning: 'original_flattened_coords' not found in PCA_SCORES_LABELS_FILE. "
                  "Real samples cannot be processed directly from this file. "
                  "Ensure your 03_morphometrics_output script saves this data.")
            pca_data['original_flattened_coords'] = None
            
    print(f"Loaded PCA model parameters from {pca_params_file}.")
    print(f"Loaded original PCA scores and labels from {pca_scores_labels_file}.")
    return pca_data

def generate_synthetic_pca_samples(pca_data: dict, samples_per_class_target: int, k_neighbors: int):
    """
    Generates synthetic PCA samples using a SMOTE-like approach based on class labels.
    """
    print(f"\nStarting synthetic data generation (SMOTE-like) with {samples_per_class_target} samples per class...")
    
    original_pca_scores = pca_data['original_pca_scores']
    original_class_labels = pd.Series(pca_data['original_class_labels']) # Use original_class_labels
    
    synthetic_X_pca = []
    synthetic_y = []
    
    class_counts = original_class_labels.value_counts()
    all_classes = class_counts.index.tolist()
    
    total_generated_samples = 0

    for class_name in all_classes:
        class_pca_samples = original_pca_scores[original_class_labels == class_name]
        
        # Ensure enough samples for NearestNeighbors for this class
        if len(class_pca_samples) < 2:
            print(f"Warning: Class '{class_name}' has less than 2 samples ({len(class_pca_samples)}). Skipping SMOTE-like augmentation for this class.")
            # If we want to include original samples for small classes, we could add them here.
            # For now, we skip generating *synthetic* ones via interpolation.
            continue
            
        n_neighbors_for_class = min(len(class_pca_samples) - 1, k_neighbors)
        if n_neighbors_for_class < 1:
            print(f"Warning: Class '{class_name}' has insufficient samples ({len(class_pca_samples)}) for meaningful NearestNeighbors calculation (k={k_neighbors}). Skipping.")
            continue

        nn = NearestNeighbors(n_neighbors=n_neighbors_for_class + 1).fit(class_pca_samples)
        
        generated_count = 0
        while generated_count < samples_per_class_target:
            idx_in_class_samples = np.random.randint(0, len(class_pca_samples))
            sample = class_pca_samples[idx_in_class_samples]
            
            distances, indices = nn.kneighbors(sample.reshape(1, -1))
            
            # Select a random neighbor (excluding the sample itself, which is at index 0)
            available_neighbors_indices_in_class_pca = indices[0][1:]
            
            if len(available_neighbors_indices_in_class_pca) == 0:
                # This can happen if k_neighbors is too high for the actual number of distinct points
                # or if a point is isolated. Simply try another sample point.
                continue
                
            neighbor_idx_in_class_pca_samples = np.random.choice(available_neighbors_indices_in_class_pca)
            neighbor = class_pca_samples[neighbor_idx_in_class_pca_samples]
            
            alpha = np.random.rand() # Random value between 0 and 1 for interpolation
            synthetic_pca_sample = sample + alpha * (neighbor - sample)
            
            synthetic_X_pca.append(synthetic_pca_sample)
            synthetic_y.append(class_name)
            generated_count += 1
            total_generated_samples += 1
            
    print(f"Finished generating {total_generated_samples} synthetic samples across {len(all_classes)} classes.")
    return np.array(synthetic_X_pca), synthetic_y

def inverse_transform_pca(pca_scores: np.ndarray, pca_components: np.ndarray, pca_mean: np.ndarray):
    """
    Inverse transforms PCA scores back to the original flattened coordinate space.
    Assumes pca_components are (n_components, n_features) and pca_mean is (n_features,).
    """
    reconstructed_data = np.dot(pca_scores, pca_components) + pca_mean
    return reconstructed_data

def process_leaf_for_cnn_output(
    sample_id: str,
    class_label: str,
    flat_coords: np.ndarray,
    ect_calculator: ECT,
    output_dirs: dict,
    metadata_records: list,
    is_real_sample: bool = False,
    apply_random_rotation: bool = False,
    global_ect_min: float = None, # NEW PARAMETER
    global_ect_max: float = None  # NEW PARAMETER
):
    """
    Processes a single leaf's flattened coordinates to produce masks, ECTs, and combined viz.
    This version focuses solely on a single 'shape' (the full leaf contour).
    Includes an option to apply random rotation and uses global ECT vmin/vmax for consistent plotting.
    """
    current_metadata = {
        "synthetic_id": sample_id,
        "class_label": class_label,
        "is_processed_valid": False,
        "reason_skipped": "",
        "num_contour_coords": 0,
        "file_shape_mask": "",
        "file_shape_ect": "",
        "file_combined_viz": "",
        "is_real": is_real_sample
    }

    temp_ect_inferno_path = output_dirs['combined_viz'] / f"{sample_id}_ect_inferno_temp.png"

    try:
        raw_contour_coords = flat_coords.reshape(TOTAL_CONTOUR_COORDS, 2)
        current_metadata["num_contour_coords"] = len(raw_contour_coords)

        # --- Apply Random Rotation if enabled ---
        if apply_random_rotation:
            random_angle_deg = np.random.uniform(*RANDOM_ROTATION_RANGE_DEG)
            processed_contour_coords = rotate_coords_2d(raw_contour_coords, random_angle_deg)
            # print(f"  Applied random rotation of {random_angle_deg:.2f} degrees to {sample_id}.") # Muted for less verbosity
        else:
            processed_contour_coords = raw_contour_coords.copy() # Use original (aligned) coords
            # print(f"  Skipping random rotation for {sample_id} (Procrustes alignment maintained).") # Muted for less verbosity
        # --- END Random Rotation ---

        # --- Validate Contour Coordinates for ECT ---
        if len(np.unique(processed_contour_coords, axis=0)) < 3:
            raise ValueError(f"Leaf '{sample_id}' has too few distinct contour points ({len(np.unique(processed_contour_coords, axis=0))}) for ECT calculation.")
            
        # --- Process Contour: Calculate its ECT and derive its transformation matrix ---
        G_contour = EmbeddedGraph()
        G_contour.add_cycle(processed_contour_coords)

        original_G_contour_coord_matrix = G_contour.coord_matrix.copy()

        # Perform ECT's internal normalization: centering, transformation (Procrustes-like), and scaling
        G_contour.center_coordinates(center_type="origin")
        G_contour.transform_coordinates()
        G_contour.scale_coordinates(BOUND_RADIUS)

        if G_contour.coord_matrix.shape[0] < 3 or np.all(G_contour.coord_matrix == 0):
            raise ValueError(f"Degenerate contour shape for '{sample_id}' after ECT transformation.")

        ect_affine_matrix = find_robust_affine_transformation_matrix(original_G_contour_coord_matrix, G_contour.coord_matrix)
        
        transformed_contour_for_mask = apply_transformation_with_affine_matrix(processed_contour_coords, ect_affine_matrix)
        
        ect_result = ect_calculator.calculate(G_contour)

        # --- Define Output Paths ---
        shape_mask_path = output_dirs['shape_masks'] / f"{sample_id}_mask.png"
        shape_ect_path = output_dirs['shape_ects'] / f"{sample_id}_ect.png"
        combined_viz_path = output_dirs['combined_viz'] / f"{sample_id}_combined.png"

        # --- Save CNN Input/Output Files (Masks & Grayscale ECTs) ---
        save_grayscale_shape_mask(transformed_contour_for_mask, shape_mask_path)
        # Pass global_ect_min and global_ect_max here for consistent scaling
        save_radial_ect_image(ect_result, shape_ect_path, cmap_name="gray", vmin=global_ect_min, vmax=global_ect_max)

        # --- Create Combined Visualization for Verification ---
        # Also pass global_ect_min and global_ect_max for consistent scaling in combined viz
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
        print(f"Skipping leaf '{sample_id}' due to error: {e}")

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
        if (i + 1) % 100 == 0 or i == num_samples - 1:
            print(f"  Calculating ECT for sample {i+1}/{num_samples}...")

        try:
            raw_contour_coords = flat_coords.reshape(TOTAL_CONTOUR_COORDS, 2)
            
            if apply_random_rotation:
                random_angle_deg = np.random.uniform(*RANDOM_ROTATION_RANGE_DEG)
                processed_contour_coords = rotate_coords_2d(raw_contour_coords, random_angle_deg)
            else:
                processed_contour_coords = raw_contour_coords.copy()

            if len(np.unique(processed_contour_coords, axis=0)) < 3:
                # print(f"  Skipping ECT min/max for sample {i} due to too few distinct contour points.")
                continue # Skip degenerate shapes for global min/max calculation

            G_contour = EmbeddedGraph()
            G_contour.add_cycle(processed_contour_coords)

            G_contour.center_coordinates(center_type="origin")
            G_contour.transform_coordinates()
            G_contour.scale_coordinates(BOUND_RADIUS)

            if G_contour.coord_matrix.shape[0] < 3 or np.all(G_contour.coord_matrix == 0):
                # print(f"  Skipping ECT min/max for sample {i} due to degenerate shape after transformation.")
                continue

            ect_result = ect_calculator.calculate(G_contour)
            
            # Update global min/max
            global_min_val = min(global_min_val, ect_result.min())
            global_max_val = max(global_max_val, ect_result.max())

        except Exception as e:
            # print(f"  Error calculating ECT for sample {i} for global min/max: {e}")
            continue # Skip samples that cause errors

    # Handle cases where no valid ECTs were found, or all values are the same (e.g., all zeros)
    if global_min_val == float('inf') or global_max_val == float('-inf'):
        print("  Warning: No valid ECT values found for global min/max calculation. Setting to default [0, 1].")
        global_min_val = 0.0
        global_max_val = 1.0
    elif global_min_val == global_max_val: # If all ECT values are identical (e.g., all 0)
        print(f"  Warning: All ECT values are identical ({global_min_val}). Adjusting max to avoid division by zero if plotting.")
        global_max_val = global_min_val + 1e-6 # Add a small epsilon to avoid issues if plotted

    print(f"  Global ECT Min: {global_min_val:.4f}, Global ECT Max: {global_max_val:.4f}")
    return global_min_val, global_max_val


def main_synthetic_generation(clear_existing_data: bool = True):
    """
    Main function to orchestrate synthetic leaf data generation AND processing of real data.
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
        print("Cannot process real data as 'original_flattened_coords' was not found. Exiting.")
        return

    # --- 3. Generate Synthetic PCA Samples First ---
    # We generate these first so we have all coordinates (real + synthetic) to calculate global min/max
    synthetic_X_pca, synthetic_y_labels = generate_synthetic_pca_samples(
        pca_data, SAMPLES_PER_CLASS_TARGET, K_NEIGHBORS_SMOTE
    )
    
    # Inverse transform ALL synthetic PCA scores to get flattened coordinates
    synthetic_flattened_coords = inverse_transform_pca(
        synthetic_X_pca, pca_data['components'], pca_data['mean']
    )

    # --- 4. Consolidate ALL Flattened Coordinates for Global ECT Min/Max Calculation ---
    all_flattened_coords = np.vstack([pca_data['original_flattened_coords'], synthetic_flattened_coords])
    
    # --- 5. Initialize ECT Calculator ---
    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)
    print("Initialized ECT calculator.")

    # --- 6. Calculate Global ECT Min/Max (NEW STEP) ---
    global GLOBAL_ECT_MIN, GLOBAL_ECT_MAX # Access global variables
    GLOBAL_ECT_MIN, GLOBAL_ECT_MAX = calculate_global_ect_min_max(all_flattened_coords, ect_calculator, APPLY_RANDOM_ROTATION)
    
    metadata_records = []
    
    # --- 7. Process Original Real Samples (now with global ECT min/max) ---
    print("\n--- Processing Original Real Leaf Samples ---")
    num_real_samples = len(pca_data['original_flattened_coords'])
    for i in range(num_real_samples):
        sample_id = f"real_leaf_{i:05d}"
        class_label = pca_data['original_class_labels'][i]
        flat_coords = pca_data['original_flattened_coords'][i]

        print(f"Processing real leaf {i+1}/{num_real_samples} ({sample_id}, Class: {class_label})")

        process_leaf_for_cnn_output(
            sample_id,
            class_label,
            flat_coords,
            ect_calculator,
            {
                'shape_masks': SYNTHETIC_SHAPE_MASK_DIR,
                'shape_ects': SYNTHETIC_SHAPE_ECT_DIR,
                'combined_viz': SYNTHETIC_COMBINED_VIZ_DIR,
            },
            metadata_records,
            is_real_sample=True,
            apply_random_rotation=APPLY_RANDOM_ROTATION,
            global_ect_min=GLOBAL_ECT_MIN, # Pass global min
            global_ect_max=GLOBAL_ECT_MAX  # Pass global max
        )

    # --- 8. Process Synthetic PCA Samples (now with global ECT min/max) ---
    total_synthetic_samples = len(synthetic_flattened_coords)
    print(f"\n--- Processing {total_synthetic_samples} Synthetic Leaf Samples ---")

    for i in range(total_synthetic_samples):
        sample_id = f"synthetic_leaf_{i:05d}"
        class_label = synthetic_y_labels[i] # Use labels from generation step
        flat_coords = synthetic_flattened_coords[i] # Use inverse transformed coords

        print(f"Processing synthetic leaf {i+1}/{total_synthetic_samples} ({sample_id}, Class: {class_label})")
            
        process_leaf_for_cnn_output(
            sample_id,
            class_label,
            flat_coords,
            ect_calculator,
            {
                'shape_masks': SYNTHETIC_SHAPE_MASK_DIR,
                'shape_ects': SYNTHETIC_SHAPE_ECT_DIR,
                'combined_viz': SYNTHETIC_COMBINED_VIZ_DIR,
            },
            metadata_records,
            is_real_sample=False,
            apply_random_rotation=APPLY_RANDOM_ROTATION,
            global_ect_min=GLOBAL_ECT_MIN, # Pass global min
            global_ect_max=GLOBAL_ECT_MAX  # Pass global max
        )

    # --- 9. Save Metadata ---
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(SYNTHETIC_METADATA_FILE, index=False)
    print(f"\nSaved combined real and synthetic leaf metadata to {SYNTHETIC_METADATA_FILE}")

    # --- 10. Prepare and Save Consolidated Data for CNN Training ---
    print("\n--- Consolidating data for CNN training ---")

    # Filter for successfully processed samples
    valid_samples_df = metadata_df[metadata_df['is_processed_valid']].copy()
    if valid_samples_df.empty:
        print("No valid samples processed to create the final CNN dataset. Exiting.")
        return

    # Initialize lists to hold image data and labels
    X_images = [] # This will hold 2-channel images
    y_labels_raw = [] # This will hold string labels
    is_real_flags = [] # To keep track of real vs. synthetic samples

    # Loop to load and stack images
    for idx, row in valid_samples_df.iterrows():
        mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_shape_mask']
        ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row['file_shape_ect']

        try:
            # Load mask image (should be 8-bit grayscale)
            mask_img = Image.open(mask_path).convert('L')
            mask_array = np.array(mask_img, dtype=np.float32) / 255.0 # Normalize to [0, 1]

            # Load ECT image (should be 8-bit grayscale)
            ect_img = Image.open(ect_path).convert('L')
            ect_array = np.array(ect_img, dtype=np.float32) / 255.0 # Normalize to [0, 1]

            # Stack them to create a 2-channel image: (Height, Width, Channels)
            # Channel 0: Mask, Channel 1: ECT
            combined_image = np.stack([mask_array, ect_array], axis=-1)

            X_images.append(combined_image)
            y_labels_raw.append(row['class_label'])
            is_real_flags.append(row['is_real'])

        except FileNotFoundError:
            print(f"Warning: Missing image file for {row['synthetic_id']} (mask: {mask_path}, ect: {ect_path}). Skipping.")
        except Exception as e:
            print(f"Error loading or processing images for {row['synthetic_id']}: {e}. Skipping.")

    if not X_images:
        print("No images were successfully loaded and prepared for CNN training. The final dataset will be empty.")
        return

    X_images_np = np.array(X_images)
    y_labels_np = np.array(y_labels_raw)
    is_real_flags_np = np.array(is_real_flags)

    # Encode class labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels_np)
    class_names = label_encoder.classes_

    # Shuffle the dataset consistently
    X_images_shuffled, y_encoded_shuffled, is_real_flags_shuffled = shuffle(
        X_images_np, y_encoded, is_real_flags_np, random_state=42
    )

    # Save the consolidated data
    final_data = {
        'X_images': X_images_shuffled,
        'y_labels_encoded': y_encoded_shuffled,
        'class_names': class_names,
        'is_real_flags': is_real_flags_shuffled,
        'image_size': IMAGE_SIZE,
        'num_channels': X_images_shuffled.shape[-1]
    }

    with open(FINAL_PREPARED_DATA_FILE, 'wb') as f:
        pickle.dump(final_data, f)
    print(f"Successfully prepared and saved CNN training data to {FINAL_PREPARED_DATA_FILE}")
    print(f"Dataset shape: X_images={X_images_shuffled.shape}, y_labels={y_encoded_shuffled.shape}")
    print(f"Class names: {class_names}")

    print("\n--- Leaf Data Processing and Augmentation Pipeline Completed ---")

if __name__ == "__main__":
    main_synthetic_generation(clear_existing_data=True)
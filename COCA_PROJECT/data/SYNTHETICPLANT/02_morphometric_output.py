import cv2
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import h5py
import os
from pathlib import Path
# Removed: from sklearn.preprocessing import StandardScaler # No longer needed based on user feedback

#################
### FUNCTIONS ###
#################

def angle_between(p1, p2, p3):
    """
    define a function to find the angle between 3 points anti-clockwise in degrees, p2 being the vertex
    inputs: three angle points, as tuples
    output: angle in degrees
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + math.degrees(math.atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + math.degrees(math.atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def rotate_points(xvals, yvals, degrees):
    """"
    define a function to rotate 2D x and y coordinate points around the origin
    inputs: x and y vals (can take pandas dataframe columns) and the degrees (positive, anticlockwise) to rotate
    outputs: rotated and y vals
    """
    angle_to_move = 90 - degrees
    rads = np.deg2rad(angle_to_move)

    new_xvals = xvals * np.cos(rads) - yvals * np.sin(rads)
    new_yvals = xvals * np.sin(rads) + yvals * np.cos(rads)

    return new_xvals, new_yvals

def interpolation(x, y, number):
    """
    define a function to return equally spaced, interpolated points for a given polyline
    inputs: arrays of x and y values for a polyline, number of points to interpolate
    ouputs: interpolated points along the polyline, inclusive of start and end points
    """
    if len(x) < 2 or len(y) < 2:
        if np.all(x == x[0]) and np.all(y == y[0]):
            return np.full(number, x[0]), np.full(number, y[0])
        elif len(x) == 1: # Handle single point case by repeating it
            return np.full(number, x[0]), np.full(number, y[0])
        else: # Unhandled short polyline, return empty or raise error
            raise ValueError("Polyline too short for interpolation and not a single point.")

    # Calculate cumulative distance along the polyline
    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))

    # Handle case where all points are identical (total distance is 0)
    if distance[-1] == 0:
        return np.full(number, x[0]), np.full(number, y[0])

    # Normalize distance to be between 0 and 1
    distance = distance / distance[-1]

    # Create interpolation functions
    fx, fy = interp1d(distance, x), interp1d(distance, y)

    # Generate equally spaced points along the normalized distance
    alpha = np.linspace(0, 1, number)
    x_regular, y_regular = fx(alpha), fy(alpha)

    return x_regular, y_regular

def euclid_dist(x1, y1, x2, y2):
    """
    define a function to return the euclidean distance between two points
    inputs: x and y values of the two points
    output: the eulidean distance
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def poly_area(x, y):
    """
    define a function to calculate the area of a polygon using the shoelace algorithm
    inputs: separate numpy arrays of x and y coordinate values
    outputs: the area of the polygon
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def gpa_mean(leaf_arr, landmark_num, dim_num):
    """
    define a function that given an array of landmark data returns the Generalized Procrustes Analysis mean
    inputs: a 3 dimensional array of samples by landmarks by coordinate values, number of landmarks, number of dimensions
    output: an array of the Generalized Procrustes Analysis mean shape
    """
    if leaf_arr.shape[0] == 0:
        return np.zeros((landmark_num, dim_num)) # Return an empty mean if no shapes

    ref_ind = 0
    ref_shape = leaf_arr[ref_ind, :, :]
    mean_diff = 10**(-6) # Increased for robustness
    old_mean = ref_shape
    d = 1000000

    # Ensure there are enough samples for meaningful Procrustes analysis
    if len(leaf_arr) < 2:
        return leaf_arr[0] # If only one sample, that is the mean.

    while d > mean_diff:
        arr = np.zeros(((len(leaf_arr)), landmark_num, dim_num))
        for i in range(len(leaf_arr)):
            # Handle potential scaling/translation issues if shapes are too similar or degenerate
            try:
                s1, s2, distance = procrustes(old_mean, leaf_arr[i])
                arr[i] = s2
            except ValueError:
                # If procrustes fails (e.g., degenerate input), just use the original or previous iteration's mean
                arr[i] = leaf_arr[i] # Fallback: use original shape, might need more robust handling
                print(f"Warning: Procrustes failed for sample {i}. Using original shape.")
        new_mean = np.mean(arr, axis=(0))
        s1, s2, d = procrustes(old_mean, new_mean)
        old_mean = new_mean
    return new_mean

def run_morphometric_analysis(metadata_df, image_data_base_dir, output_base_dir, analysis_name="combined_dataset"):
    """
    Runs the full morphometric analysis pipeline for a given dataset (or combined dataset).

    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata for the dataset(s) to analyze.
                                     Expected to have 'dataset' column to determine image path.
        image_data_base_dir (Path): Base directory containing the 'CULTIVATED1ST' and 'CULTIVATED2ND'
                                     image data folders (e.g., '../').
        output_base_dir (Path): Base directory where all outputs for this analysis will be saved.
        analysis_name (str): A descriptive name for this analysis (e.g., "combined_dataset")
                             used in print statements and specific output filenames.
    """
    print(f"\n{'='*10} Starting Morphometric Analysis for {analysis_name.upper()} Dataset {'='*10}")

    # --- Configuration and Inputs ---
    # Parameters for Preprocessing
    HIGH_RES_INTERPOLATION_POINTS = 10000
    FINAL_PSEUDO_LANDMARKS_PER_SIDE = 50
    NUM_LANDMARKS = (FINAL_PSEUDO_LANDMARKS_PER_SIDE * 2) - 1
    NUM_DIMENSIONS = 2

    # Parameters for Output Files
    GPA_MEAN_SHAPE_PLOT_FILENAME = f"gpa_mean_shape_{analysis_name}.png"
    PCA_EXPLAINED_VARIANCE_REPORT_FILENAME = f"pca_explained_variance_{analysis_name}.txt"
    PCA_PARAMS_H5_FILENAME = f"leaf_pca_model_parameters_{analysis_name}.h5"
    ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME = f"original_pca_scores_and_class_labels_{analysis_name}.h5"
    
    # Ensure output directory exists
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to directory: {output_base_dir}")

    # --- Read in Metadata (now passed as DataFrame) ---
    mdata = metadata_df.copy() # Work on a copy to avoid modifying the original DataFrame
    print(f"Metadata DataFrame received for {analysis_name}. Number of rows: {len(mdata)}")
    if mdata.empty:
        print(f"No data for {analysis_name}. Skipping analysis.")
        return

    # --- Verify essential columns ---
    required_cols_for_preprocessing = ["file", "dataset", "base_x", "base_y", "tip_x", "tip_y", "plantID", "full_name"]
    if not all(col in mdata.columns for col in required_cols_for_preprocessing):
        print(f"ERROR: Missing one or more required columns ({required_cols_for_preprocessing}) in metadata for {analysis_name}.")
        print("Please ensure '01_plant_landmarks.csv' contains these columns by running the corrected '00_metadata.py' first.")
        return # Exit the function if critical data is missing

    # --- Print plantID counts and structure ---
    print("\n--- Plant ID Class Information ---")
    if 'plantID' in mdata.columns and 'full_name' in mdata.columns:
        print(f"Number of different plantID classes for each full_name (variety) in {analysis_name}:")
        plant_id_counts_per_variety = mdata.groupby('full_name')['plantID'].nunique()
        print(plant_id_counts_per_variety.sort_index().to_string())

        print(f"\nTotal number of unique plantID classes in {analysis_name}: {mdata['plantID'].nunique()}")
        print(f"\nCounts of each plantID class in {analysis_name}:")
        plant_id_counts_overall = mdata['plantID'].value_counts().sort_index()
        print(plant_id_counts_overall.to_string())
    else:
        print("Warning: 'plantID' or 'full_name' column not found in metadata. Cannot print class information.")
    print("-----------------------------------")

    # --- Interpolate Points Creating Pseudo-Landmarks and Pre-process ---
    print("\n--- Preprocessing Images and Interpolating Pseudo-Landmarks ---")

    processed_points_list = []
    valid_rows_indices = []

    # Create a mapping for image data directories
    image_dirs = {
        'first': image_data_base_dir / 'CULTIVATED1ST' / '00_cultivated1st_data',
        'second': image_data_base_dir / 'CULTIVATED2ND' / '00_cultivated2nd_data'
    }

    # Pre-load actual image files for validation for each dataset
    actual_image_files_by_dataset = {}
    for ds, path in image_dirs.items():
        if path.exists():
            actual_image_files_by_dataset[ds] = set(f.name for f in path.iterdir() if f.is_file())
        else:
            print(f"Warning: Image directory {path} for dataset '{ds}' not found. No images from this dataset will be processed.")
            actual_image_files_by_dataset[ds] = set()

    for lf_idx, row in mdata.iterrows():
        curr_image_filename = row["file"]
        dataset_type = row["dataset"] # 'first' or 'second'

        if dataset_type not in image_dirs or curr_image_filename not in actual_image_files_by_dataset.get(dataset_type, set()):
            # print(f"Warning: Image file '{curr_image_filename}' from dataset '{dataset_type}' not found or directory missing. Skipping.")
            continue # Skip this row if image file or its directory doesn't exist

        img_path = image_dirs[dataset_type] / curr_image_filename

        try:
            # Read image and find contours
            img = cv2.bitwise_not(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)) # Use str(Path) for cv2
            
            # Save the original ECT image shape if this is the first image processed
            # We'll assume all ECT images have the same shape.
            if 'original_ect_image_shape' not in locals(): # Check if already defined
                original_ect_image_shape = img.shape
                print(f"Detected original ECT image shape: {original_ect_image_shape}")

            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            x_conts = []
            y_conts = []
            areas_conts = []
            for c in contours:
                x_vals = [pt[0][0] for pt in c]
                y_vals = [pt[0][1] for pt in c]
                # Filter out contours with too few points to prevent interpolation errors
                if len(x_vals) < 2 or len(y_vals) < 2:
                    continue
                area = (max(x_vals) - min(x_vals)) * (max(y_vals) - min(y_vals)) # Simple bounding box area
                x_conts.append(x_vals)
                y_conts.append(y_vals)
                areas_conts.append(area)

            if not areas_conts:
                # print(f"Warning: No valid contours found for image {curr_image_filename}. Skipping.")
                continue # Skip if no valid contours

            area_inds = np.flip(np.argsort(areas_conts))
            # Only consider the largest contour (index 0 after sorting)
            sorted_x_conts = np.array(x_conts, dtype=object)[area_inds][0]
            sorted_y_conts = np.array(y_conts, dtype=object)[area_inds][0]

            # Interpolate high-resolution contour
            high_res_x, high_res_y = interpolation(np.array(sorted_x_conts, dtype=np.float32),
                                                   np.array(sorted_y_conts, dtype=np.float32), HIGH_RES_INTERPOLATION_POINTS)

            # Extract base and tip points from metadata
            base_pt = np.array((row["base_x"], row["base_y"]))
            tip_pt = np.array((row["tip_x"], row["tip_y"]))

            # Find closest points on the contour to the base and tip landmarks
            base_dists = [euclid_dist(base_pt[0], base_pt[1], high_res_x[pt], high_res_y[pt]) for pt in range(len(high_res_x))]
            tip_dists = [euclid_dist(tip_pt[0], tip_pt[1], high_res_x[pt], high_res_y[pt]) for pt in range(len(high_res_x))]

            base_ind = np.argmin(base_dists)
            tip_ind = np.argmin(tip_dists)

            # Re-order contour starting from base_ind
            high_res_x_ordered = np.concatenate((high_res_x[base_ind:], high_res_x[:base_ind]))
            high_res_y_ordered = np.concatenate((high_res_y[base_ind:], high_res_y[:base_ind]))

            # Find new tip_ind in the re-ordered contour
            new_tip_dists = [euclid_dist(tip_pt[0], tip_pt[1], high_res_x_ordered[pt_idx], high_res_y_ordered[pt_idx]) for pt_idx in range(len(high_res_x_ordered))]
            tip_ind_new = np.argmin(new_tip_dists)

            lf_contour = np.column_stack((high_res_x_ordered, high_res_y_ordered))

            # Split into left and right segments
            if tip_ind_new >= 0 and tip_ind_new < len(lf_contour):
                left_segment = lf_contour[0:tip_ind_new + 1, :]
                right_segment_part1 = lf_contour[tip_ind_new:, :]
                right_segment_part2 = lf_contour[0:1, :] # Connect back to the start point (base)
                right_segment = np.concatenate((right_segment_part1, right_segment_part2), axis=0)
            else:
                print(f"Warning: Tip index out of bounds for image {curr_image_filename}. Skipping.")
                continue # Skip if tip_ind_new is somehow invalid

            if len(left_segment) < 2 or len(right_segment) < 2:
                # print(f"Warning: Segments for image {curr_image_filename} are too short for interpolation. Skipping.")
                continue

            left_inter_x, left_inter_y = interpolation(left_segment[:, 0], left_segment[:, 1], FINAL_PSEUDO_LANDMARKS_PER_SIDE)
            right_inter_x, right_inter_y = interpolation(right_segment[:, 0], right_segment[:, 1], FINAL_PSEUDO_LANDMARKS_PER_SIDE)

            # Remove duplicate end point from left segment (which is the start of right segment)
            left_inter_x = np.delete(left_inter_x, -1)
            left_inter_y = np.delete(left_inter_y, -1)

            lf_pts_left = np.column_stack((left_inter_x, left_inter_y))
            lf_pts_right = np.column_stack((right_inter_x, right_inter_y))
            lf_pts = np.row_stack((lf_pts_left, lf_pts_right))

            if lf_pts.shape[0] != NUM_LANDMARKS:
                print(f"Warning: Leaf {curr_image_filename} generated {lf_pts.shape[0]} landmarks, expected {NUM_LANDMARKS}. Check interpolation logic.")
                continue

            # Orientation and rotation
            tip_point = lf_pts[FINAL_PSEUDO_LANDMARKS_PER_SIDE - 1, :]
            base_point = lf_pts[0, :]
            ang = angle_between(tip_point, base_point, (base_point[0] + 1, base_point[1]))

            rot_x, rot_y = rotate_points(lf_pts[:, 0], lf_pts[:, 1], ang)
            rot_pts = np.column_stack((rot_x, rot_y))

            processed_points_list.append(rot_pts)
            valid_rows_indices.append(lf_idx)

        except Exception as e:
            print(f"Error processing image {curr_image_filename}: {e}. Skipping.")
            continue # Skip on any processing error

    # Rebuild mdata with only successfully processed images
    mdata = mdata.iloc[valid_rows_indices].reset_index(drop=True)
    cult_cm_arr = np.array(processed_points_list)

    if cult_cm_arr.shape[0] == 0:
        print(f"No valid images processed for {analysis_name}. Exiting analysis.")
        return

    print(f"Successfully processed {cult_cm_arr.shape[0]} images for {analysis_name}.")

    # --- Calculate GPA Mean ---
    print("--- Calculating GPA Mean ---")
    mean_shape = gpa_mean(cult_cm_arr, NUM_LANDMARKS, NUM_DIMENSIONS)

    # --- Align Leaves to GPA Mean ---
    print("--- Aligning Leaves to GPA Mean ---")
    proc_arr = np.zeros(np.shape(cult_cm_arr))
    # Additionally, calculate length and width for each aligned leaf
    leaf_lengths = []
    leaf_widths = []

    for i in range(len(cult_cm_arr)):
        s1, s2, distance = procrustes(mean_shape, cult_cm_arr[i, :, :])
        proc_arr[i] = s2

        # Calculate Length and Width from Procrustes-aligned shape
        length = np.max(s2[:, 1]) - np.min(s2[:, 1]) # Max Y - Min Y
        width = np.max(s2[:, 0]) - np.min(s2[:, 0]) # Max X - Min X
        leaf_lengths.append(length)
        leaf_widths.append(width)

    # Add length and width to the metadata DataFrame
    mdata['aligned_length'] = leaf_lengths
    mdata['aligned_width'] = leaf_widths
    mdata['length_to_width_ratio'] = mdata['aligned_length'] / mdata['aligned_width']
    print("Calculated aligned length, width, and length-to-width ratio for each leaf.")

    # --- Visualize GPA Aligned Shapes and Mean ---
    print("--- Visualizing GPA Aligned Shapes ---")
    plt.figure(figsize=(8, 8))
    for i in range(len(proc_arr)):
        plt.plot(proc_arr[i, :, 0], proc_arr[i, :, 1], c="k", alpha=0.08)
    plt.plot(np.mean(proc_arr, axis=0)[:, 0], np.mean(proc_arr, axis=0)[:, 1], c="magenta")
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.title(f"Procrustes Aligned Leaf Shapes and GPA Mean ({analysis_name.replace('_', ' ').title()})")
    plt.savefig(output_base_dir / GPA_MEAN_SHAPE_PLOT_FILENAME)
    plt.close()
    print(f"GPA mean shape plot saved to {output_base_dir / GPA_MEAN_SHAPE_PLOT_FILENAME}")

    # --- Perform Full PCA --- # MODIFIED: Removed StandardScaler application
    print("\n--- Performing Full PCA ---")
    flat_arr = proc_arr.reshape(np.shape(proc_arr)[0], np.shape(proc_arr)[1] * np.shape(proc_arr)[2])

    max_pc_components = min(flat_arr.shape[0], flat_arr.shape[1])
    pca = PCA(n_components=max_pc_components)
    PCs = pca.fit_transform(flat_arr) # PCA applied directly to Procrustes-aligned data

    pca_explained_variance_filepath = output_base_dir / PCA_EXPLAINED_VARIANCE_REPORT_FILENAME
    with open(pca_explained_variance_filepath, 'w') as f:
        f.write(f"PCA Explained Variance Report ({analysis_name.replace('_', ' ').title()} Dataset):\n")
        f.write(f"Total Samples: {flat_arr.shape[0]}\n")
        f.write(f"Total Features (landmarks * dimensions): {flat_arr.shape[1]}\n")
        f.write(f"Number of PCs Calculated: {pca.n_components_}\n\n")
        f.write("PC: var, overall\n")
        for i in range(len(pca.explained_variance_ratio_)):
            pc_variance = round(pca.explained_variance_ratio_[i] * 100, 2)
            cumulative_variance = round(pca.explained_variance_ratio_.cumsum()[i] * 100, 2)
            line = f"PC{i+1}: {pc_variance}%, {cumulative_variance}%\n"
            print(line.strip())
            f.write(line)
    print(f"PCA explained variance report saved to {pca_explained_variance_filepath}")

    # --- Save PCA Model Parameters, PC Scores, and Class Labels ---
    print("\n--- Saving PCA model parameters, PC scores, and class labels ---")
    pca_components = pca.components_
    pca_mean = pca.mean_
    pca_explained_variance = pca.explained_variance_
    pca_explained_variance_ratio = pca.explained_variance_ratio_
    n_pca_components = pca.n_components_

    print(f"  PCA Components shape: {pca_components.shape}")
    print(f"  PCA Mean shape: {pca_mean.shape}")
    print(f"  PCA Explained Variance shape: {pca_explained_variance.shape}")
    print(f"  PCA Explained Variance Ratio shape: {pca_explained_variance_ratio.shape}")
    print(f"  Number of PCA components: {n_pca_components}")
    print(f"  Original PCA Scores (PCs) shape: {PCs.shape}")
    # Confirming print statements for both labels
    print(f"  PlantID Labels length: {len(mdata['plantID'])}")
    print(f"  Full_Name Labels length: {len(mdata['full_name'])}")


    pca_params_filepath = output_base_dir / PCA_PARAMS_H5_FILENAME
    with h5py.File(pca_params_filepath, 'w') as f:
        f.create_dataset('components', data=pca_components, compression="gzip")
        f.create_dataset('mean', data=pca_mean, compression="gzip")
        f.create_dataset('explained_variance', data=pca_explained_variance, compression="gzip")
        f.create_dataset('explained_variance_ratio', data=pca_explained_variance_ratio, compression="gzip")
        f.attrs['n_components'] = n_pca_components
        
        # ADDED: Save original_ect_image_shape
        # This assumes all ECT images have the same shape. We grab the first one we process.
        if 'original_ect_image_shape' in locals():
            f.create_dataset('original_ect_image_shape', data=original_ect_image_shape, compression="gzip")
        else:
            print("Warning: original_ect_image_shape was not determined. Ensure images are processed correctly.")

    print(f"PCA parameters AND original_ect_image_shape saved to {pca_params_filepath}") # Updated print statement

    pca_scores_labels_filepath = output_base_dir / ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME
    with h5py.File(pca_scores_labels_filepath, 'w') as f:
        f.create_dataset('pca_scores', data=PCs, compression="gzip")
        # Save plantID labels
        f.create_dataset('plantid_labels', data=np.array(mdata['plantID']).astype('S'), compression="gzip")
        # Save full_name labels (the true classification labels)
        f.create_dataset('full_name_labels', data=np.array(mdata['full_name']).astype('S'), compression="gzip")
        f.create_dataset('original_flattened_coords', data=flat_arr, compression="gzip") # Save the UN-scaled flattened coords
        # Save the new morphological traits (length, width, ratio)
        f.attrs['morphometric_columns'] = ['aligned_length', 'aligned_width', 'length_to_width_ratio']
        f.create_dataset('morphometric_traits', data=mdata[['aligned_length', 'aligned_width', 'length_to_width_ratio']].values, compression="gzip")

    print(f"Original PCA scores, plantID labels, full_name labels, original flattened coordinates, AND derived morphometric traits saved to {pca_scores_labels_filepath}")

    print(f"\n{'='*10} Analysis for {analysis_name.upper()} Dataset Completed {'='*10}")


# --- Main execution block ---
if __name__ == "__main__":
    # Path to the combined metadata file (generated by 00_metadata.py)
    COMBINED_METADATA_FILE = Path("./01_plant_landmarks.csv")

    # This is the base directory where CULTIVATED1ST and CULTIVATED2ND folders reside.
    # If this script is in COCA_PROJECT/data/SYNTHETICPLANT, then '..' takes it to COCA_PROJECT/data/
    IMAGE_DATA_BASE_DIR = Path("../")

    # Output directory for the combined morphometrics
    # This will create a single output folder for all results
    OUTPUT_COMBINED_DIR = Path("./03_morphometrics_output_combined/")

    # Load the combined metadata
    try:
        combined_mdata = pd.read_csv(COMBINED_METADATA_FILE)
        print(f"Loaded combined metadata from: {COMBINED_METADATA_FILE}")
    except FileNotFoundError:
        print(f"Error: Combined metadata file not found at {COMBINED_METADATA_FILE}. Please run 00_metadata.py first.")
        exit()

    # Ensure the 'dataset' column is present and valid
    if 'dataset' not in combined_mdata.columns or not combined_mdata['dataset'].isin(['first', 'second']).all():
        print("Error: 'dataset' column is missing or contains unexpected values in 01_plant_landmarks.csv. Please check 00_metadata.py output.")
        exit()
    
    # Also ensure plantID and full_name are present as they are now explicitly saved
    if 'plantID' not in combined_mdata.columns:
        print("Error: 'plantID' column is missing in 01_plant_landmarks.csv. Ensure 00_metadata.py creates this column.")
        exit()
    if 'full_name' not in combined_mdata.columns:
        print("Error: 'full_name' column is missing in 01_plant_landmarks.csv. Ensure 00_metadata.py creates this column.")
        exit()


    # Run analysis for the combined dataset
    run_morphometric_analysis(
        metadata_df=combined_mdata, # Pass the entire combined DataFrame
        image_data_base_dir=IMAGE_DATA_BASE_DIR,
        output_base_dir=OUTPUT_COMBINED_DIR,
        analysis_name="combined" # Name the output files and folders accordingly
    )

    print("\nAll morphometric analyses completed and outputs saved to a single combined dataset.")
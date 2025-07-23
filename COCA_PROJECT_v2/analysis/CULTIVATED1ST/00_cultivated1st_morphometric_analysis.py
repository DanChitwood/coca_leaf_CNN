#######################
### LOAD IN MODULES ###
#######################

import cv2 # to install on mac: pip install opencv-python
from scipy.interpolate import interp1d # for interpolating points
from sklearn.decomposition import PCA # for principal component analysis
from scipy.spatial import procrustes # for Procrustes analysis
from scipy.spatial import ConvexHull # for convex hull (not used in provided code yet)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # for LDA (not used yet)
from sklearn.metrics import confusion_matrix # for confusion matrix (not used yet)
import scipy.stats as stats # for kruskal wallis test (not used yet)
import statsmodels.stats.multitest as multitest # multiple test adjustment (not used yet)
import itertools # for pairwise combinations (not used yet)
from os import listdir # for retrieving files from directory
from os.path import isfile, join # for retrieving files from directory
import matplotlib.pyplot as plt # for plotting
import numpy as np # for using arrays
import math # for mathematical operations
import pandas as pd # for using pandas dataframes
import seaborn as sns # for plotting in seaborn
from matplotlib.colors import LogNorm # for log scale (not used yet)
import phate # for using PHATE (not used yet)
import scprep # for using PHATE (not not used yet)
import h5py # For saving large arrays and PCA model parameters
import pickle # For saving Python objects (not strictly needed now, as leaf_indices is removed)
import os # For path operations and directory creation

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
    angle_to_move = 90-degrees
    rads = np.deg2rad(angle_to_move)

    new_xvals = xvals*np.cos(rads)-yvals*np.sin(rads)
    new_yvals = xvals*np.sin(rads)+yvals*np.cos(rads)

    return new_xvals, new_yvals

def interpolation(x, y, number):
    """
    define a function to return equally spaced, interpolated points for a given polyline
    inputs: arrays of x and y values for a polyline, number of points to interpolate
    ouputs: interpolated points along the polyline, inclusive of start and end points
    """
    # Check if x or y are empty or have single point, which would cause issues with ediff1d or division by zero.
    if len(x) < 2 or len(y) < 2:
        # Handle cases where segments are too short
        # For a minimum of 2 points, interpolation can work but distance[ -1] might be 0 if points are identical.
        # If points are identical, distance will be all zeros.
        if np.all(x == x[0]) and np.all(y == y[0]): # all points are identical
            # If all points are identical, return the same point 'number' times
            return np.full(number, x[0]), np.full(number, y[0])
        elif len(x) == 1: # Single point, replicate it
            return np.full(number, x[0]), np.full(number, y[0])
        else: # Likely a segment with just two points, where the distance calculation might still be problematic if they are identical.
            # If `distance[-1]` is zero, it means start and end points are identical.
            # This should ideally be caught by len(x) < 2 if it's truly a single point.
            # For two points, `ediff1d` works.
            pass # Continue with normal interpolation, it should handle 2 points

    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    
    # Check if the total distance is zero (e.g., all points are identical) to prevent division by zero
    if distance[-1] == 0:
        # If all points are identical, just return 'number' copies of the first point
        return np.full(number, x[0]), np.full(number, y[0])

    distance = distance/distance[-1]

    fx, fy = interp1d( distance, x ), interp1d( distance, y )

    alpha = np.linspace(0, 1, number)
    x_regular, y_regular = fx(alpha), fy(alpha)

    return x_regular, y_regular

def euclid_dist(x1, y1, x2, y2):
    """
    define a function to return the euclidean distance between two points
    inputs: x and y values of the two points
    output: the eulidean distance
    """
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def poly_area(x,y):
    """
    define a function to calculate the area of a polygon using the shoelace algorithm
    inputs: separate numpy arrays of x and y coordinate values
    outputs: the area of the polygon
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def gpa_mean(leaf_arr, landmark_num, dim_num):

    """
    define a function that given an array of landmark data returns the Generalized Procrustes Analysis mean
    inputs: a 3 dimensional array of samples by landmarks by coordinate values, number of landmarks, number of dimensions
    output: an array of the Generalized Procrustes Analysis mean shape

    """

    ref_ind = 0 # select arbitrary reference index to calculate procrustes distances to
    ref_shape = leaf_arr[ref_ind, :, :] # select the reference shape

    mean_diff = 10**(-30) # set a distance between means to stop the algorithm

    old_mean = ref_shape # for the first comparison between means, set old_mean to an arbitrary reference shape

    d = 1000000 # set d initially arbitraily high

    while d > mean_diff: # set boolean criterion for Procrustes distance between mean to stop calculations

        arr = np.zeros( ((len(leaf_arr)),landmark_num,dim_num) ) # empty 3D array: # samples, landmarks, coord vals

        for i in range(len(leaf_arr)): # for each leaf shape

            s1, s2, distance = procrustes(old_mean, leaf_arr[i]) # calculate procrustes adjusted shape to ref for current leaf
            arr[i] = s2 # store procrustes adjusted shape to array

        new_mean = np.mean(arr, axis=(0)) # calculate mean of all shapes adjusted to reference

        s1, s2, d = procrustes(old_mean, new_mean) # calculate procrustes distance of new mean to old mean

        old_mean = new_mean # set the old_mean to the new_mea before beginning another iteration

    return new_mean

# --- Configuration and Inputs ---

# Set a random seed for reproducibility
# --- ADDED: Random seed for reproducibility ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)

# Input File Paths
METADATA_FILE = "../../data/CULTIVATED1ST_landmarks.csv" # Adjusted path based on new directory structure
IMAGE_DATA_DIR = "../../data/CULTIVATED1ST_data/" # Adjusted path based on new directory structure

# Output Directory (will be created if it doesn't exist)
OUTPUT_BASE_DIR = "./outputs/" # Changed to "outputs" as per instructions for current analysis directory
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True) # Ensure output directory exists

# --- Parameters for Preprocessing ---
HIGH_RES_INTERPOLATION_POINTS = 10000 # Initial high resolution outline points
FINAL_PSEUDO_LANDMARKS_PER_SIDE = 50  # Number of equidistant points on each side (excluding the tip duplicate)
                                       # Total pseudo-landmarks will be (FINAL_PSEUDO_LANDMARKS_PER_SIDE * 2) - 1

# --- Parameters for Procrustes Analysis ---
NUM_LANDMARKS = (FINAL_PSEUDO_LANDMARKS_PER_SIDE * 2) - 1 # Derived from above
NUM_DIMENSIONS = 2                                         # For 2D coordinates

# --- Parameters for PCA (Full Component Analysis) ---
# This PCA is for explained variance analysis and later data augmentation.
# It will calculate all possible components up to min(samples, features).

# --- Parameters for Morphospace Visualization (2-Component PCA) ---
MORPHOSPACE_PLOT_LENGTH = 10 # Plot length in inches
MORPHOSPACE_PLOT_WIDTH = 10  # Plot width in inches
MORPHOSPACE_PC1_INTERVALS = 20 # Number of PC1 intervals for eigenleaf grid
MORPHOSPACE_PC2_INTERVALS = 6  # Number of PC2 intervals for eigenleaf grid
MORPHOSPACE_HUE_COLUMN = "type" # Column in mdata to color points by for the morphospace plot
EIGENLEAF_SCALE = 0.08 # Scaling of the inverse eigenleaves
EIGENLEAF_COLOR = "lightgray" # Color of inverse eigenleaf
EIGENLEAF_ALPHA = 0.5 # Alpha of inverse eigenleaf
POINT_SIZE = 80 # Size of data points
POINT_LINEWIDTH = 0 # Line width of data points (set to 0 for no edges)
POINT_ALPHA = 0.6 # Alpha of the data points
AXIS_LABEL_FONTSIZE = 12 # Font size of the x and y axis titles
AXIS_TICK_FONTSIZE = 8 # Font size of the axis ticks
FACE_COLOR = "white" # Color of the plot background
GRID_ALPHA = 0.5 # Alpha of the grid

# --- Parameters for Output Files ---
GPA_MEAN_SHAPE_PLOT_FILENAME = "gpa_mean_shape.png"
PCA_EXPLAINED_VARIANCE_REPORT_FILENAME = "pca_explained_variance.txt"
MORPHOSPACE_PLOT_FILENAME = "morphospace_plot.png"

# Specific filenames for saving PCA components, scores, and labels (using h5py)
PCA_PARAMS_H5_FILENAME = "leaf_pca_model_parameters.h5"
ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME = "original_pca_scores_and_class_labels.h5"
CLASS_LABEL_COLUMN_FOR_SAVING = "type" # The column from mdata to use for class labels (e.g., 'type', 'cultivar', etc.)

# E.g., FIGURE_DPI = 300 # Default DPI for saved figures

# --- End Configuration ---

print(f"Saving outputs to directory: {OUTPUT_BASE_DIR}")

########################
### READ IN METADATA ###
########################

mdata = pd.read_csv(METADATA_FILE) # read in csv

print(f"Metadata loaded from: {METADATA_FILE}")
print("First 5 rows of loaded metadata:")
print(mdata.head())


#######################################
### MAKE A LIST OF IMAGE FILE NAMES ###
#######################################

file_names = mdata['file'].tolist() # Get filenames directly from the metadata DataFrame
file_names.sort() # Ensure consistent order

print(f"Found {len(file_names)} image files to process from metadata.")


#####################################################################
### INTERPOLATE POINTS CREATING PSEUDO-LANDMARKS AND PRE-PROCESS ###
#####################################################################

print("\n--- Preprocessing Images and Interpolating Pseudo-Landmarks ---")
# an array to store pseudo-landmarks
# It's better to dynamically determine the actual number of successful samples
# or pre-filter mdata, but for now, we'll initialize and fill.
# We'll use a list to collect valid `rot_pts` and then convert to a numpy array.
processed_leaf_data = []
processed_mdata_rows = []

# for each leaf . . .
for lf_idx, row in mdata.iterrows(): # Use lf_idx for iteration, row for data access

    curr_image_filename = row["file"] # Select the current image filename from the row
    # print(f"Processing leaf {lf_idx+1}/{len(mdata)}: {curr_image_filename}") # Optional: progress indicator

    img_path = os.path.join(IMAGE_DATA_DIR, curr_image_filename)
    if not os.path.exists(img_path):
        print(f"Warning: Image file not found at {img_path}. Skipping '{curr_image_filename}'.")
        continue # Skip to the next iteration

    # Ensure image can be read (sometimes files can be corrupted or not actual images)
    try:
        img = cv2.bitwise_not(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY))
        if img is None:
            raise ValueError(f"Image at {img_path} could not be loaded.")
    except Exception as e:
        print(f"Error loading or processing image {curr_image_filename}: {e}. Skipping.")
        continue

    contours, hierarchy = cv2.findContours(img,
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_conts = []
    y_conts = []
    areas_conts = []
    for c in contours:
        x_vals = []
        y_vals = []
        for i in range(len(c)):
            x_vals.append(c[i][0][0])
            y_vals.append(c[i][0][1])
        # A simple "area" approximation - may not be precise for complex shapes but works for filtering
        if x_vals and y_vals: # Ensure there are points to calculate min/max
            area = (max(x_vals) - min(x_vals))*(max(y_vals) - min(y_vals))
            x_conts.append(x_vals)
            y_conts.append(y_vals)
            areas_conts.append(area)
    
    # Handle cases where no contours are found or contours are too small
    if not areas_conts:
        print(f"Warning: No valid contours found for image {curr_image_filename}. Skipping.")
        continue
    
    # Select the largest contour based on area
    area_inds = np.flip(np.argsort(areas_conts))
    
    # Ensure there's at least one valid contour before accessing index 0
    if len(area_inds) == 0:
        print(f"Warning: No significant contours found for image {curr_image_filename} after area sorting. Skipping.")
        continue

    largest_contour_x = np.array(x_conts[area_inds[0]], dtype=np.float32)
    largest_contour_y = np.array(y_conts[area_inds[0]], dtype=np.float32)

    # Check for empty contours after selection
    if largest_contour_x.size == 0 or largest_contour_y.size == 0:
        print(f"Warning: Largest contour for image {curr_image_filename} is empty. Skipping.")
        continue

    high_res_x, high_res_y = interpolation(largest_contour_x,
                                           largest_contour_y, HIGH_RES_INTERPOLATION_POINTS)

    # Ensure interpolated points are valid before proceeding
    if high_res_x.size == 0 or high_res_y.size == 0:
        print(f"Warning: Interpolation failed for image {curr_image_filename}. Skipping.")
        continue


    base_pt = np.array((row["base_x"], row["base_y"]))
    tip_pt = np.array((row["tip_x"], row["tip_y"]))

    base_dists = []
    tip_dists = []

    for pt in range(len(high_res_x)):
        ed_base = euclid_dist(base_pt[0], base_pt[1], high_res_x[pt], high_res_y[pt])
        ed_tip = euclid_dist(tip_pt[0], tip_pt[1], high_res_x[pt], high_res_y[pt])
        base_dists.append(ed_base)
        tip_dists.append(ed_tip)

    base_ind = np.argmin(base_dists)
    tip_ind = np.argmin(tip_dists)

    # Reorder the contour to start at the base point
    high_res_x = np.concatenate((high_res_x[base_ind:], high_res_x[:base_ind]))
    high_res_y = np.concatenate((high_res_y[base_ind:], high_res_y[:base_ind]))
    
    # Find the index of the tip point in the newly rotated array
    new_tip_dists = []
    for pt_idx in range(len(high_res_x)):
        new_tip_dists.append(euclid_dist(tip_pt[0], tip_pt[1], high_res_x[pt_idx], high_res_y[pt_idx]))
    tip_ind_new = np.argmin(new_tip_dists)

    lf_contour = np.column_stack((high_res_x, high_res_y))

    # Define the left and right segments correctly, ensuring they loop properly and share the base/tip points
    # Left segment: from base (index 0) to tip (tip_ind_new)
    left_segment = lf_contour[0:tip_ind_new+1, :] # Includes base and tip

    # Right segment: from tip (tip_ind_new) back to base (index 0), ensuring the base point is included at the end
    # This means the right_segment starts at tip_ind_new, goes to the end, and then wraps around to the beginning (index 0).
    right_segment = np.concatenate((lf_contour[tip_ind_new:, :], lf_contour[0:1, :]), axis=0)
    
    # Ensure segments have at least 2 points for interpolation to work reliably
    if len(left_segment) < 2 or len(right_segment) < 2:
        print(f"Warning: Segments for image {curr_image_filename} are too short for interpolation. Skipping.")
        continue # Skip this image

    left_inter_x, left_inter_y = interpolation(left_segment[:,0], left_segment[:,1], FINAL_PSEUDO_LANDMARKS_PER_SIDE)
    right_inter_x, right_inter_y = interpolation(right_segment[:,0], right_segment[:,1], FINAL_PSEUDO_LANDMARKS_PER_SIDE)


    left_inter_x = np.delete(left_inter_x, -1) # Remove last point of left side (which is the tip, to avoid duplication)
    left_inter_y = np.delete(left_inter_y, -1) # Remove last point of left side (which is the tip, to avoid duplication)

    lf_pts_left = np.column_stack((left_inter_x, left_inter_y))
    lf_pts_right = np.column_stack((right_inter_x, right_inter_y))
    lf_pts = np.row_stack((lf_pts_left, lf_pts_right))

    # Ensure the total number of landmarks is correct
    if lf_pts.shape[0] != NUM_LANDMARKS:
        print(f"Warning: Leaf {curr_image_filename} generated {lf_pts.shape[0]} landmarks, expected {NUM_LANDMARKS}. Check interpolation logic.")
        # This warning is important if the `FINAL_PSEUDO_LANDMARKS_PER_SIDE` and total `NUM_LANDMARKS` calculation results in an off-by-one error for some cases.
        # The current calculation is (N_side * 2) - 1 because the tip is counted once, and the base is effectively the start/end point.
        # If FINAL_PSEUDO_LANDMARKS_PER_SIDE includes the endpoint (as interpolation does), then
        # left_inter has N points (base to tip)
        # right_inter has N points (tip to base)
        # if we remove tip from left, left has N-1 points.
        # if we assume right starts at tip and ends at base, it has N points.
        # Total = (N-1) + N = 2N-1. This seems correct.

    tip_point = lf_pts[FINAL_PSEUDO_LANDMARKS_PER_SIDE-1,:] # This should be the tip (last point of left_inter_x before deletion)
    base_point = lf_pts[0,:] # This should be the base

    # Handle cases where base_point and tip_point might be identical or very close,
    # leading to issues in angle calculation (e.g., if a tiny contour was picked).
    if np.array_equal(base_point, tip_point):
        print(f"Warning: Base and tip points are identical for {curr_image_filename}. Cannot calculate angle. Skipping.")
        continue
    
    # Ensure there is enough numerical precision/variation for angle calculation
    # If base_point and (base_point[0]+1, base_point[1]) are too close, atan2 can behave oddly.
    # This is more robustly handled by ensuring enough distance between points in interpolation,
    # but a check here is an extra safeguard.
    if euclid_dist(base_point[0], base_point[1], base_point[0]+1, base_point[1]) < 1e-6: # Arbitrary small threshold
         print(f"Warning: Reference point for angle calculation too close to base point for {curr_image_filename}. Skipping.")
         continue

    ang = angle_between(tip_point, base_point, (base_point[0]+1,base_point[1]) )

    rot_x, rot_y = rotate_points(lf_pts[:,0], lf_pts[:,1], ang)
    rot_pts = np.column_stack((rot_x, rot_y))

    # Store the processed points and metadata row for successful entries
    processed_leaf_data.append(rot_pts)
    processed_mdata_rows.append(row)

# Convert list of processed data to numpy array
if processed_leaf_data:
    cult_cm_arr = np.stack(processed_leaf_data)
    # Update mdata to only include rows that were successfully processed
    mdata = pd.DataFrame(processed_mdata_rows).reset_index(drop=True)
    print(f"Successfully processed {len(processed_leaf_data)} out of {len(file_names)} images.")
else:
    print("No images were successfully processed. Exiting script.")
    exit() # Exit if no data to process further

##########################
### CALCULATE GPA MEAN ###
##########################

print("\n--- Calculating GPA Mean ---")
mean_shape = gpa_mean(cult_cm_arr, NUM_LANDMARKS, NUM_DIMENSIONS)

################################
### ALIGN LEAVES TO GPA MEAN ###
################################

print("--- Aligning Leaves to GPA Mean ---")
proc_arr = np.zeros(np.shape(cult_cm_arr))

for i in range(len(cult_cm_arr)):
    s1, s2, distance = procrustes(mean_shape, cult_cm_arr[i, :, :])
    proc_arr[i] = s2

#### VISUALIZE GPA ALIGNED SHAPES AND MEAN
print("--- Visualizing GPA Aligned Shapes ---")
plt.figure(figsize=(8, 8))
for i in range(len(proc_arr)):
    plt.plot(proc_arr[i, :, 0], proc_arr[i, :, 1], c="k", alpha=0.08)

plt.plot(np.mean(proc_arr, axis=0)[:, 0], np.mean(proc_arr, axis=0)[:, 1], c="magenta")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title("Procrustes Aligned Leaf Shapes and GPA Mean")

plt.savefig(os.path.join(OUTPUT_BASE_DIR, GPA_MEAN_SHAPE_PLOT_FILENAME))
plt.close()
print(f"GPA mean shape plot saved to {os.path.join(OUTPUT_BASE_DIR, GPA_MEAN_SHAPE_PLOT_FILENAME)}")


#################################################
### FIRST, CALCULATE PERCENT VARIANCE ALL PCs ###
#################################################

print("\n--- Performing Full PCA and Generating Explained Variance Report ---")

# use the reshape function to flatten to 2D
flat_arr = proc_arr.reshape(np.shape(proc_arr)[0],
                            np.shape(proc_arr)[1] * np.shape(proc_arr)[2])

# Determine the maximum number of principal components possible: min(n_samples, n_features)
max_pc_components = min(flat_arr.shape[0], flat_arr.shape[1])

# Initialize PCA to calculate all possible PCs for full variance analysis
pca = PCA(n_components=max_pc_components, random_state=RANDOM_SEED) # --- ADDED: Random state for PCA ---
PCs = pca.fit_transform(flat_arr) # fit a PCA for all data

# Generate and save explained variance report
pca_explained_variance_filepath = os.path.join(OUTPUT_BASE_DIR, PCA_EXPLAINED_VARIANCE_REPORT_FILENAME)
with open(pca_explained_variance_filepath, 'w') as f:
    f.write("PCA Explained Variance Report:\n")
    f.write(f"Total Samples: {flat_arr.shape[0]}\n")
    f.write(f"Total Features (landmarks * dimensions): {flat_arr.shape[1]}\n")
    f.write(f"Number of PCs Calculated: {pca.n_components_}\n\n")

    f.write("PC: var, overall\n")
    for i in range(len(pca.explained_variance_ratio_)):
        pc_variance = round(pca.explained_variance_ratio_[i] * 100, 2)
        cumulative_variance = round(pca.explained_variance_ratio_.cumsum()[i] * 100, 2)
        line = f"PC{i+1}: {pc_variance}%, {cumulative_variance}%\n"
        print(line.strip()) # Also print to console
        f.write(line)
print(f"PCA explained variance report saved to {pca_explained_variance_filepath}")

# --- Save PCA Model Parameters, PC Scores, and Class Labels ---
print("\n--- Saving PCA model parameters, PC scores, and class labels ---")

# 1. Extract information from the PCA model and original data
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
print(f"  Class Labels ({CLASS_LABEL_COLUMN_FOR_SAVING}) length: {len(mdata[CLASS_LABEL_COLUMN_FOR_SAVING])}")

# 2. Save the PCA model parameters to an HDF5 file
pca_params_filepath = os.path.join(OUTPUT_BASE_DIR, PCA_PARAMS_H5_FILENAME)
with h5py.File(pca_params_filepath, 'w') as f:
    f.create_dataset('components', data=pca_components, compression="gzip")
    f.create_dataset('mean', data=pca_mean, compression="gzip")
    f.create_dataset('explained_variance', data=pca_explained_variance, compression="gzip")
    f.create_dataset('explained_variance_ratio', data=pca_explained_variance_ratio, compression="gzip")
    f.attrs['n_components'] = n_pca_components
print(f"PCA parameters saved to {pca_params_filepath}")

# 3. Save original PCA scores (PCs) and class labels to an HDF5 file
pca_scores_labels_filepath = os.path.join(OUTPUT_BASE_DIR, ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME)
with h5py.File(pca_scores_labels_filepath, 'w') as f:
    f.create_dataset('pca_scores', data=PCs, compression="gzip")
    # Convert labels to a numpy array of byte strings for HDF5 compatibility
    f.create_dataset('class_labels', data=np.array(mdata[CLASS_LABEL_COLUMN_FOR_SAVING]).astype('S'), compression="gzip")
    # --- ADDED: Save the original flattened coordinates ---
    f.create_dataset('original_flattened_coords', data=flat_arr, compression="gzip")
print(f"Original PCA scores, class labels, AND original flattened coordinates saved to {pca_scores_labels_filepath}")


##########################
### CREATE MORPHOSPACE ###
##########################

print("\n--- Creating Morphospace Plot ---")

# The flat_arr is already prepared from the previous full PCA step.

# Perform PCA specifically for morphospace visualization (2 components)
morphospace_pca = PCA(n_components=2, random_state=RANDOM_SEED) # --- ADDED: Random state for PCA ---
morphospace_PCs = morphospace_pca.fit_transform(flat_arr)

# Add the 2-component PCA results to the mdata DataFrame
mdata["PC1"] = morphospace_PCs[:, 0]
mdata["PC2"] = morphospace_PCs[:, 1]

# Set up the plot
plt.figure(figsize=(MORPHOSPACE_PLOT_LENGTH, MORPHOSPACE_PLOT_WIDTH))
plt.gca().set_facecolor(FACE_COLOR)
plt.gca().set_axisbelow(True)

# Create PC intervals for plotting inverse eigenleaves
# Use the PCs calculated from the FULL PCA (not the 2-component morphospace_PCs) for the overall range
PC1_vals = np.linspace(np.min(PCs[:, 0]), np.max(PCs[:, 0]), MORPHOSPACE_PC1_INTERVALS)
PC2_vals = np.linspace(np.min(PCs[:, 1]), np.max(PCs[:, 1]), MORPHOSPACE_PC2_INTERVALS)

# Plot inverse eigenleaves (the background grid shapes)
for i in PC1_vals:
    for j in PC2_vals:
        # Note: morphospace_pca.inverse_transform expects 2 components here as it was fit with 2 components.
        inv_leaf = morphospace_pca.inverse_transform(np.array([i, j]))
        inv_leaf_coords = inv_leaf.reshape(NUM_LANDMARKS, NUM_DIMENSIONS) # Reshape back to 2D points

        inv_x = inv_leaf_coords[:, 0]
        inv_y = inv_leaf_coords[:, 1]

        plt.fill(inv_x * EIGENLEAF_SCALE + i, inv_y * EIGENLEAF_SCALE + j,
                 c=EIGENLEAF_COLOR, alpha=EIGENLEAF_ALPHA)

# Plot the data points on top of the morphospace
sns.scatterplot(data=mdata, x="PC1", y="PC2", hue=MORPHOSPACE_HUE_COLUMN,
                s=POINT_SIZE, linewidth=POINT_LINEWIDTH, alpha=POINT_ALPHA)

# Add legend
plt.legend(bbox_to_anchor=(1.00, 1.02), prop={'size': 8.9})

# Customize axis labels using explained variance from the FULL PCA
# Ensure `pca.explained_variance_ratio_` is accessible from the full PCA.
xlab = f"PC1, {round(pca.explained_variance_ratio_[0] * 100, 1)}%"
ylab = f"PC2, {round(pca.explained_variance_ratio_[1] * 100, 1)}%"
plt.xlabel(xlab, fontsize=AXIS_LABEL_FONTSIZE)
plt.ylabel(ylab, fontsize=AXIS_LABEL_FONTSIZE)
plt.xticks(fontsize=AXIS_TICK_FONTSIZE)
plt.yticks(fontsize=AXIS_TICK_FONTSIZE)
plt.gca().set_aspect("equal")

# Save the figure
plt.savefig(os.path.join(OUTPUT_BASE_DIR, MORPHOSPACE_PLOT_FILENAME), bbox_inches='tight')
plt.close()
print(f"Morphospace plot saved to {os.path.join(OUTPUT_BASE_DIR, MORPHOSPACE_PLOT_FILENAME)}")

print("\n--- All processing and saving completed ---")
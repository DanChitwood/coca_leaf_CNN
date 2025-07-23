#######################
### LOAD IN MODULES ###
#######################

import cv2
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns
import h5py
import os
from os import listdir
from os.path import isfile, join


#################
### FUNCTIONS ###
#################

def angle_between(p1, p2, p3):
    """
    Calculates the angle between three points (p1, p2, p3) with p2 as the vertex,
    in degrees, measured counter-clockwise.

    Args:
        p1 (tuple): Coordinates of point 1 (x1, y1).
        p2 (tuple): Coordinates of the vertex point (x2, y2).
        p3 (tuple): Coordinates of point 3 (x3, y3).

    Returns:
        float: Angle in degrees (0 to 360).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + math.degrees(math.atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + math.degrees(math.atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def rotate_points(xvals, yvals, degrees):
    """
    Rotates 2D x and y coordinate points around the origin.

    Args:
        xvals (numpy.ndarray): Array of x-coordinates.
        yvals (numpy.ndarray): Array of y-coordinates.
        degrees (float): Degrees to rotate (positive for counter-clockwise).

    Returns:
        tuple: (new_xvals, new_yvals) as numpy arrays.
    """
    angle_to_move = 90 - degrees
    rads = np.deg2rad(angle_to_move)

    new_xvals = xvals * np.cos(rads) - yvals * np.sin(rads)
    new_yvals = xvals * np.sin(rads) + yvals * np.cos(rads)

    return new_xvals, new_yvals

def interpolation(x, y, number):
    """
    Returns equally spaced, interpolated points for a given polyline.

    Args:
        x (numpy.ndarray): Array of x values for the polyline.
        y (numpy.ndarray): Array of y values for the polyline.
        number (int): Number of points to interpolate.

    Returns:
        tuple: (x_regular, y_regular) as numpy arrays of interpolated points.
    """
    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
    distance = distance / distance[-1]

    fx, fy = interp1d(distance, x), interp1d(distance, y)

    alpha = np.linspace(0, 1, number)
    x_regular, y_regular = fx(alpha), fy(alpha)

    return x_regular, y_regular

def euclid_dist(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two 2D points.

    Args:
        x1 (float): x-coordinate of point 1.
        y1 (float): y-coordinate of point 1.
        x2 (float): x-coordinate of point 2.
        y2 (float): y-coordinate of point 2.

    Returns:
        float: The Euclidean distance.
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def poly_area(x, y):
    """
    Calculates the area of a polygon using the shoelace algorithm.

    Args:
        x (numpy.ndarray): Array of x-coordinates of the polygon vertices.
        y (numpy.ndarray): Array of y-coordinates of the polygon vertices.

    Returns:
        float: The area of the polygon.
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def gpa_mean(leaf_arr, landmark_num, dim_num):
    """
    Calculates the Generalized Procrustes Analysis (GPA) mean shape from a
    3D array of landmark data.

    Args:
        leaf_arr (numpy.ndarray): A 3D array of samples x landmarks x coordinates.
        landmark_num (int): Number of landmarks per shape.
        dim_num (int): Number of dimensions (2 for 2D coordinates).

    Returns:
        numpy.ndarray: An array of the GPA mean shape (landmarks x coordinates).
    """
    ref_ind = 0
    ref_shape = leaf_arr[ref_ind, :, :]

    mean_diff = 1e-10

    old_mean = ref_shape

    d = np.inf

    while d > mean_diff:
        arr = np.zeros(((len(leaf_arr)), landmark_num, dim_num))

        for i in range(len(leaf_arr)):
            s1, s2, distance = procrustes(old_mean, leaf_arr[i])
            arr[i] = s2

        new_mean = np.mean(arr, axis=(0))

        s1, s2, d = procrustes(old_mean, new_mean)

        old_mean = new_mean

    return new_mean

# --- GLOBAL CONFIGURATION ---

# Define the root project directory relative to the script's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Input File Paths
METADATA_FILE = os.path.join(PROJECT_ROOT, "data", "PLOWMAN_landmarks.csv")
IMAGE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "PLOWMAN_data")

# Output Directory
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "analysis", "PLOWMAN", "outputs")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# --- Parameters for Preprocessing ---
HIGH_RES_INTERPOLATION_POINTS = 10000
FINAL_PSEUDO_LANDMARKS_PER_SIDE = 50
NUM_LANDMARKS = (FINAL_PSEUDO_LANDMARKS_PER_SIDE * 2) - 1
NUM_DIMENSIONS = 2

# --- Parameters for Morphospace Visualization (2-Component PCA) ---
MORPHOSPACE_PLOT_LENGTH = 10
MORPHOSPACE_PLOT_WIDTH = 10
MORPHOSPACE_PC1_INTERVALS = 20
MORPHOSPACE_PC2_INTERVALS = 6
MORPHOSPACE_HUE_COLUMN = "type"
EIGENLEAF_SCALE = 0.08
EIGENLEAF_COLOR = "lightgray"
EIGENLEAF_ALPHA = 0.5
POINT_SIZE = 80
POINT_LINEWIDTH = 0
POINT_ALPHA = 0.6
AXIS_LABEL_FONTSIZE = 12
AXIS_TICK_FONTSIZE = 8
FACE_COLOR = "white"
GRID_ALPHA = 0.5

# --- Parameters for Output Files ---
GPA_MEAN_SHAPE_PLOT_FILENAME = "gpa_mean_shape.png"
PCA_EXPLAINED_VARIANCE_REPORT_FILENAME = "pca_explained_variance.txt"
MORPHOSPACE_PLOT_FILENAME = "morphospace_plot.png"
PCA_PARAMS_H5_FILENAME = "plowman_pca_model_parameters.h5"
ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME = "plowman_original_pca_scores_and_class_labels.h5"
CLASS_LABEL_COLUMN_FOR_SAVING = "type"
FIGURE_DPI = 300

print(f"--- Running PLOWMAN Morphometric Analysis ---")
print(f"Project root: {PROJECT_ROOT}")
print(f"Input metadata file: {METADATA_FILE}")
print(f"Input image directory: {IMAGE_DATA_DIR}")
print(f"Saving outputs to directory: {OUTPUT_BASE_DIR}")

########################
### READ IN METADATA ###
########################

print("\n--- Reading Metadata ---")
mdata = pd.read_csv(METADATA_FILE)

mdata_sorted = mdata.sort_values(by=['Label', 'index'])

new_df = mdata_sorted.groupby('Label').agg(
    base_x=('X', lambda x: x.iloc[0]),
    base_y=('Y', lambda x: x.iloc[0]),
    tip_x=('X', lambda x: x.iloc[1]),
    tip_y=('Y', lambda x: x.iloc[1])
).reset_index()

mdata = new_df.rename(columns={'Label': 'file'})

mdata['type'] = mdata['file'].apply(lambda x: x.split('_')[0])

print(f"Metadata loaded. Found {len(mdata)} leaf entries with {len(mdata['type'].unique())} unique types.")


#######################################
### MAKE A LIST OF IMAGE FILE NAMES ###
#######################################

print("\n--- Listing Image Files ---")
image_files_in_dir = {f for f in listdir(IMAGE_DATA_DIR) if isfile(join(IMAGE_DATA_DIR, f)) and not f.startswith('.')}
file_names_to_process = [f for f in mdata['file'].values if f in image_files_in_dir]

if not file_names_to_process:
    raise FileNotFoundError(f"No image files found in '{IMAGE_DATA_DIR}' that match entries in '{METADATA_FILE}'. "
                            "Please check paths and file names.")

mdata = mdata[mdata['file'].isin(file_names_to_process)].set_index('file').loc[file_names_to_process].reset_index()

print(f"Found {len(file_names_to_process)} image files to process.")


#####################################################################
### INTERPOLATE POINTS CREATING PSEUDO-LANDMARKS AND PRE-PROCESS ###
#####################################################################

print("\n--- Preprocessing Images and Interpolating Pseudo-Landmarks ---")
cult_cm_arr = np.zeros((len(mdata), NUM_LANDMARKS, NUM_DIMENSIONS))

for lf_idx, curr_image_name in enumerate(mdata["file"]):
    img_path = os.path.join(IMAGE_DATA_DIR, curr_image_name)
    img_raw = cv2.imread(img_path)

    if img_raw is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        continue

    img = cv2.bitwise_not(cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY))

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"Warning: No contours found for image {curr_image_name}. Skipping.")
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    x_contour = largest_contour[:, 0, 0]
    y_contour = largest_contour[:, 0, 1]

    high_res_x, high_res_y = interpolation(np.array(x_contour, dtype=np.float32),
                                            np.array(y_contour, dtype=np.float32),
                                            HIGH_RES_INTERPOLATION_POINTS)

    base_pt = np.array((mdata["base_x"][lf_idx], mdata["base_y"][lf_idx]))
    tip_pt = np.array((mdata["tip_x"][lf_idx], mdata["tip_y"][lf_idx]))

    base_dists = np.sqrt((high_res_x - base_pt[0])**2 + (high_res_y - base_pt[1])**2)
    tip_dists = np.sqrt((high_res_x - tip_pt[0])**2 + (high_res_y - tip_pt[1])**2)

    base_ind = np.argmin(base_dists)
    tip_ind = np.argmin(tip_dists)

    high_res_x = np.concatenate((high_res_x[base_ind:], high_res_x[:base_ind]))
    high_res_y = np.concatenate((high_res_y[base_ind:], high_res_y[:base_ind]))

    tip_ind = (tip_ind - base_ind + HIGH_RES_INTERPOLATION_POINTS) % HIGH_RES_INTERPOLATION_POINTS

    lf_contour = np.column_stack((high_res_x, high_res_y))

    if tip_ind > 0:
        left_segment_x = lf_contour[0:tip_ind+1, 0]
        left_segment_y = lf_contour[0:tip_ind+1, 1]
        right_segment_x = np.concatenate((lf_contour[tip_ind:, 0], lf_contour[0:1, 0]))
        right_segment_y = np.concatenate((lf_contour[tip_ind:, 1], lf_contour[0:1, 1]))
    else:
        left_segment_x = lf_contour[0:1, 0]
        left_segment_y = lf_contour[0:1, 1]
        right_segment_x = lf_contour[:, 0]
        right_segment_y = lf_contour[:, 1]

    left_inter_x, left_inter_y = interpolation(left_segment_x, left_segment_y, FINAL_PSEUDO_LANDMARKS_PER_SIDE)
    right_inter_x, right_inter_y = interpolation(right_segment_x, right_segment_y, FINAL_PSEUDO_LANDMARKS_PER_SIDE)

    left_inter_x = np.delete(left_inter_x, -1)
    left_inter_y = np.delete(left_inter_y, -1)

    lf_pts_left_processed = np.column_stack((left_inter_x, left_inter_y))
    lf_pts_right_processed = np.column_stack((right_inter_x, right_inter_y))
    lf_pts = np.row_stack((lf_pts_left_processed, lf_pts_right_processed))

    if lf_pts.shape[0] != NUM_LANDMARKS:
        print(f"Error: {curr_image_name} has {lf_pts.shape[0]} landmarks, expected {NUM_LANDMARKS}.")
        # Handle this error more robustly if needed, e.g., skip the sample or raise an exception.

    tip_point = lf_pts[FINAL_PSEUDO_LANDMARKS_PER_SIDE-1,:]
    base_point = lf_pts[0,:]

    ang = angle_between(tip_point, base_point, (base_point[0] + 1, base_point[1]))

    rot_x, rot_y = rotate_points(lf_pts[:, 0], lf_pts[:, 1], ang)
    rot_pts = np.column_stack((rot_x, rot_y))

    cult_cm_arr[lf_idx, :, :] = rot_pts

print(f"Preprocessing complete. Pseudo-landmarks array shape: {cult_cm_arr.shape}")

##########################
### CALCULATE GPA MEAN ###
##########################

print("\n--- Calculating GPA Mean Shape ---")
mean_shape = gpa_mean(cult_cm_arr, NUM_LANDMARKS, NUM_DIMENSIONS)
print(f"GPA mean shape calculated. Shape: {mean_shape.shape}")


################################
### ALIGN LEAVES TO GPA MEAN ###
################################

print("--- Aligning Leaves to GPA Mean ---")
proc_arr = np.zeros(np.shape(cult_cm_arr))

for i in range(len(cult_cm_arr)):
    s1, s2, distance = procrustes(mean_shape, cult_cm_arr[i, :, :])
    proc_arr[i] = s2
print(f"Leaves aligned. Procrustes array shape: {proc_arr.shape}")

#### VISUALIZE GPA ALIGNED SHAPES AND MEAN
print("--- Visualizing GPA Aligned Shapes and Mean ---")
plt.figure(figsize=(8, 8))
for i in range(len(proc_arr)):
    plt.plot(proc_arr[i, :, 0], proc_arr[i, :, 1], c="k", alpha=0.08)

plt.plot(np.mean(proc_arr, axis=0)[:, 0], np.mean(proc_arr, axis=0)[:, 1], c="magenta")
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title("Procrustes Aligned Leaf Shapes and GPA Mean")

plt.savefig(os.path.join(OUTPUT_BASE_DIR, GPA_MEAN_SHAPE_PLOT_FILENAME), dpi=FIGURE_DPI)
plt.close()
print(f"GPA mean shape plot saved to {os.path.join(OUTPUT_BASE_DIR, GPA_MEAN_SHAPE_PLOT_FILENAME)}")


#################################################
### FIRST, CALCULATE PERCENT VARIANCE ALL PCs ###
#################################################

print("\n--- Performing Full PCA and Generating Explained Variance Report ---")

flat_arr = proc_arr.reshape(np.shape(proc_arr)[0],
                            np.shape(proc_arr)[1] * np.shape(proc_arr)[2])

max_pc_components = min(flat_arr.shape[0], flat_arr.shape[1])

pca = PCA(n_components=max_pc_components, random_state=42)
PCs = pca.fit_transform(flat_arr)

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
print(f"  Class Labels ({CLASS_LABEL_COLUMN_FOR_SAVING}) length: {len(mdata[CLASS_LABEL_COLUMN_FOR_SAVING])}")

pca_params_filepath = os.path.join(OUTPUT_BASE_DIR, PCA_PARAMS_H5_FILENAME)
with h5py.File(pca_params_filepath, 'w') as f:
    f.create_dataset('components', data=pca_components, compression="gzip")
    f.create_dataset('mean', data=pca_mean, compression="gzip")
    f.create_dataset('explained_variance', data=pca_explained_variance, compression="gzip")
    f.create_dataset('explained_variance_ratio', data=pca_explained_variance_ratio, compression="gzip")
    f.attrs['n_components'] = n_pca_components
print(f"PCA parameters saved to {pca_params_filepath}")

pca_scores_labels_filepath = os.path.join(OUTPUT_BASE_DIR, ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME)
with h5py.File(pca_scores_labels_filepath, 'w') as f:
    f.create_dataset('pca_scores', data=PCs, compression="gzip")
    f.create_dataset('class_labels', data=np.array(mdata[CLASS_LABEL_COLUMN_FOR_SAVING]).astype('S'), compression="gzip")
    f.create_dataset('original_flattened_coords', data=flat_arr, compression="gzip")
print(f"Original PCA scores, class labels, AND original flattened coordinates saved to {pca_scores_labels_filepath}")


##########################
### CREATE MORPHOSPACE ###
##########################

print("\n--- Creating Morphospace Plot ---")

morphospace_pca = PCA(n_components=2, random_state=42)
morphospace_PCs = morphospace_pca.fit_transform(flat_arr)

mdata["PC1"] = morphospace_PCs[:, 0]
mdata["PC2"] = morphospace_PCs[:, 1]

plt.figure(figsize=(MORPHOSPACE_PLOT_LENGTH, MORPHOSPACE_PLOT_WIDTH))
ax = plt.gca()
ax.set_facecolor(FACE_COLOR)
ax.set_axisbelow(True)

PC1_vals = np.linspace(np.min(morphospace_PCs[:, 0]), np.max(morphospace_PCs[:, 0]), MORPHOSPACE_PC1_INTERVALS)
PC2_vals = np.linspace(np.min(morphospace_PCs[:, 1]), np.max(morphospace_PCs[:, 1]), MORPHOSPACE_PC2_INTERVALS)

for i in PC1_vals:
    for j in PC2_vals:
        inv_leaf = morphospace_pca.inverse_transform(np.array([i, j]))
        inv_leaf_coords = inv_leaf.reshape(NUM_LANDMARKS, NUM_DIMENSIONS)

        inv_x = inv_leaf_coords[:, 0]
        inv_y = inv_leaf_coords[:, 1]

        ax.fill(inv_x * EIGENLEAF_SCALE + i, inv_y * EIGENLEAF_SCALE + j,
                c=EIGENLEAF_COLOR, alpha=EIGENLEAF_ALPHA)

sns.scatterplot(data=mdata, x="PC1", y="PC2", hue=MORPHOSPACE_HUE_COLUMN,
                s=POINT_SIZE, linewidth=POINT_LINEWIDTH, alpha=POINT_ALPHA, ax=ax)

ax.legend(bbox_to_anchor=(1.00, 1.02), prop={'size': 8.9})

xlab = f"PC1, {round(morphospace_pca.explained_variance_ratio_[0] * 100, 1)}%"
ylab = f"PC2, {round(morphospace_pca.explained_variance_ratio_[1] * 100, 1)}%"
ax.set_xlabel(xlab, fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel(ylab, fontsize=AXIS_LABEL_FONTSIZE)
ax.tick_params(axis='x', labelsize=AXIS_TICK_FONTSIZE)
ax.tick_params(axis='y', labelsize=AXIS_TICK_FONTSIZE)
ax.set_aspect("equal")
ax.grid(True, linestyle=':', alpha=GRID_ALPHA)

plt.savefig(os.path.join(OUTPUT_BASE_DIR, MORPHOSPACE_PLOT_FILENAME),
            bbox_inches='tight', dpi=FIGURE_DPI)
plt.close()
print(f"Morphospace plot saved to {os.path.join(OUTPUT_BASE_DIR, MORPHOSPACE_PLOT_FILENAME)}")

print("\n--- All PLOWMAN morphometric analysis and saving completed ---")
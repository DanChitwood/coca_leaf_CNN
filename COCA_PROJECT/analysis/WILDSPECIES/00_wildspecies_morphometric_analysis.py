#######################
### LOAD IN MODULES ###
#######################

import numpy as np # for using arrays
import pandas as pd # for using pandas dataframes
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for plotting in seaborn
from sklearn.decomposition import PCA # for principal component analysis
import h5py # For saving large arrays and PCA model parameters
import os # For path operations and directory creation
from pathlib import Path # For robust path management

#############################
### CONFIGURATION & PATHS ###
#############################

# --- Project Structure Configuration ---
# Define the base project directory by navigating up from the script's location.
# This ensures it points to the root 'COCA_PROJECT' folder regardless of the current working directory.
COCA_PROJECT_BASE = Path(__file__).resolve().parent.parent.parent

# Define the specific analysis subfolder for this run (WILDSPECIES)
WILDSPECIES_ANALYSIS_DIR = COCA_PROJECT_BASE / "analysis" / "WILDSPECIES"

# Define the data input directory
DATA_INPUT_DIR = COCA_PROJECT_BASE / "data"

# --- Random Seed for Reproducibility ---
# Set random seeds for NumPy operations to ensure reproducibility of results.
np.random.seed(42)
print("Random seeds set for reproducibility.")

# --- Input File Paths ---
# Paths to the pre-computed Procrustes array and metadata DataFrame.
PROC_ARRAY_FILE = DATA_INPUT_DIR / "WILDSPECIES_procrustes.npy"
METADATA_DF_FILE = DATA_INPUT_DIR / "WILDSPECIES_metadata.csv"

# --- Output Directory Setup ---
# All generated plots and data will be saved here.
OUTPUT_BASE_DIR = WILDSPECIES_ANALYSIS_DIR / "outputs"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True) # Create the directory if it doesn't exist

print(f"Base COCA_PROJECT directory set to: {COCA_PROJECT_BASE}")
print(f"WILDSPECIES analysis outputs will be saved to: {OUTPUT_BASE_DIR}")
print(f"Input data expected from: {DATA_INPUT_DIR}")

# --- Parameters for Morphometric Analysis ---
NUM_LANDMARKS = 0 # Will be derived from proc_arr.shape[1] after loading
NUM_DIMENSIONS = 0 # Will be derived from proc_arr.shape[2] after loading

# --- Parameters for Morphospace Visualization (2-Component PCA) ---
MORPHOSPACE_PLOT_LENGTH = 10 # Plot length in inches
MORPHOSPACE_PLOT_WIDTH = 10 # Plot width in inches
MORPHOSPACE_PC1_INTERVALS = 20 # Number of PC1 intervals for eigenleaf grid
MORPHOSPACE_PC2_INTERVALS = 6 # Number of PC2 intervals for eigenleaf grid
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

# --- Parameters for Output Filenames ---
GPA_MEAN_SHAPE_PLOT_FILENAME = "gpa_mean_shape.png"
PCA_EXPLAINED_VARIANCE_REPORT_FILENAME = "pca_explained_variance.txt"
MORPHOSPACE_PLOT_FILENAME = "morphospace_plot.png"

# Specific filenames for saving PCA components, scores, and labels (using h5py)
PCA_PARAMS_H5_FILENAME = "leaf_pca_model_parameters.h5"
ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME = "original_pca_scores_and_class_labels.h5"
CLASS_LABEL_COLUMN_FOR_SAVING = "type" # The column from mdata to use for class labels

# --- End Configuration ---

print(f"Saving outputs to directory: {OUTPUT_BASE_DIR}")

##############################
### LOAD PRE-COMPUTED DATA ###
##############################

print(f"\n--- Loading Pre-computed Procrustes Array from {PROC_ARRAY_FILE} ---")
try:
    proc_arr = np.load(PROC_ARRAY_FILE)
    print(f"Successfully loaded proc_arr. Shape: {proc_arr.shape}")
    NUM_SAMPLES = proc_arr.shape[0]
    NUM_LANDMARKS = proc_arr.shape[1]
    NUM_DIMENSIONS = proc_arr.shape[2]
    print(f"Derived parameters: Samples={NUM_SAMPLES}, Landmarks={NUM_LANDMARKS}, Dimensions={NUM_DIMENSIONS}")

except FileNotFoundError:
    print(f"Error: {PROC_ARRAY_FILE} not found. Please ensure the file exists at: {PROC_ARRAY_FILE}")
    exit()
except Exception as e:
    print(f"An error occurred while loading proc_arr: {e}")
    exit()

print(f"\n--- Loading Metadata DataFrame from {METADATA_DF_FILE} ---")
try:
    mdata = pd.read_csv(METADATA_DF_FILE)
    print(f"Successfully loaded mdata. Shape: {mdata.shape}")
    print("First 5 rows of loaded metadata:")
    print(mdata.head())
    print(f"Value counts for '{CLASS_LABEL_COLUMN_FOR_SAVING}' column:")
    print(mdata[CLASS_LABEL_COLUMN_FOR_SAVING].value_counts())

except FileNotFoundError:
    print(f"Error: {METADATA_DF_FILE} not found. Please ensure the file exists at: {METADATA_DF_FILE}")
    exit()
except Exception as e:
    print(f"An error occurred while loading mdata: {e}")
    exit()

# --- Verify consistency ---
if proc_arr.shape[0] != len(mdata):
    print(f"\nCRITICAL WARNING: Number of samples in proc_arr ({proc_arr.shape[0]}) does not match number of rows in metadata ({len(mdata)}).")
    print("This may lead to incorrect class assignments for PCA and plotting.")
    # Exit if inconsistency is critical for your analysis.
    # exit()


#### VISUALIZE GPA ALIGNED SHAPES AND MEAN
print("\n--- Visualizing GPA Aligned Shapes and Mean ---")

if NUM_SAMPLES == 0:
    print("No samples to visualize. Exiting.")
    exit()

plt.figure(figsize=(8, 8))
for i in range(NUM_SAMPLES):
    plt.plot(proc_arr[i, :, 0], proc_arr[i, :, 1], c="k", alpha=0.08)

# The mean of the Procrustes-aligned array is the GPA mean
gpa_mean_shape = np.mean(proc_arr, axis=0)
plt.plot(gpa_mean_shape[:, 0], gpa_mean_shape[:, 1], c="magenta", linewidth=2)
plt.gca().set_aspect("equal")
plt.axis("off")
plt.title("Procrustes Aligned Leaf Shapes and GPA Mean")

plt.savefig(OUTPUT_BASE_DIR / GPA_MEAN_SHAPE_PLOT_FILENAME)
plt.close()
print(f"GPA mean shape plot saved to {OUTPUT_BASE_DIR / GPA_MEAN_SHAPE_PLOT_FILENAME}")


#################################################
### FIRST, CALCULATE PERCENT VARIANCE ALL PCs ###
#################################################

print("\n--- Performing Full PCA and Generating Explained Variance Report ---")

# Reshape the Procrustes array to a 2D matrix for PCA
flat_arr = proc_arr.reshape(NUM_SAMPLES, NUM_LANDMARKS * NUM_DIMENSIONS)

# Determine the maximum number of principal components possible: min(n_samples, n_features).
# PCA will calculate all components up to this maximum.
max_pc_components = min(flat_arr.shape[0], flat_arr.shape[1])

# Initialize PCA to calculate all possible PCs for full variance analysis
pca = PCA(n_components=max_pc_components)
PCs = pca.fit_transform(flat_arr) # Fit PCA and transform the data

# Generate and save explained variance report
pca_explained_variance_filepath = OUTPUT_BASE_DIR / PCA_EXPLAINED_VARIANCE_REPORT_FILENAME
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
pca_params_filepath = OUTPUT_BASE_DIR / PCA_PARAMS_H5_FILENAME
with h5py.File(pca_params_filepath, 'w') as f:
    f.create_dataset('components', data=pca_components, compression="gzip")
    f.create_dataset('mean', data=pca_mean, compression="gzip")
    f.create_dataset('explained_variance', data=pca_explained_variance, compression="gzip")
    f.create_dataset('explained_variance_ratio', data=pca_explained_variance_ratio, compression="gzip")
    f.attrs['n_components'] = n_pca_components
print(f"PCA parameters saved to {pca_params_filepath}")

# 3. Save original PCA scores (PCs) and class labels to an HDF5 file
pca_scores_labels_filepath = OUTPUT_BASE_DIR / ORIGINAL_PCA_SCORES_AND_LABELS_H5_FILENAME
with h5py.File(pca_scores_labels_filepath, 'w') as f:
    f.create_dataset('pca_scores', data=PCs, compression="gzip")
    # Convert labels to a numpy array of byte strings for HDF5 compatibility
    f.create_dataset('class_labels', data=np.array(mdata[CLASS_LABEL_COLUMN_FOR_SAVING]).astype('S'), compression="gzip")
    f.create_dataset('original_flattened_coords', data=flat_arr, compression="gzip")
print(f"Original PCA scores, class labels, AND original flattened coordinates saved to {pca_scores_labels_filepath}")


##########################
### CREATE MORPHOSPACE ###
##########################

print("\n--- Creating Morphospace Plot ---")

# Perform PCA specifically for morphospace visualization (2 components)
morphospace_pca = PCA(n_components=2)
morphospace_PCs = morphospace_pca.fit_transform(flat_arr)

# Add the 2-component PCA results to the mdata DataFrame
mdata["PC1"] = morphospace_PCs[:, 0]
mdata["PC2"] = morphospace_PCs[:, 1]

# Set up the plot
plt.figure(figsize=(MORPHOSPACE_PLOT_LENGTH, MORPHOSPACE_PLOT_WIDTH))
plt.gca().set_facecolor(FACE_COLOR)
plt.gca().set_axisbelow(True)

# Create PC intervals for plotting inverse eigenleaves
# Note: Using the ranges from the full PCA (PCs) for consistency in morphospace
PC1_vals = np.linspace(np.min(PCs[:, 0]), np.max(PCs[:, 0]), MORPHOSPACE_PC1_INTERVALS)
PC2_vals = np.linspace(np.min(PCs[:, 1]), np.max(PCs[:, 1]), MORPHOSPACE_PC2_INTERVALS)

# Plot inverse eigenleaves (the background grid shapes)
for i in PC1_vals:
    for j in PC2_vals:
        # Inverse transform current PC scores back to landmark space
        inv_leaf = morphospace_pca.inverse_transform(np.array([i, j]))
        # Reshape from flattened (1, N*D) to (N, D) landmark coordinates
        inv_leaf_coords = inv_leaf.reshape(NUM_LANDMARKS, NUM_DIMENSIONS) 

        inv_x = inv_leaf_coords[:, 0]
        inv_y = inv_leaf_coords[:, 1]

        # Plot the inverse eigenleaf, scaled and translated to its position in morphospace
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
plt.savefig(OUTPUT_BASE_DIR / MORPHOSPACE_PLOT_FILENAME, bbox_inches='tight')
plt.close()
print(f"Morphospace plot saved to {OUTPUT_BASE_DIR / MORPHOSPACE_PLOT_FILENAME}")

print("\n--- All morphometric analysis and saving completed ---")
#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import Path
import sys
import torch
import torch.nn as tnn
from torch.utils.data import Dataset, DataLoader
import copy

############################
### CONFIGURATION (SET THESE PARAMETERS) ###
############################

# --- Base Paths (relative to the script's location) ---
# Assuming the script is in COCA_PROJECT/data/RECIPROCAL_PREDICTION/
# This should point to COCA_PROJECT/data/
BASE_DIR = Path(__file__).parent.parent 

# Define the root directory where all data and models are now located
# Based on your latest output, it appears both data and models are organized under CULTIVATED2ND
# This simplifies things, as we just need one 'master' data/model root
COCA_PROJECT_DATA_ROOT = BASE_DIR / "CULTIVATED2ND"

# --- Model Paths ---
# Model trained on CULTIVATED1ST data, saved under COCA_PROJECT_DATA_ROOT/trained_models_Cultivated1st
MODEL_CULTIVATED1ST_PATH = COCA_PROJECT_DATA_ROOT / "trained_models_Cultivated1st" / "models" / "Cultivated1st_ECT_Mask_2Channel_CNN_Ensemble_Improved_fold0_best_model.pth"

# Model trained on CULTIVATED2ND data, saved under COCA_PROJECT_DATA_ROOT/trained_models_Cultivated2nd
MODEL_CULTIVATED2ND_PATH = COCA_PROJECT_DATA_ROOT / "trained_models_Cultivated2nd" / "models" / "Cultivated2nd_ECT_Mask_2Channel_CNN_Ensemble_Improved_fold0_best_model.pth"

# --- Data Paths ---
# Data for CULTIVATED1ST (to be predicted by CULTIVATED2ND's model)
# Now correctly points to the path under CULTIVATED2ND for 'cultivated1st' data
DATA_CULTIVATED1ST_PATH = COCA_PROJECT_DATA_ROOT / "05_synthetic_leaf_data_cultivated1st" / "final_cnn_dataset.pkl"

# Data for CULTIVATED2ND (to be predicted by CULTIVATED1ST's model)
DATA_CULTIVATED2ND_PATH = COCA_PROJECT_DATA_ROOT / "05_synthetic_leaf_data_cultivated2nd" / "final_cnn_dataset.pkl"


# --- General Prediction Configuration ---
BATCH_SIZE = 32

# --- Output Directories for Results ---
OUTPUT_DIR = Path("./results_reciprocal_prediction") # Create a dedicated folder for reciprocal results
os.makedirs(OUTPUT_DIR, exist_ok=True) # Create output directory if it doesn't exist

##########################
### DEVICE SETUP ###
##########################

# Ensure MPS is available (or use CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

###########################
### PYTORCH DATASET & MODEL (Identical to your training script) ###
###########################

class LeafDataset(Dataset):
    def __init__(self, images, labels, is_real_flags):
        self.images = images
        self.labels = labels
        self.is_real_flags = is_real_flags

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.is_real_flags[idx]

class LeafCNN(tnn.Module):
    def __init__(self, num_classes, image_size, num_input_channels):
        super(LeafCNN, self).__init__()
        self.features = tnn.Sequential(
            tnn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1),
            tnn.BatchNorm2d(32),
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),

            tnn.Conv2d(32, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64),
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),

            tnn.Conv2d(64, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            # Create a dummy model instance on CPU to calculate flattened_size
            temp_features_model = copy.deepcopy(self.features).to("cpu")
            dummy_input = torch.zeros(1, num_input_channels, image_size[0], image_size[1]).to("cpu")
            flattened_size = temp_features_model(dummy_input).view(1, -1).shape[1]
            del temp_features_model # Free memory

        self.classifier = tnn.Sequential(
            tnn.Flatten(),
            tnn.Linear(flattened_size, 512),
            tnn.ReLU(),
            tnn.Dropout(0.5),
            tnn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------------------------------------
## Helper function to load data and model
# ----------------------------------------------------
def load_data_and_model(model_path, data_path):
    """
    Loads data and the corresponding model.
    Returns: X_real_images_tensor, y_real_labels_tensor, class_names, model
    """
    # --- Load Data for Prediction ---
    print(f"Loading data from: {data_path}")
    try:
        with open(data_path, 'rb') as f:
            final_data = pickle.load(f)

        X_images = final_data['X_images']
        y_labels_encoded = final_data['y_labels_encoded']
        is_real_flags = final_data['is_real_flags']
        class_names = final_data['class_names']
        image_size_tuple = final_data['image_size']
        num_channels = final_data['num_channels']

        # Filter for real samples only
        real_indices = np.where(is_real_flags)[0]
        X_real_images = X_images[real_indices]
        y_real_labels_encoded = y_labels_encoded[real_indices]
        
        # Convert to PyTorch tensors and permute dimensions
        X_real_images_tensor = torch.from_numpy(X_real_images).float().permute(0, 3, 1, 2)
        y_real_labels_tensor = torch.from_numpy(y_real_labels_encoded).long()

        print(f"Loaded data shape (all samples): {X_images.shape}")
        print(f"Number of real samples for prediction: {len(real_indices)}")
        print(f"Number of classes in data: {len(class_names)}")
        print(f"Class names: {class_names}")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        sys.exit(1)

    # --- Load Model ---
    print(f"Loading model from: {model_path}")
    try:
        model = LeafCNN(num_classes=len(class_names), image_size=image_size_tuple, num_input_channels=num_channels).to(device)
        # Load the state dictionary, ensuring map_location is set correctly
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict']) # Access 'model_state_dict' from the checkpoint
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

    return X_real_images_tensor, y_real_labels_tensor, class_names, model, len(real_indices)


###########################
### PREDICTION FUNCTION ###
###########################

def predict_and_evaluate(model_obj, prediction_data_tensor, true_labels_tensor, class_names, script_name, output_prefix):
    """
    Performs prediction and evaluation for a given model on specified data.
    """
    print(f"\n--- Running {script_name} ---")

    # --- Prepare DataLoader for real samples ---
    if prediction_data_tensor.size(0) == 0:
        print("No real samples found in the dataset for evaluation. Skipping prediction.")
        return

    # Use a dummy is_real_flags as we've already filtered for real samples
    prediction_dataset = LeafDataset(prediction_data_tensor, true_labels_tensor, torch.ones_like(true_labels_tensor, dtype=torch.bool))
    prediction_loader = DataLoader(prediction_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Perform Predictions ---
    all_predictions_encoded = []
    all_true_labels_encoded = []

    print("\nStarting predictions...")
    with torch.no_grad():
        for images, labels, _ in prediction_loader:
            images = images.to(device)
            outputs = model_obj(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions_encoded.extend(predicted.cpu().numpy())
            all_true_labels_encoded.extend(labels.cpu().numpy())
    print("Predictions complete.")

    # --- Decode Labels and Evaluate ---
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names) # Ensure LabelEncoder is fitted with the consistent class names

    final_true_labels_names = label_encoder.inverse_transform(np.array(all_true_labels_encoded))
    final_predictions_names = label_encoder.inverse_transform(np.array(all_predictions_encoded))

    # --- Classification Report ---
    print(f"\n--- Classification Report for {script_name} ---")
    report = classification_report(final_true_labels_names, final_predictions_names, target_names=class_names, zero_division=0)
    print(report)

    # Save classification report to a text file
    report_path = OUTPUT_DIR / f"{output_prefix}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to: {report_path}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(final_true_labels_names, final_predictions_names, labels=class_names)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized[np.isnan(cm_normalized)] = 0 # Handle cases with no true samples for a class

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({script_name})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = OUTPUT_DIR / f"{output_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    # plt.show() # Commented out to prevent plot from popping up and pausing script execution
    print(f"Confusion matrix saved to: {cm_path}")

    # Plot and save normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Normalized Confusion Matrix ({script_name})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_norm_path = OUTPUT_DIR / f"{output_prefix}_normalized_confusion_matrix.png"
    plt.savefig(cm_norm_path, dpi=300)
    # plt.show() # Commented out
    print(f"Normalized confusion matrix saved to: {cm_norm_path}")

    print(f"\n--- {script_name} Complete ---")

# Run the predictions and evaluations
if __name__ == "__main__":
    print(f"Script location: {Path(__file__).resolve()}")
    print(f"Base project directory calculated: {BASE_DIR.resolve()}")
    print(f"Output directory for results: {OUTPUT_DIR.resolve()}")

    # --- Scenario 1: Cultivated1st Model predicting on Cultivated2nd Data ---
    print("\n======== Running RECIPROCAL PREDICTION: Cultivated1st Model on Cultivated2nd Data ========\n")
    X_data_CULTIVATED2ND, y_labels_CULTIVATED2ND, class_names_CULTIVATED2ND, model_CULTIVATED1ST, _ = \
        load_data_and_model(MODEL_CULTIVATED1ST_PATH, DATA_CULTIVATED2ND_PATH)
    
    predict_and_evaluate(
        model_CULTIVATED1ST,
        X_data_CULTIVATED2ND,
        y_labels_CULTIVATED2ND,
        class_names_CULTIVATED2ND,
        "Cultivated1st Model on Cultivated2nd Data",
        "Cultivated1st_on_Cultivated2nd"
    )

    # Clean up memory
    del X_data_CULTIVATED2ND, y_labels_CULTIVATED2ND, model_CULTIVATED1ST
    if torch.backends.mps.is_available(): # Only try to empty cache if MPS is available
        torch.mps.empty_cache()

    # --- Scenario 2: Cultivated2nd Model predicting on Cultivated1st Data ---
    print("\n======== Running RECIPROCAL PREDICTION: Cultivated2nd Model on Cultivated1st Data ========\n")
    X_data_CULTIVATED1ST, y_labels_CULTIVATED1ST, class_names_CULTIVATED1ST, model_CULTIVATED2ND, _ = \
        load_data_and_model(MODEL_CULTIVATED2ND_PATH, DATA_CULTIVATED1ST_PATH)
    
    predict_and_evaluate(
        model_CULTIVATED2ND,
        X_data_CULTIVATED1ST,
        y_labels_CULTIVATED1ST,
        class_names_CULTIVATED1ST,
        "Cultivated2nd Model on Cultivated1st Data",
        "Cultivated2nd_on_Cultivated1st"
    )

    print("\nAll reciprocal predictions and evaluations complete! ✨")
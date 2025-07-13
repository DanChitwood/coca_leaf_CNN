# predict_cultivated1st_on_plowman.py
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

# --- Paths Configuration ---
# Path to the trained model checkpoint (Cultivated1st's model)
# Assumes this script is in RECIPROCAL_PREDICTION/
# and the model is in CULTIVATED1ST/trained_models/
MODEL_PATH = Path("../CULTIVATED1ST/trained_models/ECT_Mask_2Channel_CNN_Ensemble_Improved_fold0_best_Leaf_Class.pth")

# Path to the data for prediction (Plowman's data)
# Assumes this script is in RECIPROCAL_PREDICTION/
# and the data is in PLOWMAN/05_synthetic_leaf_data/
DATA_PATH = Path("../PLOWMAN/05_synthetic_leaf_data/final_cnn_dataset.pkl")

# --- General Prediction Configuration ---
BATCH_SIZE = 32
SCRIPT_NAME = "Cultivated1st Model Predicting on Plowman Data"
OUTPUT_PREFIX = "Cultivated1st_on_Plowman" # Prefix for saved plots/reports

# --- Output Directories for Results ---
OUTPUT_DIR = Path("./results")
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

###########################
### PREDICTION FUNCTION ###
###########################

def predict_and_evaluate(model_path, data_path, script_name, output_prefix):
    print(f"\n--- Running {script_name} ---")

    # --- Load Data for Prediction ---
    print(f"Loading data from: {data_path}")
    try:
        with open(data_path, 'rb') as f:
            final_data = pickle.load(f)

        # Extract only real samples for prediction evaluation
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
        is_real_flag_tensor = torch.from_numpy(is_real_flags).bool() # Keep for dataset creation

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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

    # --- Prepare DataLoader for real samples ---
    if len(real_indices) == 0:
        print("No real samples found in the dataset for evaluation. Exiting.")
        sys.exit(0)

    prediction_dataset = LeafDataset(X_real_images_tensor, y_real_labels_tensor, torch.ones_like(y_real_labels_tensor, dtype=torch.bool))
    prediction_loader = DataLoader(prediction_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Perform Predictions ---
    all_predictions_encoded = []
    all_true_labels_encoded = []

    print("\nStarting predictions...")
    with torch.no_grad():
        for images, labels, _ in prediction_loader:
            images = images.to(device)
            outputs = model(images)
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
    plt.show()
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
    plt.show()
    print(f"Normalized confusion matrix saved to: {cm_norm_path}")

    print(f"\n--- {script_name} Complete ---")

# Run the prediction and evaluation
if __name__ == "__main__":
    predict_and_evaluate(MODEL_PATH, DATA_PATH, SCRIPT_NAME, OUTPUT_PREFIX)
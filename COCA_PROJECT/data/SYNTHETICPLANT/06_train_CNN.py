#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pickle
from pathlib import Path
import sys
import json

# PyTorch Imports
import torch
import torch.nn as tnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import cv2
from PIL import Image

############################
### CONFIGURATION (ALL PARAMETERS UP FRONT) ###
############################

# --- General Configuration ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
K_FOLDS = 5
PATIENCE = 10 # Early stopping patience (for validation loss)
LR_SCHEDULER_PATIENCE = 5 # Patience for ReduceLROnPlateau
LR_SCHEDULER_FACTOR = 0.1 # Factor by which to reduce LR
MODEL_IDENTIFIER = 'ECT_Mask_PlantID_Collection_CNN' # Updated identifier to reflect new input

# --- Data Input Configuration ---
# PATH_CHANGE: Point to the new plantID-level dataset
# Reverted to original name `final_cnn_dataset.pkl` as that's what was saved.
FINAL_PREPARED_DATA_FILE = Path("./05_synthetic_leaf_data/final_cnn_dataset.pkl")

# --- Model Saving Setup ---
MODEL_SAVE_DIR = "trained_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Output Directories for Figures/Data (NEW) ---
METRICS_SAVE_DIR = Path(MODEL_SAVE_DIR) / "metrics_output"
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)
CONFUSION_MATRIX_DATA_DIR = Path(MODEL_SAVE_DIR) / "confusion_matrix_data"
os.makedirs(CONFUSION_MATRIX_DATA_DIR, exist_ok=True)
GRAD_CAM_OUTPUT_DIR = Path(MODEL_SAVE_DIR) / "grad_cam_images"
os.makedirs(GRAD_CAM_OUTPUT_DIR, exist_ok=True)

# Grad-CAM specific configurations
# NOTE: Grad-CAM on 20 channels can be complex to interpret.
# We will adapt it to visualize on the first mask/ECT pair (channels 0 and 1).
# You might want to adjust this or customize further later.
NUM_SAMPLES_PER_CLASS_FOR_CAM_PLOT = 5

##########################
### DEVICE SETUP ###
##########################

# Ensure MPS is available (or use CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

###########################
### DATA LOADING ###
###########################

print("\n--- Loading data from FINAL_PREPARED_DATA_FILE ---")
try:
    with open(FINAL_PREPARED_DATA_FILE, 'rb') as f:
        final_data = pickle.load(f)

    # --- CORRECTED: Renamed variables to match the new structure from the previous script ---
    # `X_images` from the saving script now maps to `X_plantid_collections` here
    X_plantid_collections = final_data['X_images'] # (Num_PlantIDs, H, W, 2*n) numpy array
    
    # `y_labels_encoded` from the saving script now maps to `y_plantid_labels_encoded` here
    y_plantid_labels_encoded = final_data['y_labels_encoded'] # (Num_PlantIDs,) numpy array
    
    # `class_names` from the saving script now maps to `plantid_class_names` here
    plantid_class_names = final_data['class_names'] # List of string class names
    
    # `image_size` from the saving script now maps to `image_size_per_leaf` here
    image_size_per_leaf = final_data['image_size'] # (H, W) tuple, e.g., (256, 256)
    
    # `target_leaves_per_plantid` from the saving script now maps to `num_leaves_per_plantid` here
    num_leaves_per_plantid = final_data['target_leaves_per_plantid'] # int, should be 10
    
    # `num_channels_per_leaf` from saving script used to derive total_channels_per_plantid
    num_channels_per_leaf_actual = final_data['num_channels_per_leaf']
    total_channels_per_plantid = num_leaves_per_plantid * num_channels_per_leaf_actual # int, should be 2*10=20

    # `plant_ids` from the saving script can be used if you need to trace back
    # `plantid_is_fully_real_collection` was not explicitly saved, if needed, you might derive it
    # from the `synthetic_metadata.csv` or consider whether it's critical for CNN training logic.
    # For now, it's not directly used in the model's forward pass or loss calculation.
    
    # Recreate LabelEncoder from class_names for inverse_transform functionality
    label_encoder = LabelEncoder()
    label_encoder.fit(plantid_class_names) # Fit with the actual class names

    # Assuming classification is by 'PlantID_Variety' now, based on previous discussion
    target_column_used_for_data = 'PlantID_Variety'

    print(f"Loaded PlantID collection data shape: {X_plantid_collections.shape}")
    print(f"Number of unique PlantIDs (classes): {len(plantid_class_names)}")
    print(f"Image size per individual leaf: {image_size_per_leaf}")
    print(f"Number of leaves per PlantID collection: {num_leaves_per_plantid}")
    print(f"Total input channels per PlantID collection: {total_channels_per_plantid}")
    # print(f"Number of fully real PlantID collections: {np.sum(plantid_is_fully_real_collection)}") # Remove or re-derive if needed
    # print(f"Number of mixed/synthetic PlantID collections: {np.sum(~plantid_is_fully_real_collection)}") # Remove or re-derive if needed
    print(f"Data will be processed for classification of: '{target_column_used_for_data}'")

except FileNotFoundError:
    print(f"Error: Data file not found at {FINAL_PREPARED_DATA_FILE}.")
    print("Please ensure the data generation script has been run successfully and the output filename matches.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    sys.exit(1)

# --- PyTorch Data Preparation ---
# Permute dimensions for PyTorch: (N, H, W, C) -> (N, C, H, W)
X_collections_tensor = torch.from_numpy(X_plantid_collections).float().permute(0, 3, 1, 2)
y_encoded_tensor = torch.from_numpy(y_plantid_labels_encoded).long()
# `is_fully_real_collection_tensor` is not available from the `.pkl` file currently.
# If you need it for analysis later, you would need to modify the data generation script
# to explicitly save a boolean array indicating which plant_id collections are purely real.


print(f"Tensor PlantID collection data shape (after permute): {X_collections_tensor.shape}")

###########################
### PYTORCH DATASET & MODEL ###
###########################

class PlantIDDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # We no longer pass is_real_flags in __getitem__ as the K-Fold handles the splitting
        # on the _entire_ dataset (which is now plantID collections)
        return self.images[idx], self.labels[idx]

class LeafCNN(tnn.Module):
    def __init__(self, num_classes, image_size, num_input_channels):
        super(LeafCNN, self).__init__()
        # The number of input channels is now `total_channels_per_plantid` (e.g., 20)
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
            temp_features_model = self.features.to(device)
            dummy_input = torch.zeros(1, num_input_channels, image_size[0], image_size[1]).to(device)
            flattened_size = temp_features_model(dummy_input).view(1, -1).shape[1]
            temp_features_model.to("cpu") # Move back to CPU to free device memory

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

def save_model_checkpoint(model, optimizer, epoch, accuracy, model_identifier, target_column, fold_idx):
    filepath = os.path.join(MODEL_SAVE_DIR, f"{model_identifier}_fold{fold_idx}_best_{target_column}.pth")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(state, filepath)
    print(f"  --> Saved best model for Fold {fold_idx} (Acc: {accuracy:.4f})")

###########################################################
### PYTORCH CNN TRAINING AND EVALUATION (Ensemble with K-Fold) ###
###########################################################

print(f"\n--- Performing PyTorch CNN with {K_FOLDS}-Fold Stratified Cross-Validation (PlantID Collection Data) ---")

# The K-Fold split is now directly on the entire dataset of PlantID collections
# The `is_real_flags` logic for splitting is no longer necessary here, as we're training
# on a mixed dataset (real and synthetic plantID collections) and validating on *some* of them.
# The `plantid_is_fully_real_collection` flag is for analysis if needed later, but not for train/val split logic.

skf_pytorch = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

all_predictions_logits = [] # To store logits for ensemble prediction
all_true_labels_encoded = [] # To store true labels for ensemble evaluation

saved_model_paths_per_fold = [None] * K_FOLDS

# --- Calculate class weights for imbalanced dataset ---
# Use the labels from the entire plantID collection dataset for calculating class weights
all_labels_for_weights = y_encoded_tensor.cpu().numpy()
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(all_labels_for_weights),
    y=all_labels_for_weights
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"\nCalculated class weights: {class_weights_tensor.cpu().numpy()}")


# Split the entire plantID collection dataset into K folds
for fold_idx, (train_indices, val_indices) in enumerate(skf_pytorch.split(X_collections_tensor.cpu().numpy(), y_encoded_tensor.cpu().numpy())):
    print(f"\n--- Fold {fold_idx + 1}/{K_FOLDS} ---")

    # Define training and validation datasets for the current fold
    X_train_img_fold_tensor = X_collections_tensor[train_indices]
    y_train_fold_tensor = y_encoded_tensor[train_indices]
    train_dataset = PlantIDDataset(X_train_img_fold_tensor, y_train_fold_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    X_val_img_fold = X_collections_tensor[val_indices]
    y_val_fold = y_encoded_tensor[val_indices]
    val_dataset = PlantIDDataset(X_val_img_fold, y_val_fold)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model, criterion (with weights), optimizer, and scheduler for the current fold
    # Use total_channels_per_plantid for num_input_channels
    model = LeafCNN(num_classes=len(plantid_class_names), image_size=image_size_per_leaf, num_input_channels=total_channels_per_plantid).to(device)
    criterion = tnn.CrossEntropyLoss(weight=class_weights_tensor) # Using weighted loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=True) # LR scheduler

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_overall_accuracy_for_saving_this_fold = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader: # Removed is_real_flag from here
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        current_overall_accuracy_on_val_samples = 0.0 # Renamed for clarity

        with torch.no_grad():
            for images, labels in val_loader: # Removed is_real_flag from here
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            avg_train_loss = running_loss / len(train_loader.dataset)
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = correct_predictions / total_samples

            current_overall_accuracy_on_val_samples = val_accuracy # Using the accuracy on the current fold's validation set

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == NUM_EPOCHS -1:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f} (PlantID Collections)")

        # Step the LR scheduler
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            # Save the model if this is the best validation accuracy for this fold
            if current_overall_accuracy_on_val_samples > best_overall_accuracy_for_saving_this_fold:
                best_overall_accuracy_for_saving_this_fold = current_overall_accuracy_on_val_samples
                save_model_checkpoint(model, optimizer, epoch, best_overall_accuracy_for_saving_this_fold, MODEL_IDENTIFIER, target_column_used_for_data, fold_idx)
                saved_model_paths_per_fold[fold_idx] = os.path.join(MODEL_SAVE_DIR, f"{MODEL_IDENTIFIER}_fold{fold_idx}_best_{target_column_used_for_data}.pth")

        else:
            epochs_no_improve += 1
            if epochs_no_improve == PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

    model.load_state_dict(best_model_wts)
    print(f"Fold {fold_idx + 1} training complete. Best validation loss for fold: {best_val_loss:.4f}")

    # Predict logits for the validation set of this fold using the best model of this fold
    # These are the logits for the samples that form the basis of the ensemble's final evaluation
    model.eval()
    fold_val_predictions_logits = []
    fold_val_true_labels = []

    with torch.no_grad():
        for images_batch, labels_batch in val_loader: # Removed is_real_flag from here
            images_batch = images_batch.to(device)
            outputs = model(images_batch)
            fold_val_predictions_logits.append(outputs.cpu().numpy())
            fold_val_true_labels.append(labels_batch.cpu().numpy())

    all_predictions_logits.append(np.concatenate(fold_val_predictions_logits, axis=0))
    all_true_labels_encoded.append(np.concatenate(fold_val_true_labels, axis=0))


#########################################################
### FINAL ENSEMBLE EVALUATION ON ALL DATA ###
#########################################################

# We now evaluate the ensemble on the combined predictions from the validation sets of each fold.
# This represents a full evaluation across the entire plantID collection dataset,
# as each sample appeared in exactly one validation set across the K folds.
print("\n--- Final Ensemble Evaluation on ALL PlantID Collections ---")

# Concatenate all true labels and predictions from the validation sets
final_true_labels_encoded = np.concatenate(all_true_labels_encoded, axis=0)
averaged_logits = np.concatenate(all_predictions_logits, axis=0)
final_predictions_encoded = np.argmax(averaged_logits, axis=1)

final_true_labels_names = label_encoder.inverse_transform(final_true_labels_encoded)
final_predictions_names = label_encoder.inverse_transform(final_predictions_encoded)

overall_accuracy_pt = accuracy_score(final_true_labels_names, final_predictions_names)
print(f"\n--- Overall Accuracy ({MODEL_IDENTIFIER} Ensemble, Evaluated on ALL PlantID Collections - {target_column_used_for_data}): {overall_accuracy_pt:.4f} ---")

print(f"\n--- Classification Report ({MODEL_IDENTIFIER} Ensemble, Evaluated on ALL PlantID Collections - {target_column_used_for_data}) ---")
report_dict = classification_report(final_true_labels_names, final_predictions_names, target_names=plantid_class_names, zero_division=0, output_dict=True)
print(classification_report(final_true_labels_names, final_predictions_names, target_names=plantid_class_names, zero_division=0))

# --- NEW: Save Classification Report to JSON ---
metrics_output_path = METRICS_SAVE_DIR / f"{MODEL_IDENTIFIER}_classification_report_{target_column_used_for_data}.json"
with open(metrics_output_path, 'w') as f:
    json.dump(report_dict, f, indent=4)
print(f"Classification report saved to: {metrics_output_path}")

cm_pt = confusion_matrix(final_true_labels_names, final_predictions_names, labels=plantid_class_names)

# --- NEW: Save True and Predicted Labels for Confusion Matrix Plotting ---
np.save(CONFUSION_MATRIX_DATA_DIR / f"{MODEL_IDENTIFIER}_true_labels_{target_column_used_for_data}.npy", final_true_labels_names)
np.save(CONFUSION_MATRIX_DATA_DIR / f"{MODEL_IDENTIFIER}_predicted_labels_{target_column_used_for_data}.npy", final_predictions_names)
print(f"True and predicted labels for confusion matrix saved to {CONFUSION_MATRIX_DATA_DIR}.")


# Existing Confusion Matrix Plots (for immediate visual confirmation)
plt.figure(figsize=(16, 14))
sns.heatmap(cm_pt, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=plantid_class_names, yticklabels=plantid_class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix ({MODEL_IDENTIFIER} Ensemble, Evaluated on ALL PlantID Collections - {target_column_used_for_data})')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, f"{MODEL_IDENTIFIER}_ConfusionMatrix_{target_column_used_for_data}.png"), dpi=300)
plt.show()

cm_normalized_pt = cm_pt.astype('float') / cm_pt.sum(axis=1)[:, np.newaxis]
cm_normalized_pt[np.isnan(cm_normalized_pt)] = 0

plt.figure(figsize=(16, 14))
sns.heatmap(cm_normalized_pt, annot=True, fmt='.2f', cmap='Blues', cbar=True,
            xticklabels=plantid_class_names, yticklabels=plantid_class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Normalized Confusion Matrix ({MODEL_IDENTIFIER} Ensemble, Evaluated on ALL PlantID Collections - {target_column_used_for_data})')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, f"{MODEL_IDENTIFIER}_NormalizedConfusionMatrix_{target_column_used_for_data}.png"), dpi=300)
plt.show()

# Global results storage remains the same for consistency
if 'results_storage' not in globals():
    results_storage = {}

if target_column_used_for_data not in results_storage:
    results_storage[target_column_used_for_data] = {
        'class_counts': {},
        'model_metrics': {}
    }
print(f"Global target_column_used_for_data for this session: '{target_column_used_for_data}'")

MODEL_NAME = MODEL_IDENTIFIER
if target_column_used_for_data not in results_storage:
    results_storage[target_column_used_for_data] = {'class_counts': {}, 'model_metrics': {}}

if not results_storage[target_column_used_for_data]['class_counts']:
    class_counts_series = pd.Series(final_true_labels_encoded).value_counts().sort_index()
    for encoded_label, count in class_counts_series.items():
        class_name_str = label_encoder.inverse_transform([encoded_label])[0]
        results_storage[target_column_used_for_data]['class_counts'][class_name_str] = count
    print(f"Class counts populated for '{target_column_used_for_data}'.")

results_storage[target_column_used_for_data]['model_metrics'][MODEL_NAME] = {
    'precision': {cls: report_dict[cls]['precision'] for cls in plantid_class_names}, # Changed class_names to plantid_class_names
    'recall': {cls: report_dict[cls]['recall'] for cls in plantid_class_names}, # Changed class_names to plantid_class_names
    'f1-score': {cls: report_dict[cls]['f1-score'] for cls in plantid_class_names}, # Changed class_names to plantid_class_names
    'accuracy': report_dict['accuracy'],
    'macro avg precision': report_dict['macro avg']['precision'],
    'macro avg recall': report_dict['macro avg']['recall'],
    'macro avg f1-score': report_dict['macro avg']['f1-score'],
    'weighted avg precision': report_dict['weighted avg']['precision'],
    'weighted avg recall': report_dict['weighted avg']['recall'],
    'weighted avg f1-score': report_dict['weighted avg']['f1-score'],
}
print(f"Metrics for '{MODEL_NAME}' stored in results_storage for '{target_column_used_for_data}'.")

print("\n--- Current contents of results_storage (should include new model metrics) ---")
print(results_storage)


###################################
### GRAD-CAM VISUALIZATION ###
###################################

print(f"\n--- Generating Average Grad-CAM Visualizations for {MODEL_IDENTIFIER} (Model from Fold 0) ---")

# Ensure a model was saved for Fold 0 before attempting Grad-CAM
if len(saved_model_paths_per_fold) <= 0 or saved_model_paths_per_fold[0] is None or not os.path.exists(saved_model_paths_per_fold[0]):
    print(f"Error: Model file for Fold 0 not found or not saved at {saved_model_paths_per_fold[0] if len(saved_model_paths_per_fold) > 0 else 'N/A'}. Skipping Grad-CAM visualization.")
else:
    model_to_visualize_path = saved_model_paths_per_fold[0]

    # Initialize Grad-CAM model with new num_input_channels
    cam_model = LeafCNN(num_classes=len(plantid_class_names), image_size=image_size_per_leaf, num_input_channels=total_channels_per_plantid).to(device)
    checkpoint = torch.load(model_to_visualize_path, map_location=device)
    cam_model.load_state_dict(checkpoint['model_state_dict'])
    cam_model.eval()

    # Target the last convolutional layer in the `features` sequential block
    target_layer = cam_model.features[-3]


    class GradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None

            found_layer = False
            for name, module in self.model.named_modules():
                if module is self.target_layer:
                    module.register_forward_hook(self._save_activation)
                    module.register_backward_hook(self._save_gradient)
                    found_layer = True
                    break
            if not found_layer:
                raise ValueError(f"Target layer {target_layer} not found in model named modules.")

        def _save_activation(self, module, input, output):
            self.activations = output

        def _save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def __call__(self, input_tensor, target_class=None):
            # input_tensor is (1, C, H, W)
            self.model.zero_grad()
            output = self.model(input_tensor) # output is (1, num_classes)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item() # Use predicted class if not specified

            # Compute gradients with respect to target_class
            one_hot = torch.zeros_like(output).to(device)
            one_hot[0][target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True) # Retain graph for multiple calls if needed

            # Get gradients and activations
            gradients = self.gradients[0].cpu().data.numpy() # (num_features, H_conv, W_conv)
            activations = self.activations[0].cpu().data.numpy() # (num_features, H_conv, W_conv)

            # Pool the gradients to get weights for each feature map
            weights = np.mean(gradients, axis=(1, 2)) # (num_features,)
            
            # Compute CAM
            cam = np.zeros(activations.shape[1:], dtype=np.float32) # (H_conv, W_conv)
            for i, w in enumerate(weights):
                cam += w * activations[i]

            cam = np.maximum(cam, 0) # Apply ReLU to the CAM
            cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3])) # Resize to input image size
            
            # Normalize CAM to 0-1
            cam = cam - np.min(cam)
            if np.max(cam) == 0:
                cam = np.zeros_like(cam)
            else:
                cam = cam / np.max(cam)
            return cam

    def show_cam_on_black_background(cam_heatmap, original_image_tensor, image_size_tuple):
        # original_image_tensor is (C, H, W), where C is 20
        # For visualization, we will show the ECT channel of the *first leaf* in the collection.
        # Channels are interleaved: Mask1, ECT1, Mask2, ECT2, ...
        # So, ECT of the first leaf is at index 1.
        
        # Extract the ECT channel of the first leaf for visualization
        ect_channel = original_image_tensor[1, :, :].cpu().numpy() # ECT of first leaf is channel 1

        # Normalize ECT channel to 0-1 for display
        ect_channel_display = ect_channel - ect_channel.min()
        if ect_channel_display.max() > 0:
            ect_channel_display = ect_channel_display / ect_channel_display.max()
        else:
            ect_channel_display = np.zeros_like(ect_channel_display) # Handle all-zero case

        # Create a 3-channel grayscale image from the ECT for overlay
        img_display_base = np.stack([ect_channel_display, ect_channel_display, ect_channel_display], axis=-1)
        img_display_base = np.uint8(255 * img_display_base) # Convert to 0-255 for OpenCV

        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)
        heatmap_colored = np.float32(heatmap_colored) / 255 # Normalize heatmap to 0-1

        # Blend the heatmap with the original image
        alpha = 0.5 # Transparency factor for the heatmap
        final_cam_img = np.uint8(255 * (heatmap_colored * alpha + np.float32(img_display_base) / 255 * (1-alpha)))

        return final_cam_img


    grad_cam = GradCAM(cam_model, target_layer)
    average_class_heatmaps = {}

    # Gather indices of samples for each class for Grad-CAM
    # We will pick a few representative samples (plantID collections) for each class
    # from the overall dataset to calculate average CAM.
    indices_by_class = {cls_idx: [] for cls_idx in range(len(plantid_class_names))}
    for idx in range(len(y_encoded_tensor)):
        class_label = y_encoded_tensor[idx].item()
        indices_by_class[class_label].append(idx)

    print("Calculating average Grad-CAM heatmaps per class...")
    for class_idx in range(len(plantid_class_names)):
        class_name = plantid_class_names[class_idx]
        class_samples_indices = indices_by_class[class_idx]

        if not class_samples_indices:
            print(f"  No samples for class '{class_name}'. Skipping average Grad-CAM.")
            average_class_heatmaps[class_idx] = None
            continue

        summed_heatmap = np.zeros(image_size_per_leaf, dtype=np.float32)
        count_for_average = 0

        # Select a random subset of samples for CAM plotting to avoid excessive computation
        samples_for_cam = np.random.choice(class_samples_indices, min(NUM_SAMPLES_PER_CLASS_FOR_CAM_PLOT, len(class_samples_indices)), replace=False)

        for sample_idx in samples_for_cam:
            image_tensor = X_collections_tensor[sample_idx]
            input_image_for_cam = image_tensor.unsqueeze(0).to(device) # Add batch dimension

            heatmap = grad_cam(input_image_for_cam, target_class=class_idx) # Compute CAM for the specific class
            
            summed_heatmap += heatmap
            count_for_average += 1
            
        if count_for_average > 0:
            avg_heatmap = summed_heatmap / count_for_average
            avg_heatmap = avg_heatmap - np.min(avg_heatmap) # Normalize to 0-1
            if np.max(avg_heatmap) == 0:
                avg_heatmap = np.zeros_like(avg_heatmap)
            else:
                avg_heatmap = avg_heatmap / np.max(avg_heatmap)
            average_class_heatmaps[class_idx] = avg_heatmap
            print(f"  Calculated average for class: '{class_name}' ({count_for_average} samples)")
        else:
            average_class_heatmaps[class_idx] = None


    num_plots_total = len(plantid_class_names)
    num_cols_grid = math.ceil(math.sqrt(num_plots_total))
    num_rows_grid = math.ceil(num_plots_total / num_cols_grid)

    fig_width = num_cols_grid * 3.0
    fig_height = num_rows_grid * 3.5

    sns.set_style("white")
    plt.rcParams.update({'font.size': 10})

    fig, axes = plt.subplots(num_rows_grid, num_cols_grid, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes.flatten()

    print(f"\nPlotting average Grad-CAMs in a {num_rows_grid}x{num_cols_grid} grid...")

    for i in range(len(plantid_class_names)):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(plantid_class_names[i], fontsize=10)

        avg_heatmap = average_class_heatmaps[i]
        if avg_heatmap is not None:
            # Get an example image for background, take the first one available for this class
            if indices_by_class[i]:
                example_image_tensor = X_collections_tensor[indices_by_class[i][0]]
                cam_image_on_background = show_cam_on_black_background(avg_heatmap, example_image_tensor, image_size_per_leaf)
                ax.imshow(cam_image_on_background)
                
                # --- NEW: Save individual Grad-CAM image (no text/axes) ---
                individual_cam_output_path = GRAD_CAM_OUTPUT_DIR / f"{MODEL_IDENTIFIER}_GradCAM_{plantid_class_names[i].replace(' ', '_')}.png"
                
                # Create a clean figure for saving the individual CAM image
                fig_single = plt.figure(figsize=(image_size_per_leaf[0]/100, image_size_per_leaf[1]/100), dpi=100)
                ax_single = fig_single.add_subplot(111)
                ax_single.imshow(cam_image_on_background)
                ax_single.set_axis_off() # Turn off axes
                ax_single.set_position([0,0,1,1]) # Set to occupy entire figure
                fig_single.savefig(individual_cam_output_path, bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close(fig_single) # Close the figure to free memory
                print(f"  Saved individual Grad-CAM for class '{plantid_class_names[i]}' to: {individual_cam_output_path}")

            else:
                ax.text(0.5, 0.5, 'No Samples', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Samples', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray', fontsize=10)


    for j in range(num_plots_total, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Average Grad-CAM Visualizations ({MODEL_IDENTIFIER}, Target: {target_column_used_for_data})', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f"{MODEL_IDENTIFIER}_AverageGradCAM_{target_column_used_for_data}.png"), dpi=300)
    plt.show()
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
# Updated MODEL_IDENTIFIER_BASE for binary collection classification
MODEL_IDENTIFIER_BASE = 'ECT_Mask_2Channel_CNN_Ensemble_CollectionBinary'

# --- Dataset-Specific Configuration for Loading ---
# This dictionary holds configurations for each dataset you want to train on.
# These paths point to the outputs of 05_synthetic_data_preparation.py
CNN_DATASET_LOAD_CONFIGS = {
    "cultivated1st": {
        "DATASET_NICKNAME": "Cultivated1st",
        "FINAL_PREPARED_DATA_FILE": Path("./05_synthetic_leaf_data_cultivated1st/final_cnn_dataset.pkl"),
    },
    "cultivated2nd": {
        "DATASET_NICKNAME": "Cultivated2nd",
        "FINAL_PREPARED_DATA_FILE": Path("./05_synthetic_leaf_data_cultivated2nd/final_cnn_dataset.pkl"),
    }
}

# Grad-CAM specific configurations
NUM_SAMPLES_PER_CLASS_FOR_CAM_PLOT = 5

##########################
### DEVICE SETUP ###
##########################

# Ensure MPS is available (or use CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

###########################
### PYTORCH DATASET & MODEL ###
###########################

class LeafDataset(Dataset):
    def __init__(self, images, labels, is_real_flags):
        self.images = images
        self.labels = labels
        self.is_real_flags = is_real_flags

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Ensure labels are float32 for BCEWithLogitsLoss and have correct shape [1]
        return self.images[idx], self.labels[idx].float().unsqueeze(0), self.is_real_flags[idx]

class LeafCNN(tnn.Module):
    def __init__(self, num_classes, image_size, num_input_channels):
        super(LeafCNN, self).__init__()
        # Added Batch Normalization layers
        self.features = tnn.Sequential(
            tnn.Conv2d(num_input_channels, 32, kernel_size=3, padding=1),
            tnn.BatchNorm2d(32), # Added Batch Norm
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),

            tnn.Conv2d(32, 64, kernel_size=3, padding=1),
            tnn.BatchNorm2d(64), # Added Batch Norm
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),

            tnn.Conv2d(64, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128), # Added Batch Norm
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, num_input_channels, image_size[0], image_size[1]).cpu()
            flattened_size = self.features.cpu()(dummy_input).view(1, -1).shape[1]

        # num_classes will be 1 for binary classification
        self.classifier = tnn.Sequential(
            tnn.Flatten(),
            tnn.Linear(flattened_size, 512),
            tnn.ReLU(),
            tnn.Dropout(0.5),
            tnn.Linear(512, num_classes) # num_classes should be 1 for binary
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def save_model_checkpoint(model, optimizer, epoch, val_loss, model_identifier, fold_idx, model_save_dir):
    filepath = os.path.join(model_save_dir, f"{model_identifier}_fold{fold_idx}_best_val_loss_{val_loss:.4f}.pth")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(state, filepath)
    print(f"  --> Saved best model for Fold {fold_idx} (Val Loss: {val_loss:.4f}) to {filepath}")
    return filepath

# Grad-CAM utility class (copied directly as it's self-contained and effective)
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
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # For binary classification, target_class will be 0 or 1.
        # We need to compute gradients with respect to the output for the target class.
        # BCEWithLogitsLoss doesn't give a 'target_class' in the same way CrossEntropyLoss does for argmax.
        # For Grad-CAM with BCEWithLogitsLoss, typically you backpropagate through the *predicted class* score.
        # If target_class is None, we assume it's the class with higher probability (closer to 1 if binary).
        # Here, for binary, we treat the single output as the logit for class 1.
        # If the target_class is 0, we'd want to maximize (1-sigmoid(output)), or minimize sigmoid(output)
        # For simplicity for Grad-CAM, let's just backpropagate through the output logit directly.
        # If the target is class 0, we can use 1 - sigmoid(output) as the "score" to backpropagate.
        
        # Original:
        # if target_class is None:
        #    target_class = output.argmax(dim=1).item()
        # one_hot = torch.zeros_like(output).to(device)
        # one_hot[0][target_class] = 1
        # output.backward(gradient=one_hot, retain_graph=True)

        # Adjusted for binary classification (assuming output is single logit for class 1):
        if target_class is None: # Use the predicted class if not specified
             predicted_class_logit = output.item()
             target_class = 1 if predicted_class_logit >= 0 else 0 # Threshold 0 for logit

        if target_class == 1: # Maximize the logit for class 1
            output_scalar = output.squeeze() # Get a scalar from the [1,1] tensor
            output_scalar.backward(retain_graph=True)
        elif target_class == 0: # Minimize the logit for class 1 (or maximize -(logit))
            # Backpropagate through -output to find gradients that lead to class 0
            (-output.squeeze()).backward(retain_graph=True)
        else:
            raise ValueError("target_class for binary Grad-CAM must be 0 or 1.")


        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        return cam

def show_cam_on_black_background(cam_heatmap, original_image_tensor, image_size_tuple):
    # original_image_tensor is (C, H, W)
    # Assuming ECT is channel 1 (index 1) and Mask is channel 0 (index 0)
    ect_channel = original_image_tensor[1, :, :].cpu().numpy() # ECT is channel 1

    # Normalize ECT channel to 0-1 for display, if not already
    ect_channel_display = ect_channel - ect_channel.min()
    if ect_channel_display.max() > 0:
        ect_channel_display = ect_channel_display / ect_channel_display.max()
    else:
        ect_channel_display = np.zeros_like(ect_channel_display) # Handle all-zero case

    img_display_base = np.stack([ect_channel_display, ect_channel_display, ect_channel_display], axis=-1)
    img_display_base = np.uint8(255 * img_display_base)

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255

    alpha = 0.5
    final_cam_img = np.uint8(255 * (heatmap_colored * alpha + np.float32(img_display_base) / 255 * (1-alpha)))

    return final_cam_img

# Global storage for results
results_storage = {}

def run_cnn_training_for_collection_binary(configs: dict):
    """
    Runs the full CNN training and evaluation pipeline for binary collection classification.
    """
    print(f"\n{'='*10} Starting CNN Training for Collection (1st vs 2nd) Binary Classification {'='*10}")

    # --- Setup Dynamic Paths and Identifiers for Collection Binary Dataset ---
    dataset_nickname = "CollectionBinary" # Identifier for combined output
    model_identifier = f"{dataset_nickname}_{MODEL_IDENTIFIER_BASE}"

    # Create dataset-specific output directories for combined results
    base_output_dir = Path(f"./trained_models_{dataset_nickname}/")
    model_save_dir = base_output_dir / "models"
    metrics_save_dir = base_output_dir / "metrics"
    confusion_matrix_data_dir = base_output_dir / "confusion_matrix_data"
    grad_cam_output_dir = base_output_dir / "grad_cam_images"

    for d in [model_save_dir, metrics_save_dir, confusion_matrix_data_dir, grad_cam_output_dir]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories for Collection Binary Classification under {base_output_dir}")

    # --- Data Loading and Combination with New Binary Labels ---
    all_X_images = []
    all_y_binary_labels = [] # Will store 0 for _1st, 1 for _2nd
    all_is_real_flags = []
    
    # Store initial image_size and num_channels to ensure consistency
    first_image_size = None
    first_num_channels = None

    print(f"\n--- Loading and combining data from multiple datasets with binary collection labels ---")
    for ds_name, ds_config in configs.items():
        data_file_path = ds_config['FINAL_PREPARED_DATA_FILE']
        collection_label = 0 if ds_name == "cultivated1st" else 1 # 0 for 1st, 1 for 2nd
        print(f"Loading {ds_name} data from {data_file_path} and assigning label '{collection_label}'...")
        try:
            with open(data_file_path, 'rb') as f:
                data = pickle.load(f)

            if first_image_size is None:
                first_image_size = data['image_size']
                first_num_channels = data['num_channels']
            else:
                if data['image_size'] != first_image_size:
                    raise ValueError(f"Image sizes do not match across datasets: {first_image_size} vs {data['image_size']}")
                if data['num_channels'] != first_num_channels:
                    raise ValueError(f"Number of channels do not match across datasets: {first_num_channels} vs {data['num_channels']}")

            all_X_images.append(data['X_images'])
            all_is_real_flags.append(data['is_real_flags'])
            
            # Create binary labels for all samples from this collection
            all_y_binary_labels.extend([collection_label] * len(data['X_images']))
            
            print(f"  Loaded {len(data['X_images'])} samples from {ds_name}.")

        except FileNotFoundError:
            print(f"Error: Data file not found at {data_file_path}. Skipping this dataset.")
            sys.exit(1) # Exit if critical data is missing
        except Exception as e:
            print(f"An error occurred while loading {ds_name} data: {e}. Skipping this dataset.")
            sys.exit(1) # Exit if critical data loading fails

    # Consolidate all loaded data
    X_images = np.concatenate(all_X_images, axis=0)
    is_real_flags = np.concatenate(all_is_real_flags, axis=0)
    y_labels_encoded = np.array(all_y_binary_labels) # These are already 0s and 1s

    # Define the two class names
    class_names = ['Collection_1st', 'Collection_2nd']
    num_classes = 1 # Output dimension of the model is 1 for binary classification

    image_size_tuple = first_image_size
    num_channels = first_num_channels

    print(f"\n--- Collection Binary Dataset Statistics ---")
    print(f"Total loaded image data shape: {X_images.shape}")
    print(f"Number of classes (binary collection): {len(class_names)}")
    print(f"Class names (binary collection): {class_names}")
    print(f"Image size: {image_size_tuple}")
    print(f"Number of channels: {num_channels}")
    print(f"Total number of real samples: {np.sum(is_real_flags)}")
    print(f"Total number of synthetic samples: {np.sum(~is_real_flags)}")
    print(f"Data will be processed for classification of: 'Leaf_Collection_Binary'")

    # --- PyTorch Data Preparation ---
    X_images_tensor = torch.from_numpy(X_images).float().permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
    y_encoded_tensor = torch.from_numpy(y_labels_encoded).long() # Labels should be long for indexing if needed, but float for loss

    is_real_flag_tensor = torch.from_numpy(is_real_flags).bool()

    print(f"Tensor image data shape (after permute): {X_images_tensor.shape}")

    # --- PyTorch CNN Training and Evaluation (Ensemble with K-Fold) ---
    print(f"\n--- Performing PyTorch CNN with {K_FOLDS}-Fold Stratified Cross-Validation (2-Channel Image Data) on Collection Binary Dataset ---")

    # Separate real samples for K-Fold splitting and validation (ONLY REAL SAMPLES)
    real_original_indices_global = torch.where(is_real_flag_tensor)[0].cpu().numpy()

    X_original_images_for_skf = X_images_tensor[real_original_indices_global]
    y_original_for_skf = y_encoded_tensor[real_original_indices_global]

    # Stratified K-Fold splitting should be based on the NEW binary collection labels
    skf_pytorch = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42) # Consider adding random seed here for reproducibility

    all_predictions_logits = []
    saved_model_paths_per_fold = [None] * K_FOLDS

    # --- Calculate pos_weight for BCEWithLogitsLoss (on ALL training labels) ---
    # Count samples for each class in the combined dataset (real + synthetic)
    num_neg_samples = torch.sum(y_encoded_tensor == 0).item() # Count of Collection_1st samples
    num_pos_samples = torch.sum(y_encoded_tensor == 1).item() # Count of Collection_2nd samples

    if num_pos_samples > 0:
        pos_weight_value = torch.tensor(num_neg_samples / num_pos_samples, dtype=torch.float32).to(device)
    else: # Handle case where there are no positive samples (shouldn't happen with synthetic data)
        pos_weight_value = torch.tensor(1.0, dtype=torch.float32).to(device) # Default to 1 if no positive samples

    print(f"\nCalculated pos_weight for BCEWithLogitsLoss: {pos_weight_value.item():.4f} (based on combined real and synthetic data)")


    for fold_idx, (train_original_real_indices, val_original_real_indices) in enumerate(skf_pytorch.split(X_original_images_for_skf.cpu().numpy(), y_original_for_skf.cpu().numpy())):
        print(f"\n--- Fold {fold_idx + 1}/{K_FOLDS} ---")

        # Validation set: ONLY real data from the current fold's validation split
        X_val_img_fold = X_original_images_for_skf[val_original_real_indices]
        y_val_fold = y_original_for_skf[val_original_real_indices]
        val_dataset = LeafDataset(X_val_img_fold, y_val_fold, torch.ones_like(y_val_fold, dtype=torch.bool))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Training set: All synthetic data + real data from the current fold's training split
        synthetic_indices = torch.where(~is_real_flag_tensor)[0].cpu().numpy()
        global_real_train_indices = real_original_indices_global[train_original_real_indices]
        all_training_indices_global = np.concatenate((global_real_train_indices, synthetic_indices))

        X_train_img_fold_tensor = X_images_tensor[all_training_indices_global]
        y_train_fold_tensor = y_encoded_tensor[all_training_indices_global]
        is_real_train_fold_tensor = is_real_flag_tensor[all_training_indices_global]

        train_dataset = LeafDataset(X_train_img_fold_tensor, y_train_fold_tensor, is_real_train_fold_tensor)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # Initialize model, criterion, optimizer, and scheduler for the current fold
        model = LeafCNN(num_classes=num_classes, image_size=image_size_tuple, num_input_channels=num_channels).to(device)
        criterion = tnn.BCEWithLogitsLoss(pos_weight=pos_weight_value) # Using BCEWithLogitsLoss with pos_weight
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=True) # LR scheduler

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        best_model_path_for_fold = None # To store path to the saved model

        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images) # outputs are logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images) # outputs are logits
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)

                    probabilities = torch.sigmoid(outputs) # Convert logits to probabilities
                    # Binary predictions: 0 if prob < 0.5, 1 if prob >= 0.5
                    predicted_classes = (probabilities >= 0.5).float()
                    
                    total_samples += labels.size(0)
                    correct_predictions += (predicted_classes == labels).sum().item()

                avg_train_loss = running_loss / len(train_loader.dataset)
                avg_val_loss = val_loss / len(val_loader.dataset)
                val_accuracy = correct_predictions / total_samples

                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == NUM_EPOCHS -1:
                    print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f} (Real Samples)")

            # Step the LR scheduler
            scheduler.step(avg_val_loss)

            # Early stopping and best model saving logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                
                best_model_path_for_fold = save_model_checkpoint(
                    model, optimizer, epoch, best_val_loss, model_identifier, fold_idx, model_save_dir
                )
                saved_model_paths_per_fold[fold_idx] = best_model_path_for_fold

            else:
                epochs_no_improve += 1
                if epochs_no_improve == PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                    break

    model.load_state_dict(best_model_wts)
    print(f"Fold {fold_idx + 1} training complete. Best validation loss for fold: {best_val_loss:.4f}")

    # Predict logits for ALL real samples using the best model of this fold
    model.eval()
    fold_predictions_logits = [] # Will store raw logits

    # Note: y_original_for_skf are the original (0 or 1) integer labels
    real_dataset_for_pred = LeafDataset(X_original_images_for_skf, y_original_for_skf, torch.ones_like(y_original_for_skf, dtype=torch.bool))
    real_loader_for_pred = DataLoader(real_dataset_for_pred, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    with torch.no_grad():
        for images_batch, _, _ in real_loader_for_pred:
            images_batch = images_batch.to(device)
            outputs = model(images_batch) # Get logits
            fold_predictions_logits.append(outputs.cpu().numpy())

    all_predictions_logits.append(np.concatenate(fold_predictions_logits, axis=0))

    # --- Final Ensemble Evaluation on ALL Real Samples ---
    print("\n--- Final Ensemble Evaluation on ALL REAL Samples (from both collections) ---")

    # Average the logits across folds
    averaged_logits = np.mean(np.array(all_predictions_logits), axis=0) # Shape will be (N, 1)

    # Convert averaged logits to probabilities, then to binary predictions
    final_predictions_probs = torch.sigmoid(torch.from_numpy(averaged_logits)).cpu().numpy()
    final_predictions_encoded = (final_predictions_probs >= 0.5).astype(int).flatten() # Ensure 1D array of 0s and 1s

    final_true_labels_encoded = y_original_for_skf.cpu().numpy() # These are already 0s and 1s

    # Use class_names directly as they are already defined ['Collection_1st', 'Collection_2nd']
    final_true_labels_names = [class_names[int(x)] for x in final_true_labels_encoded]
    final_predictions_names = [class_names[int(x)] for x in final_predictions_encoded]

    overall_accuracy_real_pt = accuracy_score(final_true_labels_names, final_predictions_names)
    print(f"\n--- Overall Accuracy ({model_identifier} Ensemble, Evaluated on ALL REAL samples - Leaf_Collection_Binary): {overall_accuracy_real_pt:.4f} ---")

    print(f"\n--- Classification Report ({model_identifier} Ensemble, Evaluated on ALL REAL samples - Leaf_Collection_Binary) ---")
    report_dict = classification_report(final_true_labels_names, final_predictions_names, target_names=class_names, zero_division=0, output_dict=True)
    print(classification_report(final_true_labels_names, final_predictions_names, target_names=class_names, zero_division=0))

    # --- Save Classification Report to JSON ---
    metrics_output_path = metrics_save_dir / f"{model_identifier}_classification_report.json"
    with open(metrics_output_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report saved to: {metrics_output_path}")

    cm_real_pt = confusion_matrix(final_true_labels_names, final_predictions_names, labels=class_names)

    # --- Save True and Predicted Labels for Confusion Matrix Plotting ---
    np.save(confusion_matrix_data_dir / f"{model_identifier}_true_labels.npy", final_true_labels_names)
    np.save(confusion_matrix_data_dir / f"{model_identifier}_predicted_labels.npy", final_predictions_names)
    print(f"True and predicted labels for confusion matrix saved to {confusion_matrix_data_dir}.")

    # Existing Confusion Matrix Plots (for immediate visual confirmation)
    plt.figure(figsize=(8, 7)) # Adjusted size for binary classification
    sns.heatmap(cm_real_pt, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({model_identifier} Ensemble, Evaluated on ALL REAL samples - Leaf_Collection_Binary)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, f"{model_identifier}_ConfusionMatrix.png"), dpi=300)
    plt.show()

    cm_normalized_real_pt = cm_real_pt.astype('float') / cm_real_pt.sum(axis=1)[:, np.newaxis]
    cm_normalized_real_pt[np.isnan(cm_normalized_real_pt)] = 0

    plt.figure(figsize=(8, 7)) # Adjusted size for binary classification
    sns.heatmap(cm_normalized_real_pt, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Normalized Confusion Matrix ({model_identifier} Ensemble, Evaluated on ALL REAL samples - Leaf_Collection_Binary)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir, f"{model_identifier}_NormalizedConfusionMatrix.png"), dpi=300)
    plt.show()

    # Store results in the global results_storage dictionary
    results_storage['Collection_Binary_Dataset'] = {
        'class_counts': {},
        'model_metrics': {}
    }
    
    # Corrected class_counts population for binary
    for i, class_name_str in enumerate(class_names):
        count = np.sum(final_true_labels_encoded == i)
        results_storage['Collection_Binary_Dataset']['class_counts'][class_name_str] = int(count) # Store as int
    print(f"Class counts populated for 'Collection_Binary_Dataset'.")

    results_storage['Collection_Binary_Dataset']['model_metrics'][model_identifier] = {
        'precision': {cls: report_dict[cls]['precision'] for cls in class_names},
        'recall': {cls: report_dict[cls]['recall'] for cls in class_names},
        'f1-score': {cls: report_dict[cls]['f1-score'] for cls in class_names},
        'accuracy': report_dict['accuracy'],
        'macro avg precision': report_dict['macro avg']['precision'],
        'macro avg recall': report_dict['macro avg']['recall'],
        'macro avg f1-score': report_dict['macro avg']['f1-score'],
        'weighted avg precision': report_dict['weighted avg']['precision'],
        'weighted avg recall': report_dict['weighted avg']['recall'],
        'weighted avg f1-score': report_dict['weighted avg']['f1-score'],
    }
    print(f"Metrics for '{model_identifier}' stored in results_storage for 'Collection_Binary_Dataset'.")

    print("\n--- Current contents of results_storage (should include new model metrics) ---")
    print(json.dumps(results_storage, indent=4))

    # --- Grad-CAM Visualization ---
    print(f"\n--- Generating Average Grad-CAM Visualizations for {model_identifier} (Model from Fold 0) ---")

    if saved_model_paths_per_fold[0] is None or not Path(saved_model_paths_per_fold[0]).exists():
        print(f"Error: Model file for Fold 0 not found or not saved. Skipping Grad-CAM visualization for combined dataset.")
    else:
        model_to_visualize_path = saved_model_paths_per_fold[0]

        cam_model = LeafCNN(num_classes=num_classes, image_size=image_size_tuple, num_input_channels=num_channels).to(device)
        checkpoint = torch.load(model_to_visualize_path, map_location=device)
        cam_model.load_state_dict(checkpoint['model_state_dict'])
        cam_model.eval()

        target_layer = cam_model.features[-3] # Ensure this is the correct last convolutional layer

        grad_cam = GradCAM(cam_model, target_layer)
        average_class_heatmaps = {}

        real_indices_by_class = {cls_idx: [] for cls_idx in range(len(class_names))}
        for idx in real_original_indices_global: # Use real_original_indices_global for actual real samples
            class_label = y_encoded_tensor[idx].item()
            real_indices_by_class[class_label].append(idx)

        print("Calculating average Grad-CAM heatmaps per class...")
        for class_idx in range(len(class_names)): # Will iterate for 0 and 1
            class_name = class_names[class_idx]
            class_samples_indices = real_indices_by_class[class_idx]

            if not class_samples_indices:
                print(f"  No real samples for class '{class_name}'. Skipping average Grad-CAM.")
                average_class_heatmaps[class_idx] = None
                continue

            summed_heatmap = np.zeros(image_size_tuple, dtype=np.float32)
            count_for_average = 0

            # Randomly sample images for CAM visualization
            samples_for_cam = np.random.choice(class_samples_indices, min(NUM_SAMPLES_PER_CLASS_FOR_CAM_PLOT, len(class_samples_indices)), replace=False)

            for sample_idx in samples_for_cam:
                image_tensor = X_images_tensor[sample_idx]
                input_image_for_cam = image_tensor.unsqueeze(0).to(device)

                # For binary, Grad-CAM needs a target class.
                # If target_class is None in grad_cam(input_image_for_cam), it defaults to predicted.
                # Here, we specify the true class for the sample.
                heatmap = grad_cam(input_image_for_cam, target_class=class_idx)
                
                summed_heatmap += heatmap
                count_for_average += 1
                
            if count_for_average > 0:
                avg_heatmap = summed_heatmap / count_for_average
                avg_heatmap = avg_heatmap - np.min(avg_heatmap)
                if np.max(avg_heatmap) == 0:
                    avg_heatmap = np.zeros_like(avg_heatmap)
                else:
                    avg_heatmap = avg_heatmap / np.max(avg_heatmap)
                average_class_heatmaps[class_idx] = avg_heatmap
                print(f"  Calculated average for class: '{class_name}' ({count_for_average} samples)")
            else:
                average_class_heatmaps[class_idx] = None


        num_plots_total = len(class_names)
        num_cols_grid = 2 # Fixed to 2 columns for binary classification
        num_rows_grid = math.ceil(num_plots_total / num_cols_grid)

        fig_width = num_cols_grid * 3.0
        fig_height = num_rows_grid * 3.5

        sns.set_style("white")
        plt.rcParams.update({'font.size': 10})

        fig, axes = plt.subplots(num_rows_grid, num_cols_grid, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()

        print(f"\nPlotting average Grad-CAMs in a {num_rows_grid}x{num_cols_grid} grid...")

        for i in range(len(class_names)):
            ax = axes[i]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(class_names[i], fontsize=10)

            avg_heatmap = average_class_heatmaps[i]
            if avg_heatmap is not None:
                if real_indices_by_class[i]:
                    example_image_tensor = X_images_tensor[real_indices_by_class[i][0]] # Use first available real sample for background
                    cam_image_on_background = show_cam_on_black_background(avg_heatmap, example_image_tensor, image_size_tuple)
                    ax.imshow(cam_image_on_background)
                    
                    # --- Save individual Grad-CAM image (no text/axes) ---
                    individual_cam_output_path = grad_cam_output_dir / f"{model_identifier}_GradCAM_{class_names[i]}.png"
                    
                    fig_single = plt.figure(figsize=(image_size_tuple[0]/100, image_size_tuple[1]/100), dpi=100)
                    ax_single = fig_single.add_subplot(111)
                    ax_single.imshow(cam_image_on_background)
                    ax_single.set_axis_off()
                    ax_single.set_position([0,0,1,1])
                    fig_single.savefig(individual_cam_output_path, bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close(fig_single)
                    print(f"  Saved individual Grad-CAM for class '{class_names[i]}' to: {individual_cam_output_path}")

                else:
                    ax.text(0.5, 0.5, 'No Samples', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Samples', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='gray', fontsize=10)

        for j in range(num_plots_total, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f'Average Grad-CAM Visualizations ({model_identifier}, Target: Leaf_Collection_Binary)', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(base_output_dir, f"{model_identifier}_AverageGradCAM.png"), dpi=300)
        plt.show()

    print(f"\n{'='*10} Finished CNN Training for Collection (1st vs 2nd) Binary Classification {'='*10}")


if __name__ == "__main__":
    run_cnn_training_for_collection_binary(CNN_DATASET_LOAD_CONFIGS)

    print("\nAll CNN training and evaluation pipelines for Collection Binary datasets completed.")
    print("Final aggregated results_storage:")
    print(json.dumps(results_storage, indent=4))
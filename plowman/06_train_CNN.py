#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight # Added for weighted loss
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pickle
from pathlib import Path
import sys # Added for sys.exit

# PyTorch Imports
import torch
import torch.nn as tnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # Added for LR scheduling
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
MODEL_IDENTIFIER = 'ECT_Mask_2Channel_CNN_Ensemble_Improved' # Updated identifier

# --- Data Input Configuration ---
# Path to the final prepared dataset from the data generation script
FINAL_PREPARED_DATA_FILE = Path("./05_synthetic_leaf_data/final_cnn_dataset.pkl")

# --- Model Saving Setup ---
MODEL_SAVE_DIR = "trained_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Grad-CAM specific configurations
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

    X_images = final_data['X_images'] # (N, H, W, 2) numpy array
    y_labels_encoded = final_data['y_labels_encoded'] # (N,) numpy array
    is_real_flags = final_data['is_real_flags'] # (N,) boolean numpy array
    class_names = final_data['class_names'] # List of string class names
    image_size_tuple = final_data['image_size'] # (H, W) tuple, e.g., (256, 256)
    num_channels = final_data['num_channels'] # int, should be 2

    # Recreate LabelEncoder from class_names for inverse_transform functionality
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names) # Fit with the actual class names

    # Assuming classification is by 'class_label' for plotting titles etc.
    target_column_used_for_data = 'Leaf_Class'

    print(f"Loaded image data shape: {X_images.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Image size: {image_size_tuple}")
    print(f"Number of channels: {num_channels}")
    print(f"Number of real samples: {np.sum(is_real_flags)}")
    print(f"Number of synthetic samples: {np.sum(~is_real_flags)}")
    print(f"Data will be processed for classification of: '{target_column_used_for_data}'")

except FileNotFoundError:
    print(f"Error: Data file not found at {FINAL_PREPARED_DATA_FILE}.")
    print("Please ensure the data generation script (05_synthetic_leaf_data.py) has been run successfully.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    sys.exit(1)

# --- PyTorch Data Preparation ---
X_images_tensor = torch.from_numpy(X_images).float().permute(0, 3, 1, 2)
y_encoded_tensor = torch.from_numpy(y_labels_encoded).long()
is_real_flag_tensor = torch.from_numpy(is_real_flags).bool()

print(f"Tensor image data shape (after permute): {X_images_tensor.shape}")

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
        return self.images[idx], self.labels[idx], self.is_real_flags[idx]

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

print(f"\n--- Performing PyTorch CNN with {K_FOLDS}-Fold Stratified Cross-Validation (2-Channel Image Data) ---")

# Separate real samples for K-Fold splitting and validation
real_original_indices_global = torch.where(is_real_flag_tensor)[0].cpu().numpy()

X_original_images_for_skf = X_images_tensor[real_original_indices_global]
y_original_for_skf = y_encoded_tensor[real_original_indices_global]

skf_pytorch = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

all_predictions_logits = []

saved_model_paths_per_fold = [None] * K_FOLDS

# --- Calculate class weights for imbalanced dataset ---
# Use the labels from the entire real+synthetic dataset for calculating class weights
# as the training set will include both.
all_training_labels_for_weights = y_encoded_tensor.cpu().numpy()
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(all_training_labels_for_weights),
    y=all_training_labels_for_weights
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"\nCalculated class weights: {class_weights_tensor.cpu().numpy()}")


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

    # Initialize model, criterion (with weights), optimizer, and scheduler for the current fold
    model = LeafCNN(num_classes=len(class_names), image_size=image_size_tuple, num_input_channels=num_channels).to(device)
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
        for images, labels, _ in train_loader:
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
        current_overall_accuracy_on_real_samples = 0.0

        with torch.no_grad():
            for images, labels, _ in val_loader:
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

            current_overall_accuracy_on_real_samples = val_accuracy

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == NUM_EPOCHS -1:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f} (Real Samples)")

        # Step the LR scheduler
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            if current_overall_accuracy_on_real_samples > best_overall_accuracy_for_saving_this_fold:
                best_overall_accuracy_for_saving_this_fold = current_overall_accuracy_on_real_samples
                save_model_checkpoint(model, optimizer, epoch, best_overall_accuracy_for_saving_this_fold, MODEL_IDENTIFIER, target_column_used_for_data, fold_idx)
                saved_model_paths_per_fold[fold_idx] = os.path.join(MODEL_SAVE_DIR, f"{MODEL_IDENTIFIER}_fold{fold_idx}_best_{target_column_used_for_data}.pth")

        else:
            epochs_no_improve += 1
            if epochs_no_improve == PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

    model.load_state_dict(best_model_wts)
    print(f"Fold {fold_idx + 1} training complete. Best validation loss for fold: {best_val_loss:.4f}")

    # Predict logits for ALL real samples using the best model of this fold
    model.eval()
    fold_predictions_logits = []

    real_dataset_for_pred = LeafDataset(X_original_images_for_skf, y_original_for_skf, torch.ones_like(y_original_for_skf, dtype=torch.bool))
    real_loader_for_pred = DataLoader(real_dataset_for_pred, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    with torch.no_grad():
        for images_batch, _, _ in real_loader_for_pred:
            images_batch = images_batch.to(device)
            outputs = model(images_batch)
            fold_predictions_logits.append(outputs.cpu().numpy())

    all_predictions_logits.append(np.concatenate(fold_predictions_logits, axis=0))

#########################################################
### FINAL ENSEMBLE EVALUATION ON REAL SAMPLES ONLY ###
#########################################################

print("\n--- Final Ensemble Evaluation on ALL REAL Samples ---")

averaged_logits = np.mean(np.array(all_predictions_logits), axis=0)
final_predictions_encoded = np.argmax(averaged_logits, axis=1)

final_true_labels_encoded = y_original_for_skf.cpu().numpy()

final_true_labels_names = label_encoder.inverse_transform(final_true_labels_encoded)
final_predictions_names = label_encoder.inverse_transform(final_predictions_encoded)

overall_accuracy_real_pt = accuracy_score(final_true_labels_names, final_predictions_names)
print(f"\n--- Overall Accuracy ({MODEL_IDENTIFIER} Ensemble, Evaluated on REAL samples ONLY - {target_column_used_for_data}): {overall_accuracy_real_pt:.4f} ---")

print(f"\n--- Classification Report ({MODEL_IDENTIFIER} Ensemble, Evaluated on REAL samples ONLY - {target_column_used_for_data}) ---")
report_dict = classification_report(final_true_labels_names, final_predictions_names, target_names=class_names, zero_division=0, output_dict=True)
print(classification_report(final_true_labels_names, final_predictions_names, target_names=class_names, zero_division=0))

cm_real_pt = confusion_matrix(final_true_labels_names, final_predictions_names, labels=class_names)

plt.figure(figsize=(16, 14))
sns.heatmap(cm_real_pt, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix ({MODEL_IDENTIFIER} Ensemble, Evaluated on REAL samples ONLY - {target_column_used_for_data})')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, f"{MODEL_IDENTIFIER}_ConfusionMatrix_{target_column_used_for_data}.png"), dpi=300)
plt.show()

cm_normalized_real_pt = cm_real_pt.astype('float') / cm_real_pt.sum(axis=1)[:, np.newaxis]
cm_normalized_real_pt[np.isnan(cm_normalized_real_pt)] = 0

plt.figure(figsize=(16, 14))
sns.heatmap(cm_normalized_real_pt, annot=True, fmt='.2f', cmap='Blues', cbar=True,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Normalized Confusion Matrix ({MODEL_IDENTIFIER} Ensemble, Evaluated on REAL samples ONLY - {target_column_used_for_data})')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, f"{MODEL_IDENTIFIER}_NormalizedConfusionMatrix_{target_column_used_for_data}.png"), dpi=300)
plt.show()

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
print(f"Metrics for '{MODEL_NAME}' stored in results_storage for '{target_column_used_for_data}'.")

print("\n--- Current contents of results_storage (should include new model metrics) ---")
print(results_storage)


###################################
### GRAD-CAM VISUALIZATION ###
###################################

print(f"\n--- Generating Average Grad-CAM Visualizations for {MODEL_IDENTIFIER} (Model from Fold 0) ---")

if len(saved_model_paths_per_fold) <= 0 or saved_model_paths_per_fold[0] is None or not os.path.exists(saved_model_paths_per_fold[0]):
    print(f"Error: Model file for Fold 0 not found or not saved at {saved_model_paths_per_fold[0] if len(saved_model_paths_per_fold) > 0 else 'N/A'}. Skipping Grad-CAM visualization.")
else:
    model_to_visualize_path = saved_model_paths_per_fold[0]

    cam_model = LeafCNN(num_classes=len(class_names), image_size=image_size_tuple, num_input_channels=num_channels).to(device)
    checkpoint = torch.load(model_to_visualize_path, map_location=device)
    cam_model.load_state_dict(checkpoint['model_state_dict'])
    cam_model.eval()

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
            self.model.zero_grad()
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()

            one_hot = torch.zeros_like(output).to(device)
            one_hot[0][target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)

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
        ect_channel = original_image_tensor[1, :, :].cpu().numpy() # ECT is channel 1

        img_display_base = np.stack([ect_channel, ect_channel, ect_channel], axis=-1)
        img_display_base = np.uint8(255 * img_display_base)

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_heatmap), cv2.COLORMAP_JET)
        heatmap_colored = np.float32(heatmap_colored) / 255

        alpha = 0.5
        final_cam_img = np.uint8(255 * (heatmap_colored * alpha + np.float32(img_display_base) / 255 * (1-alpha)))

        return final_cam_img


    grad_cam = GradCAM(cam_model, target_layer)
    average_class_heatmaps = {}

    real_indices_by_class = {cls_idx: [] for cls_idx in range(len(class_names))}
    for idx in real_original_indices_global:
        class_label = y_encoded_tensor[idx].item()
        real_indices_by_class[class_label].append(idx)

    print("Calculating average Grad-CAM heatmaps per class...")
    for class_idx in range(len(class_names)):
        class_name = class_names[class_idx]
        class_samples_indices = real_indices_by_class[class_idx]

        if not class_samples_indices:
            print(f"  No real samples for class '{class_name}'. Skipping average Grad-CAM.")
            average_class_heatmaps[class_idx] = None
            continue

        summed_heatmap = np.zeros(image_size_tuple, dtype=np.float32)
        count_for_average = 0

        samples_for_cam = np.random.choice(class_samples_indices, min(NUM_SAMPLES_PER_CLASS_FOR_CAM_PLOT, len(class_samples_indices)), replace=False)

        for sample_idx in samples_for_cam:
            image_tensor = X_images_tensor[sample_idx]
            input_image_for_cam = image_tensor.unsqueeze(0).to(device)

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
    num_cols_grid = math.ceil(math.sqrt(num_plots_total))
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
                example_image_tensor = X_images_tensor[real_indices_by_class[i][0]]
                cam_image_on_background = show_cam_on_black_background(avg_heatmap, example_image_tensor, image_size_tuple)
                ax.imshow(cam_image_on_background)
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
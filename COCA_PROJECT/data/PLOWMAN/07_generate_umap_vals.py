import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import umap # For UMAP dimensionality reduction

# --- Configuration (Copied/Adapted from your 06_train_CNN.py) ---
# Path to the final prepared dataset from the data generation script
FINAL_PREPARED_DATA_FILE = Path("./05_synthetic_leaf_data/final_cnn_dataset.pkl")

# Model Saving Setup - where the trained model is located and where UMAP output will go
MODEL_SAVE_DIR = "trained_models"
MODEL_IDENTIFIER = 'ECT_Mask_2Channel_CNN_Ensemble_Improved'
TARGET_COLUMN_USED_FOR_DATA = 'Leaf_Class' # From your script
FOLD_TO_LOAD = 0 # As discussed, we'll use the Fold 0 model

# --- Device Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# --- PYTORCH DATASET & MODEL (Copied from your 06_train_CNN.py) ---
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
            tnn.MaxPool2d(kernel_size=2, stride=2), # This is the last layer in features, output will be passed to Flatten

            tnn.Conv2d(64, 128, kernel_size=3, padding=1),
            tnn.BatchNorm2d(128),
            tnn.ReLU(),
            tnn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            temp_features_model = self.features.to(device)
            dummy_input = torch.zeros(1, num_input_channels, image_size[0], image_size[1]).to(device)
            # Pass through features to calculate flattened_size
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

# --- 1. Load Data ---
print("\n--- Loading data from FINAL_PREPARED_DATA_FILE ---")
try:
    with open(FINAL_PREPARED_DATA_FILE, 'rb') as f:
        final_data = pickle.load(f)

    X_images = final_data['X_images'] # (N, H, W, 2) numpy array
    y_labels_encoded = final_data['y_labels_encoded'] # (N,) numpy array
    is_real_flags = final_data['is_real_flags'] # (N,) boolean numpy array
    class_names = final_data['class_names'] # List of string class names
    image_size_tuple = final_data['image_size'] # (H, W) tuple
    num_channels = final_data['num_channels'] # int

    # Recreate LabelEncoder from class_names for inverse_transform functionality
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    # Filter for REAL samples only for UMAP
    real_indices = np.where(is_real_flags)[0]
    X_real_images = torch.from_numpy(X_images[real_indices]).float().permute(0, 3, 1, 2)
    y_real_labels_encoded = torch.from_numpy(y_labels_encoded[real_indices]).long()
    y_real_labels_names = label_encoder.inverse_transform(y_real_labels_encoded.cpu().numpy())

    print(f"Loaded real image data shape for UMAP: {X_real_images.shape}")
    print(f"Number of real samples for UMAP: {X_real_images.shape[0]}")

except FileNotFoundError:
    print(f"Error: Data file not found at {FINAL_PREPARED_DATA_FILE}.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    sys.exit(1)

# --- 2. Load the Trained Model (Fold 0) ---
model_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_IDENTIFIER}_fold{FOLD_TO_LOAD}_best_{TARGET_COLUMN_USED_FOR_DATA}.pth")

if not os.path.exists(model_path):
    print(f"Error: Model for Fold {FOLD_TO_LOAD} not found at {model_path}.")
    print("Please ensure you have run your 06_train_CNN.py script and the model was saved.")
    sys.exit(1)

print(f"\n--- Loading trained model from: {model_path} ---")
model = LeafCNN(num_classes=len(class_names), image_size=image_size_tuple, num_input_channels=num_channels).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Set model to evaluation mode

# --- 3. Feature Extraction ---
print("\n--- Extracting features from real images ---")
features_list = []
real_data_loader = DataLoader(LeafDataset(X_real_images, y_real_labels_encoded, torch.ones_like(y_real_labels_encoded, dtype=torch.bool)),
                              batch_size=32, shuffle=False, num_workers=0)

with torch.no_grad():
    for images, _, _ in real_data_loader:
        images = images.to(device)
        # Pass images through the feature extractor part of the model
        # The output of model.features(images) is the high-dimensional embedding
        features = model.features(images)
        features_list.append(features.cpu().numpy())

# Concatenate all features and flatten them
# Features will be (Batch_size, Channels, Height, Width) after model.features
# Need to flatten to (Batch_size, Channels * Height * Width)
all_features = np.concatenate(features_list, axis=0)
all_features_flattened = all_features.reshape(all_features.shape[0], -1)

print(f"Extracted features shape: {all_features_flattened.shape}")

# --- 4. UMAP Dimensionality Reduction ---
print("\n--- Applying UMAP dimensionality reduction ---")
reducer = umap.UMAP(n_components=2, random_state=42) # Using random_state for reproducibility
umap_embeddings = reducer.fit_transform(all_features_flattened)

print(f"UMAP embeddings shape: {umap_embeddings.shape}")

# --- 5. Save UMAP Results ---
umap_df = pd.DataFrame({
    'umap_x': umap_embeddings[:, 0],
    'umap_y': umap_embeddings[:, 1],
    'class_label': y_real_labels_names # Use the string class names
})

output_umap_filename = f"{MODEL_IDENTIFIER}_fold{FOLD_TO_LOAD}_umap_embeddings_real_data.csv"
output_path = os.path.join(MODEL_SAVE_DIR, output_umap_filename)
umap_df.to_csv(output_path, index=False)

print(f"\nUMAP embeddings saved to: {output_path}")
print("\n--- UMAP coordinate generation complete for PLOWMAN ---")
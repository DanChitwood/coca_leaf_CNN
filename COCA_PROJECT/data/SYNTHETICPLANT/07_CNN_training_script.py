import os
import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

# Import necessary functions and constants from the synthetic data generator library
from _04_synthetic_data_generator_library import (
    load_pca_model_data,
    SAMPLES_PER_CLASS_TARGET,
    DATASET_CONFIGS,
    inverse_transform_pca,
    calculate_ect_min_max_for_dataset,
    generate_n_ranked_leaves_for_cnn, # This is the key function to import
    ECT, # ECT class
    IMAGE_SIZE, # Global individual leaf image size from script 04
    PADDING_BETWEEN_LEAVES, # Global padding from script 04
    COMBINED_IMAGE_HEIGHT, # Global combined image height from script 04
    APPLY_RANDOM_ROTATION # From script 04
)
from imblearn.over_sampling import SMOTE # Import SMOTE here

# --- Configuration Constants ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR_BASE = os.path.join(BASE_DIR, 'trained_models')
PCA_DATA_PATH = os.path.join(BASE_DIR, '03_morphometrics_output_combined')
# ORIGINAL_IMAGE_DIR_BASE and ECT_MAPS_DIR_BASE are no longer used for direct file loading in Dataset
# but can be kept for conceptual path understanding if needed elsewhere.
ORIGINAL_IMAGE_DIR_BASE = os.path.join(BASE_DIR, '01_raw_images_selected_combined')
ECT_MAPS_DIR_BASE = os.path.join(BASE_DIR, '02_ect_output_combined')


# Global ECT Min/Max to be calculated once and passed to datasets
GLOBAL_ECT_MIN = None
GLOBAL_ECT_MAX = None

# Global PCA data and ECT calculator instance
GLOBAL_PCA_DATA = None
GLOBAL_ECT_CALCULATOR = None


# --- Custom Combined Leaf Dataset Class (Refactored) ---
class CombinedLeafDataset(Dataset):
    def __init__(self, sample_recipes, n_leaves, pca_data, ect_calculator,
                 global_ect_min, global_ect_max, image_size, transform=None, verbose=False):
        """
        Args:
            sample_recipes (list of dict): Each dict contains 'class_label_str' (decoded),
                                            'encoded_label', 'is_synthetic', and 'original_idx' (for real).
            n_leaves (int): Number of leaves to generate/combine for each sample.
            pca_data (dict): PCA model components, mean, scores, labels, flattened coords.
            ect_calculator (ECT): Initialized ECT calculator instance.
            global_ect_min (float): Global minimum ECT value for normalization.
            global_ect_max (float): Global maximum ECT value for normalization.
            image_size (tuple): Desired (width, height) for individual leaf components.
            transform (torchvision.transforms.Compose): Transformations for the RGB image.
            verbose (bool): If True, print detailed generation messages.
        """
        self.sample_recipes = sample_recipes
        self.n_leaves = n_leaves
        self.pca_data = pca_data
        self.ect_calculator = ect_calculator
        self.global_ect_min = global_ect_min
        self.global_ect_max = global_ect_max
        self.image_size = image_size
        self.transform = transform
        self.verbose = verbose

    def __len__(self):
        return len(self.sample_recipes)

    def __getitem__(self, idx):
        recipe = self.sample_recipes[idx]
        class_label_str = recipe['class_label_str']
        encoded_label = recipe['encoded_label']
        is_synthetic = recipe['is_synthetic']
        
        pre_selected_real_indices = None
        # If it's a real sample and original_idx is present, use it for pre-selection
        if not is_synthetic and 'original_indices' in recipe and recipe['original_indices'] is not None:
             pre_selected_real_indices = recipe['original_indices']
        
        # Call the generation function from script 04
        # For synthetic samples, real_sample_pct will effectively be 0 unless there are
        # some real samples also specified via pre_selected_real_indices
        combined_image_array, _ = generate_n_ranked_leaves_for_cnn(
            n=self.n_leaves,
            class_label=class_label_str,
            pca_data=self.pca_data,
            ect_calculator=self.ect_calculator,
            ect_min_val=self.global_ect_min,
            ect_max_val=self.global_ect_max,
            real_sample_pct=1.0 if not is_synthetic else 0.0, # Use 100% real if not synthetic, else 0%
            image_size=self.image_size,
            padding=PADDING_BETWEEN_LEAVES, # Use global constant
            verbose=self.verbose,
            pre_selected_real_indices=pre_selected_real_indices # Pass pre-selected indices
        )

        # Separate channels: combined_image_array is (H, W_combined, 2)
        # Channel 0 is mask, Channel 1 is ECT
        # The CNN expects RGB image and a grayscale ECT map, but both are derived from combined_image_array.
        # Here, we'll make the "image" input a 3-channel version of the mask
        # and the "ect_map" input the single ECT channel.
        
        # Ensure the combined_image_array is not empty (e.g., if n=0 or generation failed)
        if combined_image_array.shape[1] == 0: # Check combined width
            # Return dummy tensors if generation failed
            combined_width = self.n_leaves * self.image_size[0] + (self.n_leaves - 1) * PADDING_BETWEEN_LEAVES
            dummy_rgb = torch.zeros((3, COMBINED_IMAGE_HEIGHT, combined_width), dtype=torch.float32)
            dummy_ect = torch.zeros((1, COMBINED_IMAGE_HEIGHT, combined_width), dtype=torch.float32)
            if self.verbose:
                print(f"  Warning: Generation of combined image failed for class {class_label_str}, returning dummy tensors.")
            return dummy_rgb, dummy_ect, torch.tensor(encoded_label, dtype=torch.long)


        combined_mask = combined_image_array[:, :, 0]
        combined_ect = combined_image_array[:, :, 1]

        # Convert combined mask to a 3-channel "RGB" image (e.g., repeating the mask across channels)
        # The CNN expects 3 channels for its image branch.
        combined_rgb_image = np.stack([combined_mask, combined_mask, combined_mask], axis=-1)
        
        # Convert to PIL Image for torchvision transforms
        combined_rgb_image_pil = Image.fromarray((combined_rgb_image * 255).astype(np.uint8))
        
        if self.transform:
            combined_rgb_image_tensor = self.transform(combined_rgb_image_pil)
        else:
            combined_rgb_image_tensor = transforms.ToTensor()(combined_rgb_image_pil)
            combined_rgb_image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(combined_rgb_image_tensor)

        # Convert ECT map to a 1-channel tensor
        # It's already normalized [0,1] by generate_n_ranked_leaves_for_cnn
        combined_ect_tensor = torch.from_numpy(combined_ect).unsqueeze(0).float() # Add channel dimension

        return combined_rgb_image_tensor, combined_ect_tensor, torch.tensor(encoded_label, dtype=torch.long)

# --- CNN Model Definition (No change here, as it expects 3-channel image, 1-channel ECT) ---
class LeafCNN(nn.Module):
    def __init__(self, num_classes):
        super(LeafCNN, self).__init__()
        # Image Branch (RGB)
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # ECT Branch (Grayscale)
        self.ect_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate input features for the fully connected layer
        # The input feature map size now depends on the COMBINED_IMAGE_HEIGHT and calculated combined_width.
        # However, the CNN layers are defined to downsample based on individual leaf IMAGE_SIZE (256x256).
        # We need to ensure the final feature map size is consistent regardless of N.
        # This implies that each *branch* (image_conv, ect_conv) should process its own input,
        # and then the outputs are concatenated.
        # Given the design of `generate_n_ranked_leaves_for_cnn` creating a *single wide image*,
        # the CNN's first layers need to handle this wide image.
        # So, the final width for feature map will be `(combined_width // 8)`
        # And the final height for feature map will be `(COMBINED_IMAGE_HEIGHT // 8)`
        
        # Calculate these dynamically based on expected input dimensions
        # Assuming input to CNN is (BATCH_SIZE, CHANNELS, H_combined, W_combined)
        # Initial dimensions for a combined N-leaf image:
        # H = COMBINED_IMAGE_HEIGHT (e.g., 256)
        # W_combined = n * IMAGE_SIZE[0] + (n-1)*PADDING_BETWEEN_LEAVES

        # After 3 MaxPool2d (stride=2), the dimensions are divided by 2^3 = 8
        mock_input_h = COMBINED_IMAGE_HEIGHT # From script 04
        mock_input_w = IMAGE_SIZE[0] * 1 # Only if N=1; for N>1 this calculation is complex
        # A safer way to calculate fc_input_features is to pass a dummy tensor
        # through the conv layers:
        
        # Dummy forward pass to calculate the size of flattened features
        dummy_image = torch.zeros(1, 3, mock_input_h, mock_input_w) # Shape for N=1
        dummy_ect = torch.zeros(1, 1, mock_input_h, mock_input_w) # Shape for N=1

        dummy_img_features = self.image_conv(dummy_image)
        dummy_ect_features = self.ect_conv(dummy_ect)

        self.fc_input_features = dummy_img_features.view(dummy_img_features.size(0), -1).size(1) + \
                                 dummy_ect_features.view(dummy_ect_features.size(0), -1).size(1)
        
        # The above calculation is for N=1. If N can vary, the input size to FC must vary or be max.
        # If N_VALUES includes N > 1, the architecture *must* accommodate the wider input.
        # The simplest way is to ensure `generate_n_ranked_leaves_for_cnn` produces a fixed width,
        # or use adaptive pooling before the FC layer.
        # Since `generate_n_ranked_leaves_for_cnn` produces a variable width, adaptive pooling is needed.

        self.image_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # Pool to 1x1 regardless of input size
        self.ect_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Recalculate fc_input_features after adaptive pooling
        # 128 channels * 1 * 1 for each branch * 2 branches = 256
        self.fc_input_features = 128 * 1 * 1 * 2


        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, image, ect_map):
        img_features = self.image_conv(image)
        ect_features = self.ect_conv(ect_map)

        # Apply adaptive pooling
        img_features = self.image_adaptive_pool(img_features)
        ect_features = self.ect_adaptive_pool(ect_features)

        # Flatten features
        img_features = img_features.view(img_features.size(0), -1)
        ect_features = ect_features.view(ect_features.size(0), -1)

        # Concatenate features from both branches
        combined_features = torch.cat((img_features, ect_features), dim=1)

        output = self.fc(combined_features)
        return output

# --- Training and Evaluation Functions (No changes needed) ---
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for images, ect_maps, labels in train_loader:
        images, ect_maps, labels = images.to(device), ect_maps.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, ect_maps)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for images, ect_maps, labels in val_loader:
            images, ect_maps, labels = images.to(device), ect_maps.to(device), labels.to(device)

            outputs = model(images, ect_maps)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    return epoch_loss, epoch_accuracy

# --- Helper Functions for Data Handling ---
def setup_output_directories(dataset_name, n_leaves):
    base_output_dir = os.path.join(OUTPUT_DIR_BASE, f'trained_models_{dataset_name}_n{n_leaves}')
    models_dir = os.path.join(base_output_dir, 'models')
    plots_dir = os.path.join(base_output_dir, 'plots')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"📁 Created output directories for {dataset_name} (N={n_leaves}) under {base_output_dir}")
    return base_output_dir, models_dir, plots_dir


def create_synthetic_samples_per_class(original_pca_scores, original_class_labels, class_label_encoder, min_samples_per_class_target):
    """
    Generates synthetic samples using SMOTE for classes below a target threshold.
    Returns a list of sample "recipes" (class_label_str, encoded_label, is_synthetic, original_indices).
    original_indices will be a list of indices from original_class_flattened_coords for real samples.
    """
    print(f"\n✨ Generating synthetic samples to achieve at least {min_samples_per_class_target} samples per class...")

    encoded_labels = class_label_encoder.transform(original_class_labels)
    class_counts = Counter(encoded_labels)
    print(f"Initial class distribution: {class_counts}")

    all_sample_recipes = []

    # First, add all original samples as recipes
    # Group original_pca_scores by their encoded labels and store their original indices
    original_samples_by_class = {label: [] for label in np.unique(encoded_labels)}
    for i, (score, label) in enumerate(zip(original_pca_scores, encoded_labels)):
        original_samples_by_class[label].append(i) # Store index in original_flattened_coords

    for encoded_label, count in class_counts.items():
        class_label_str = class_label_encoder.inverse_transform([encoded_label])[0]
        original_indices_for_class = original_samples_by_class[encoded_label]

        # Add all original samples as individual recipes for now.
        # Later, `generate_combined_samples_for_training` will select `n_leaves` from these.
        for _ in range(count): # Add placeholder for each real sample. We'll sample from these.
            all_sample_recipes.append({
                'class_label_str': class_label_str,
                'encoded_label': encoded_label,
                'is_synthetic': False,
                # Store the original indices so generate_n_ranked_leaves_for_cnn can fetch
                # the exact original shape if needed, for faithful real data representation.
                # For now, we'll store a list of available indices, and `generate_combined_samples_for_training`
                # will need to pick `n_leaves` from these, or generate `n` new synthetic.
                # This needs refinement to handle k-fold correctly for *real* samples.
                # Let's adjust this: original_indices should point to the *specific* original samples being used.
                # For `create_synthetic_samples_per_class`, we're just determining counts.
                # The actual selection happens in `generate_combined_samples_for_training`.
                # For now, this function just generates a list of 'target' counts for each class.
                # The 'recipes' will be built later.
            })


    sampling_strategy = {}
    for cls_idx, count in class_counts.items():
        if count < min_samples_per_class_target:
            sampling_strategy[cls_idx] = min_samples_per_class_target
    print(f"Classes to oversample and target counts: {sampling_strategy}")

    if not sampling_strategy:
        print("No classes need oversampling.")
        # Return only the real sample recipes if no oversampling is needed.
        # This function should probably return the "desired distribution" to `generate_combined_samples_for_training`
        # not the actual recipes. Let's simplify this.
        return original_pca_scores, encoded_labels, {} # Return original data and empty sampling strategy

    min_samples_in_oversampled_classes = float('inf')
    if sampling_strategy:
        min_samples_in_oversampled_classes = min(count for cls_idx, count in class_counts.items() if cls_idx in sampling_strategy)

    n_neighbors_for_smote = max(1, min_samples_in_oversampled_classes - 1)
    if n_neighbors_for_smote == 0 and min_samples_in_oversampled_classes > 0:
        n_neighbors_for_smote = 1

    print(f"Adjusting SMOTE n_neighbors to: {n_neighbors_for_smote}")

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=n_neighbors_for_smote)
    synthetic_pca_scores_resampled, synthetic_encoded_labels_resampled = smote.fit_resample(original_pca_scores, encoded_labels)

    # Filter out original samples, keep only truly synthetic ones generated by SMOTE
    num_original_samples = len(original_pca_scores)
    synthetic_only_pca_scores = synthetic_pca_scores_resampled[num_original_samples:]
    synthetic_only_encoded_labels = synthetic_encoded_labels_resampled[num_original_samples:]

    print(f"Generated {len(synthetic_only_pca_scores)} truly synthetic samples.")
    
    # This function now returns the original PCA data + the newly generated synthetic PCA data.
    # The 'recipes' will be constructed based on these.
    return synthetic_pca_scores_resampled, synthetic_encoded_labels_resampled, sampling_strategy


def generate_combined_samples_for_training(
    df_with_pca_and_labels, n_leaves, class_label_encoder,
    original_image_dir_base, ect_maps_dir_base, # These are largely conceptual now
    pca_data_for_all_experiments # Pass global PCA data to get flattened coords for real samples
    ):
    """
    Generates a list of "sample recipes" for the CombinedLeafDataset.
    Each recipe specifies how to generate one combined N-leaf sample (e.g., its class,
    and whether it's a real or synthetic base sample).
    
    Args:
        df_with_pca_and_labels (pd.DataFrame): DataFrame containing 'original_pca_score_idx' (index into GLOBAL_PCA_DATA['original_pca_scores']),
                                              'encoded_label', and 'is_synthetic'
        n_leaves (int): Number of leaves to combine per sample.
        class_label_encoder (LabelEncoder): Fitted LabelEncoder.
        pca_data_for_all_experiments (dict): The loaded global PCA data.

    Returns:
        list[dict]: A list of sample recipes. Each dict contains:
                    'class_label_str' (decoded plantID string),
                    'encoded_label' (integer label),
                    'is_synthetic' (boolean),
                    'original_indices' (list of int or None): If not synthetic, indices into
                                        pca_data_for_all_experiments['original_flattened_coords']
                                        for the *N* leaves to be used.
    """
    all_sample_recipes = []
    
    # Group samples by encoded label and synthetic status
    # This df is from the output of SMOTE, so it contains original and synthetic "slots"
    
    # First, separate real and synthetic samples based on their 'is_synthetic' flag
    real_samples_df = df_with_pca_and_labels[~df_with_pca_and_labels['is_synthetic']].copy()
    synthetic_samples_df = df_with_pca_and_labels[df_with_pca_and_labels['is_synthetic']].copy()

    # Create mappings for real samples: encoded_label -> list of original_flattened_coords indices
    # We need to map the original_pca_score_idx back to the original_flattened_coords index.
    # The 'original_pca_score_idx' in the real_samples_df corresponds to the index
    # within `pca_data_for_all_experiments['original_pca_scores']` and
    # `pca_data_for_all_experiments['original_flattened_coords']`.
    real_original_indices_by_class = {label: [] for label in class_label_encoder.transform(
                                                pca_data_for_all_experiments['original_class_labels']).tolist()}
    for i, original_label_encoded in enumerate(class_label_encoder.transform(pca_data_for_all_experiments['original_class_labels'])):
        real_original_indices_by_class[original_label_encoded].append(i)

    # Calculate desired number of samples per class for the final training dataset
    # We want roughly the same number of combined samples for each class.
    # The SMOTE step already ensures a balanced *base* dataset.
    # Now we decide how many *combined* N-leaf samples to generate from this balanced base.
    
    # Let's target SAMPLES_PER_CLASS_TARGET * some_factor (e.g., 2-3x to have enough variation)
    # as the number of combined samples for each class.
    target_combined_samples_per_class = SAMPLES_PER_CLASS_TARGET * 3 # Example factor

    unique_encoded_labels = df_with_pca_and_labels['encoded_label'].unique()

    for encoded_label in unique_encoded_labels:
        class_label_str = class_label_encoder.inverse_transform([encoded_label])[0]
        
        # Get all real original indices available for this class
        available_real_indices = real_original_indices_by_class.get(encoded_label, [])
        num_available_real = len(available_real_indices)

        # Get the count of synthetic samples for this class from the SMOTE output
        num_synthetic_for_class = synthetic_samples_df[synthetic_samples_df['encoded_label'] == encoded_label].shape[0]

        # Total "base" samples (real + synthetic) for this class
        total_base_samples_for_class = num_available_real + num_synthetic_for_class
        
        if total_base_samples_for_class < n_leaves:
            print(f"Warning: Class '{class_label_str}' has only {total_base_samples_for_class} base samples, which is less than n_leaves={n_leaves}. Skipping this class for combined sample generation.")
            continue # Cannot form N-leaf samples if not enough base samples

        num_generated_for_class = 0
        # Iterate until we have enough combined samples for this class
        while num_generated_for_class < target_combined_samples_per_class:
            # Decide whether to use real or synthetic leaves for this combined sample
            # A simple heuristic: if real samples are available and not exhausted, use them.
            # Otherwise, use synthetic.
            
            # This logic needs to be careful for k-fold splits.
            # During k-fold, the `df_with_pca_and_labels` that comes in here will already
            # be split into train/val. So, `available_real_indices` here should already be
            # limited to *that specific fold's real samples*.

            use_real = False
            if num_available_real >= n_leaves: # Can we draw N real samples?
                 use_real = True # Prioritize real samples if possible
            elif num_synthetic_for_class < n_leaves and num_available_real > 0: # Mix if not enough synthetic
                 # If we don't have n_leaves synthetic, but have some real, try to mix
                 # This gets complicated quickly. The simplest is to ensure the base data is balanced,
                 # and then randomly draw from the combined pool of real and synthetic indices
                 # for the n_leaves, as `generate_n_ranked_leaves_for_cnn` does internally.
                 # So, `is_synthetic` here will primarily guide the *source* of the samples for `generate_n_ranked_leaves_for_cnn`
                 # (whether it picks 100% real or 100% synthetic from the global pool).
                 # We need to create recipes that explicitly say "make a sample of N real leaves" or "make a sample of N synthetic leaves".
                 pass # Let's assume we either generate pure real or pure synthetic combined samples for now.
            
            # Simplified approach: Create recipes for a target number of combined real and synthetic samples per class.
            # Real samples for a combined N-leaf recipe:
            # Randomly select `n_leaves` indices from `available_real_indices`.
            if use_real and num_available_real >= n_leaves:
                selected_indices_for_n_leaves = random.sample(available_real_indices, n_leaves)
                all_sample_recipes.append({
                    'class_label_str': class_label_str,
                    'encoded_label': encoded_label,
                    'is_synthetic': False, # This combined sample is composed of real leaves
                    'original_indices': selected_indices_for_n_leaves # Store the exact indices
                })
                num_generated_for_class += 1
            elif num_synthetic_for_class >= n_leaves: # Can we draw N synthetic samples?
                # For synthetic samples, we don't need to pass specific indices.
                # `generate_n_ranked_leaves_for_cnn` will generate them on the fly.
                all_sample_recipes.append({
                    'class_label_str': class_label_str,
                    'encoded_label': encoded_label,
                    'is_synthetic': True, # This combined sample is composed of synthetic leaves
                    'original_indices': None # Not applicable for on-the-fly synthetic generation
                })
                num_generated_for_class += 1
            else:
                 # Not enough real or synthetic samples to create an N-leaf combined sample
                 # if we insist on pure real or pure synthetic batches.
                 # If mixing is allowed for a single N-leaf sample, the logic for `is_synthetic`
                 # and `pre_selected_real_indices` needs to be more complex.
                 # For simplicity now, if we can't get pure real/synthetic N-leaf, we stop for this class.
                break # Exit while loop if cannot generate more
    
    print(f"Generated {len(all_sample_recipes)} combined sample recipes.")
    return all_sample_recipes


# --- Main Experiment Function ---
def run_experiment_for_n(n_leaves, config, device,
                         pca_data_for_all_experiments, ect_calculator_instance,
                         global_ect_min, global_ect_max):
    """
    Runs the CNN training experiment for a given number of leaves (N).
    """
    dataset_name = config['dataset_name']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    k_folds = config['k_folds']
    image_size = config['image_size'] # This is IMAGE_SIZE from script 04
    
    print(f"\n=============== Starting CNN Training for {dataset_name} Dataset with N={n_leaves} ===============")

    # Setup output directories
    base_output_dir, models_dir, plots_dir = setup_output_directories(dataset_name, n_leaves)

    # Load original PCA data and class labels for all samples
    original_pca_scores_all = pca_data_for_all_experiments['original_pca_scores']
    original_class_labels_all = pca_data_for_all_experiments['original_class_labels']

    # Encode class labels
    class_label_encoder = LabelEncoder()
    encoded_labels_all = class_label_encoder.fit_transform(original_class_labels_all)
    num_classes = len(class_label_encoder.classes_)
    print(f"📊 Total unique plantIDs (classes) for this run: {num_classes}")

    # The combined image size will be handled dynamically by the dataset and generate_n_ranked_leaves_for_cnn
    # The CNN model now uses AdaptiveAvgPool2d to handle variable input widths.
    print(f"🖼️ Individual leaf image size (base): {image_size}")
    print(f"📐 Combined image height: {COMBINED_IMAGE_HEIGHT}")
    # Combined image width will be n * IMAGE_SIZE[0] + (n-1)*PADDING_BETWEEN_LEAVES

    # Use the imported SAMPLES_PER_CLASS_TARGET directly
    min_samples_for_smote_target = SAMPLES_PER_CLASS_TARGET

    # Generate synthetic samples and get the resampled PCA data and labels
    # This step produces the 'pool' of samples (original + synthetic) that we will draw from
    # to create our N-leaf combined training samples.
    resampled_pca_scores, resampled_encoded_labels, _ = create_synthetic_samples_per_class(
        original_pca_scores_all,
        original_class_labels_all,
        class_label_encoder,
        min_samples_for_smote_target
    )

    # Create a DataFrame that links encoded labels, PCA scores, and indicates if synthetic.
    # This DataFrame represents the "pool" of *individual leaf data points* (real or synthetic)
    # from which N-leaf samples will be drawn.
    combined_pool_df = pd.DataFrame({
        'pca_scores': list(resampled_pca_scores), # Store actual PCA scores if needed
        'encoded_label': resampled_encoded_labels,
        'is_synthetic': [False]*len(original_pca_scores_all) + [True]*(len(resampled_pca_scores) - len(original_pca_scores_all))
    })

    # Prepare labels for K-Fold Cross-Validation from the *original* data for stratification
    # SMOTE is applied per fold to the training split, not globally then split.
    # So, K-Fold needs to happen on the original samples first.
    # Then, for each fold's training split, SMOTE is applied to *that split*.
    # For the validation split, only real data is used.

    # This means `create_synthetic_samples_per_class` should ideally be called *inside* the fold loop
    # for the training data, or we pre-generate a massive pool and manage indices carefully.
    # Let's adjust to common practice: K-fold on original data, then SMOTE on train split.

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_losses = []

    # Define common transformations (applied to the combined RGB image)
    transform = transforms.Compose([
        transforms.ToPILImage(), # Ensure it's PIL for Resize
        transforms.Resize((COMBINED_IMAGE_HEIGHT, IMAGE_SIZE[0] * n_leaves + (n_leaves - 1) * PADDING_BETWEEN_LEAVES)), # Resize to target combined size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet means and stds
    ])
    # NOTE: The above transform.Resize is problematic if n_leaves varies and images are concatenated
    # to variable widths. The CNN needs fixed input. The AdaptiveAvgPool2d solves the FC part,
    # but the Conv layers still need a consistent input size across a batch.
    # The `generate_n_ranked_leaves_for_cnn` produces a combined image of *variable width*.
    # For CNN, we need a FIXED INPUT SIZE.

    # REVISIT: CNN input handling for N-leaves.
    # Option 1: Pad/Crop all combined N-leaf images to a largest common size.
    # Option 2: Use a backbone that accepts variable input size (less common for custom CNNs).
    # Option 3: Rework generate_n_ranked_leaves_for_cnn to output fixed width for N=1, and another function for N>1
    #           that also pads to a fixed larger width if N>1.
    
    # Let's assume for now that the CombinedLeafDataset *outputs* a fixed size due to the `image_size` argument
    # passed to `generate_n_ranked_leaves_for_cnn` for *individual* leaves, and that the adaptive pooling handles the
    # variable final feature map. However, the initial `transforms.Resize` needs to match the expected
    # output of `generate_n_ranked_leaves_for_cnn`.
    # Let the `generate_n_ranked_leaves_for_cnn` produce its natural size, and then `__getitem__`
    # handles the final resizing to a fixed common size that the CNN expects.

    # The LeafCNN init now computes fc_input_features based on N=1 and then uses adaptive pooling.
    # For fixed CNN input, we need `generate_n_ranked_leaves_for_cnn` to pad/resize its output
    # to a consistent maximum combined width, or the `CombinedLeafDataset` does it.
    
    # Let's enforce a *maximum combined image width* that the CNN will train on,
    # and all generated images will be padded to this size.
    MAX_COMBINED_WIDTH = IMAGE_SIZE[0] * N_VALUES[-1] + (N_VALUES[-1] - 1) * PADDING_BETWEEN_LEAVES
    # This assumes N_VALUES is sorted and N_VALUES[-1] is the largest N.
    # If N=1, max width will be IMAGE_SIZE[0]. If N=10, it's 10 * IMAGE_SIZE[0] + 9 * PADDING.

    # Update transform to resize to a fixed (height, max_combined_width)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((COMBINED_IMAGE_HEIGHT, MAX_COMBINED_WIDTH)), # Resize to max combined width
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # And the CNN's forward pass should also take inputs of this fixed size.
    # This implies the LeafCNN's __init__ needs to compute the feature map size
    # based on (COMBINED_IMAGE_HEIGHT, MAX_COMBINED_WIDTH).
    # This requires changing the LeafCNN's `__init__` again to use MAX_COMBINED_WIDTH.
    # The current LeafCNN with AdaptiveAvgPool2d should be more robust to variable input size.
    # Let's simplify and make the `generate_n_ranked_leaves_for_cnn` output a fixed size.
    # Or, the `CombinedLeafDataset` should do it. Let's make `CombinedLeafDataset` ensure fixed size.

    # Revised __getitem__ in CombinedLeafDataset will pad the image *before* returning it.
    
    for fold, (train_index, val_index) in enumerate(skf.split(original_pca_scores_all, encoded_labels_all)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        # Get original data for this fold
        fold_train_pca_scores = original_pca_scores_all[train_index]
        fold_train_class_labels = original_class_labels_all[train_index]
        fold_val_pca_scores = original_pca_scores_all[val_index]
        fold_val_class_labels = original_class_labels_all[val_index]

        # Apply SMOTE only to the training data for this fold
        resampled_train_pca_scores, resampled_train_encoded_labels, smote_applied_strategy = \
            create_synthetic_samples_per_class(
                fold_train_pca_scores,
                fold_train_class_labels,
                class_label_encoder,
                min_samples_for_smote_target
            )
        
        # Build training recipes: mix of real and synthetic (based on SMOTE output)
        train_sample_recipes = []
        train_encoded_labels_for_kf = [] # For stratified sampling within recipes if needed

        # Prepare a lookup for original indices based on the current fold's train_index
        # This maps global original_flattened_coords index to its position in `pca_data_for_all_experiments['original_flattened_coords']`
        # for a real sample from this fold.
        # It's actually simpler: `generate_n_ranked_leaves_for_cnn` just needs *any* `n` real indices
        # from the set of real samples available to the *current class* in *this fold*.
        # So we collect the original indices available in `train_index` for each class.

        # Map original_class_labels_all indices to their decoded plantIDs
        original_indices_by_class_in_fold = {cls_str: [] for cls_str in class_label_encoder.classes_}
        for global_idx in train_index:
            class_str = original_class_labels_all[global_idx]
            original_indices_by_class_in_fold[class_str].append(global_idx) # Store global index

        # Create recipes for training set
        # The number of combined samples per class (real or synthetic) needs to be defined here.
        # Let's say we want to generate `target_combined_samples_per_class` for training for each class.
        target_combined_samples_per_class = SAMPLES_PER_CLASS_TARGET * 3 # Adjust as needed

        for encoded_label in class_label_encoder.classes_:
            class_label_str = class_label_encoder.inverse_transform([encoded_label])[0]
            
            # Real samples for training: Use `original_indices_by_class_in_fold`
            available_real_indices_for_class_in_fold = original_indices_by_class_in_fold.get(class_label_str, [])
            num_available_real_in_fold = len(available_real_indices_for_class_in_fold)

            # How many real N-leaf samples to try to create?
            num_real_combined_to_generate = min(target_combined_samples_per_class // 2, num_available_real_in_fold // n_leaves)
            if num_real_combined_to_generate > 0 and num_available_real_in_fold >= n_leaves:
                for _ in range(num_real_combined_to_generate):
                    selected_original_indices = random.sample(available_real_indices_for_class_in_fold, n_leaves)
                    train_sample_recipes.append({
                        'class_label_str': class_label_str,
                        'encoded_label': encoded_label,
                        'is_synthetic': False,
                        'original_indices': selected_original_indices # Pass these to generate_n_ranked_leaves_for_cnn
                    })
                    train_encoded_labels_for_kf.append(encoded_label)
            else:
                if n_leaves > num_available_real_in_fold:
                    print(f"  Warning: Class '{class_label_str}' in training fold {fold+1} has only {num_available_real_in_fold} real samples, less than n_leaves={n_leaves}. Cannot generate pure real N-leaf samples.")
            
            # Synthetic samples for training: Make up the rest to reach target_combined_samples_per_class
            # SMOTE has already balanced the `resampled_train_pca_scores` pool.
            # We now draw 'synthetic' combined samples from this balanced pool.
            # `generate_n_ranked_leaves_for_cnn` will handle the actual PCA-to-shape generation.
            
            # How many synthetic N-leaf samples to create?
            num_synthetic_combined_to_generate = target_combined_samples_per_class - num_real_combined_to_generate
            if num_synthetic_combined_to_generate > 0:
                for _ in range(num_synthetic_combined_to_generate):
                    train_sample_recipes.append({
                        'class_label_str': class_label_str,
                        'encoded_label': encoded_label,
                        'is_synthetic': True,
                        'original_indices': None # For synthetic, generate from SMOTE pool within generate_n_ranked_leaves_for_cnn
                    })
                    train_encoded_labels_for_kf.append(encoded_label)
        
        # Validation recipes: Only use real samples from the validation split
        val_sample_recipes = []
        val_encoded_labels_for_kf = []

        val_original_indices_by_class_in_fold = {cls_str: [] for cls_str in class_label_encoder.classes_}
        for global_idx in val_index:
            class_str = original_class_labels_all[global_idx]
            val_original_indices_by_class_in_fold[class_str].append(global_idx)

        num_val_combined_per_class = min(SAMPLES_PER_CLASS_TARGET, np.max(
            [len(v) for v in val_original_indices_by_class_in_fold.values()]
        ) // n_leaves if val_original_indices_by_class_in_fold else SAMPLES_PER_CLASS_TARGET)
        num_val_combined_per_class = max(1, num_val_combined_per_class) # At least 1 if possible
        
        for encoded_label in class_label_encoder.classes_:
            class_label_str = class_label_encoder.inverse_transform([encoded_label])[0]
            available_real_indices_for_val_in_fold = val_original_indices_by_class_in_fold.get(class_label_str, [])
            
            if len(available_real_indices_for_val_in_fold) >= n_leaves:
                for _ in range(num_val_combined_per_class):
                    selected_original_indices = random.sample(available_real_indices_for_val_in_fold, n_leaves)
                    val_sample_recipes.append({
                        'class_label_str': class_label_str,
                        'encoded_label': encoded_label,
                        'is_synthetic': False, # Validation should always be real
                        'original_indices': selected_original_indices
                    })
                    val_encoded_labels_for_kf.append(encoded_label)
            else:
                 print(f"  Warning: Class '{class_label_str}' in validation fold {fold+1} has only {len(available_real_indices_for_val_in_fold)} real samples, less than n_leaves={n_leaves}. Skipping for validation combined samples.")


        if not train_sample_recipes or not val_sample_recipes:
            print(f"Skipping fold {fold+1} due to insufficient samples for training or validation after recipe generation.")
            continue # Skip this fold

        train_dataset = CombinedLeafDataset(
            train_sample_recipes, n_leaves, pca_data_for_all_experiments, ect_calculator_instance,
            global_ect_min, global_ect_max, image_size, transform=transform, verbose=True
        )
        val_dataset = CombinedLeafDataset(
            val_sample_recipes, n_leaves, pca_data_for_all_experiments, ect_calculator_instance,
            global_ect_min, global_ect_max, image_size, transform=transform, verbose=True
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        model = LeafCNN(num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Calculate class weights for this fold's training set
        train_labels_for_weights = [r['encoded_label'] for r in train_sample_recipes]
        class_counts_fold = Counter(train_labels_for_weights)
        total_samples_fold = len(train_labels_for_weights)
        
        # Handle classes that might not appear in a fold's training set (though stratified should prevent this mostly)
        class_weights_fold = torch.tensor([
            total_samples_fold / (num_classes * class_counts_fold.get(i, 1)) # Use .get(i,1) to avoid division by zero
            for i in range(num_classes)
        ], dtype=torch.float32).to(device)
        print(f"⚖️ Calculated class weights for fold {fold+1}: {class_weights_fold.cpu().numpy()}")

        criterion = nn.CrossEntropyLoss(weight=class_weights_fold) # Use class weights

        best_val_accuracy = 0.0
        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # Save the best model for this fold
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_save_path = os.path.join(models_dir, f'best_model_n{n_leaves}_fold{fold+1}.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f"⭐ Saved best model for Fold {fold+1} with accuracy: {best_val_accuracy:.4f}")

        fold_accuracies.append(best_val_accuracy)
        # Store the loss from the best accuracy epoch (need to re-evaluate or store history)
        # For simplicity, we'll just store the final val_loss of the last epoch if not tracked
        # more granularly, but a better approach would save val_loss from the best epoch.
        fold_losses.append(val_loss) 

    # Report results for the N-leaf experiment
    if fold_accuracies:
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        print(f"\n✅ N={n_leaves} Experiment Complete.")
        print(f"Average Validation Accuracy across {k_folds} folds: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    else:
        print(f"\n❌ N={n_leaves} Experiment Failed. No fold accuracies recorded.")


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Step 0: Initialize Global ECT Calculator ---
    GLOBAL_ECT_CALCULATOR = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)

    # --- Step 1: Load Global PCA data ---
    print("--- 📚 Loading global PCA data ---")

    dataset_config_name = "plant_predict"
    current_dataset_config = DATASET_CONFIGS[dataset_config_name]

    # Construct the exact file paths as expected by load_pca_model_data in script 04
    pca_params_file_path = Path(os.path.join(PCA_DATA_PATH, current_dataset_config['PCA_PARAMS_FILE_NAME']))
    pca_scores_labels_file_path = Path(os.path.join(PCA_DATA_PATH, current_dataset_config['PCA_SCORES_LABELS_FILE_NAME']))
    class_column_name = current_dataset_config['CLASS_COLUMN']
    exclude_classes_list = current_dataset_config['EXCLUDE_CLASSES']

    GLOBAL_PCA_DATA = load_pca_model_data( # Store in global variable
        pca_params_file_path,
        pca_scores_labels_file_path,
        class_column_name,
        exclude_classes_list
    )

    if GLOBAL_PCA_DATA['original_flattened_coords'] is None:
        print(f"Error: 'original_flattened_coords' not found in PCA data loaded by _04_synthetic_data_generator_library.py. Cannot calculate global ECT min/max.")
        sys.exit(1)

    # --- Step 2: Calculate Global ECT Min/Max ---
    # This calculation needs to be done *after* loading PCA data,
    # as it uses the original flattened coordinates (real and potentially synthetic)
    # to determine the min/max range for normalization of ECT maps.
    print("--- Calculating ECT Min/Max for Overall Dataset ---")

    # Use a dummy LabelEncoder for this initial SMOTE call for min/max
    ect_calc_label_encoder = LabelEncoder()
    # Fit on all original labels before any exclusion to get full range, then filter
    ect_calc_label_encoder.fit(GLOBAL_PCA_DATA['original_class_labels'])

    # Temporary call to create_synthetic_samples_per_class to get the full pool of
    # PCA scores (original + synthetic) that could exist in the dataset.
    # We don't care about the 'is_synthetic' flag or `sampling_strategy` here,
    # just the combined `resampled_pca_scores` that represent the potential range of data.
    resampled_pca_scores_for_minmax, _, _ = create_synthetic_samples_per_class(
        GLOBAL_PCA_DATA['original_pca_scores'],
        GLOBAL_PCA_DATA['original_class_labels'],
        ect_calc_label_encoder,
        SAMPLES_PER_CLASS_TARGET
    )
    
    # Convert resampled PCA scores back to flattened coordinates for ECT calculation
    synthetic_flattened_coords_for_minmax = inverse_transform_pca(
        resampled_pca_scores_for_minmax,
        GLOBAL_PCA_DATA['components'],
        GLOBAL_PCA_DATA['mean']
    )
    
    # Combine original and synthetic flattened coordinates for a comprehensive min/max calculation
    # Ensure not to vstack empty arrays
    if GLOBAL_PCA_DATA['original_flattened_coords'].size > 0 and synthetic_flattened_coords_for_minmax.size > 0:
        combined_flattened_coords_for_minmax_calc = np.vstack([
            GLOBAL_PCA_DATA['original_flattened_coords'],
            synthetic_flattened_coords_for_minmax
        ])
    elif GLOBAL_PCA_DATA['original_flattened_coords'].size > 0:
        combined_flattened_coords_for_minmax_calc = GLOBAL_PCA_DATA['original_flattened_coords']
    elif synthetic_flattened_coords_for_minmax.size > 0:
        combined_flattened_coords_for_minmax_calc = synthetic_flattened_coords_for_minmax
    else:
        combined_flattened_coords_for_minmax_calc = np.array([])


    GLOBAL_ECT_MIN, GLOBAL_ECT_MAX = calculate_ect_min_max_for_dataset(
        combined_flattened_coords_for_minmax_calc,
        GLOBAL_ECT_CALCULATOR,
        APPLY_RANDOM_ROTATION=APPLY_RANDOM_ROTATION, # Use constant from script 04
        dataset_name="Overall Dataset for Global ECT Min/Max"
    )
    print(f"🌍 Global ECT Min: {GLOBAL_ECT_MIN:.4f}, Global ECT Max: {GLOBAL_ECT_MAX:.4f}")

    # --- Experiment Configuration for Different N Values ---
    # N_VALUES = [1, 2, 5] # Original idea was to vary N, but current implementation fixes to 1 for simplicity.
    N_VALUES = [1] # Keep fixed to N=1 as per current need, until multi-leaf input is fully robust

    for n_leaves in N_VALUES:
        experiment_config = {
            'dataset_name': current_dataset_config['DATASET_FULL_NAME'].replace(" ", "_"),
            'learning_rate': 0.001,
            'num_epochs': 50,
            'batch_size': 32,
            'k_folds': 5,
            'image_size': IMAGE_SIZE, # Use the imported IMAGE_SIZE from script 04
        }
        run_experiment_for_n(
            n_leaves,
            experiment_config,
            DEVICE,
            GLOBAL_PCA_DATA, # Pass global PCA data
            GLOBAL_ECT_CALCULATOR, # Pass global ECT calculator
            GLOBAL_ECT_MIN,
            GLOBAL_ECT_MAX
        )
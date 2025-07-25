# Disentangling Latent Genetic and Environmental Effects on Coca Leaf Shape using Deep Learning

## Figures and Tables
![alt text](https://github.com/DanChitwood/coca_leaf_CNN/blob/main/COCA_PROJECT/figures/Figure1.png "Figure 1")

![alt text](https://github.com/DanChitwood/coca_leaf_CNN/blob/main/COCA_PROJECT/figures/Figure2.png "Figure 2")

| Dataset         | Class          |   Precision |   Recall |     F1 |
|:----------------|:---------------|------------:|---------:|-------:|
| Plowman         | macro avg.     |      0.9773 |   0.962  | 0.9691 |
| Plowman         | weighted avg.  |      0.9648 |   0.9643 | 0.9642 |
| Plowman         | coca           |      0.9545 |   0.9767 | 0.9655 |
| Plowman         | ipadu          |      1      |   1      | 1      |
| Plowman         | novogranatense |      0.9545 |   0.9545 | 0.9545 |
| Plowman         | truxillense    |      1      |   0.9167 | 0.9565 |
| Cultigens (1st) | macro avg.     |      0.906  |   0.9491 | 0.925  |
| Cultigens (1st) | weighted avg.  |      0.9171 |   0.9091 | 0.9101 |
| Cultigens (1st) | coca           |      0.9691 |   0.8722 | 0.9181 |
| Cultigens (1st) | ipadu          |      0.7935 |   0.9241 | 0.8538 |
| Cultigens (1st) | novogranatense |      0.9091 |   1      | 0.9524 |
| Cultigens (1st) | truxillense    |      0.9524 |   1      | 0.9756 |
| Wildspecies     | macro avg.     |      0.9743 |   0.989  | 0.9814 |
| Wildspecies     | weighted avg.  |      0.982  |   0.9818 | 0.9818 |
| Wildspecies     | coca           |      0.9835 |   0.9944 | 0.989  |
| Wildspecies     | ipadu          |      0.9872 |   0.9747 | 0.9809 |
| Wildspecies     | novogranatense |      1      |   1      | 1      |
| Wildspecies     | truxillense    |      0.9524 |   1      | 0.9756 |
| Wildspecies     | cataractarum   |      0.9734 |   0.9683 | 0.9708 |
| Wildspecies     | foetidum       |      0.9556 |   1      | 0.9773 |
| Wildspecies     | gracilipes     |      0.9949 |   0.975  | 0.9848 |
| Wildspecies     | lineolatum     |      0.9474 |   1      | 0.973  |

![alt text](https://github.com/DanChitwood/coca_leaf_CNN/blob/main/COCA_PROJECT/figures/Figure3.png "Figure 3")

| Class                  |   Precision 1st |   Recall 1st |   F1 1st |   Precision 2nd |   Recall 2nd |   F1 2nd |   Precision Combined |   Recall Combined |   F1 Combined |
|:-----------------------|----------------:|-------------:|---------:|----------------:|-------------:|---------:|---------------------:|------------------:|--------------:|
| amazona                |               1 |            1 |        1 |          0.6784 |       0.9062 |   0.7759 |               0.9108 |            0.9728 |        0.9408 |
| boliviana blanca       |               1 |            1 |        1 |          0.8333 |       0.8385 |   0.8359 |               0.9326 |            0.9171 |        0.9248 |
| boliviana roja         |               1 |            1 |        1 |          0.8103 |       0.5912 |   0.6836 |               0.9341 |            0.8715 |        0.9017 |
| chiparra               |               1 |            1 |        1 |          0.9242 |       0.5382 |   0.6803 |               0.8908 |            0.8833 |        0.887  |
| chirosa                |               1 |            1 |        1 |          0.8371 |       0.9198 |   0.8765 |               0.9405 |            0.956  |        0.9482 |
| crespa                 |               1 |            1 |        1 |          0.9605 |       0.914  |   0.9366 |               0.9803 |            0.966  |        0.9731 |
| dulce                  |               1 |            1 |        1 |          0.7017 |       0.8247 |   0.7582 |               0.8791 |            0.9195 |        0.8989 |
| gigante                |               1 |            1 |        1 |          0.7865 |       0.8974 |   0.8383 |               0.9461 |            0.8977 |        0.9213 |
| guayaba roja           |               1 |            1 |        1 |          0.8263 |       0.9079 |   0.8652 |               0.9278 |            0.9709 |        0.9489 |
| patirroja              |               1 |            1 |        1 |          0.678  |       0.8476 |   0.7534 |               0.9247 |            0.9348 |        0.9297 |
| peruana roja           |               1 |            1 |        1 |          0.5833 |       0.7636 |   0.6614 |               0.8168 |            0.8919 |        0.8527 |
| tingo maria            |               1 |            1 |        1 |          0.8525 |       0.6582 |   0.7429 |               0.9701 |            0.9101 |        0.9391 |
| tingo pajarita         |               1 |            1 |        1 |          0.8475 |       0.9615 |   0.9009 |               0.976  |            0.9839 |        0.9799 |
| tingo pajarita caucana |               1 |            1 |        1 |          0.965  |       0.965  |   0.965  |               0.9818 |            0.9818 |        0.9818 |
| tingo peruana          |               1 |            1 |        1 |          0.7778 |       0.7727 |   0.7752 |               0.9207 |            0.8678 |        0.8935 |
| trujillense caucana    |               1 |            1 |        1 |          0.8089 |       0.8141 |   0.8115 |               0.9438 |            0.9545 |        0.9492 |
| macro avg.             |               1 |            1 |        1 |          0.8045 |       0.82   |   0.8038 |               0.9298 |            0.93   |        0.9294 |
| weighted avg.          |               1 |            1 |        1 |          0.8166 |       0.8003 |   0.7973 |               0.9278 |            0.9268 |        0.9268 |

![alt text](https://github.com/DanChitwood/coca_leaf_CNN/blob/main/COCA_PROJECT/figures/Figure4.png "Figure 4")

## Methods  
This study employs a comprehensive computational pipeline for the morphometric analysis and image-based classification of leaf shapes, encompassing feature extraction using the Euler Characteristic Transform (ECT), synthetic data augmentation, and a custom Convolutional Neural Network (CNN) architecture. All computational steps were performed using Python (version 3.12.2) with key libraries including NumPy (version 1.26.4), Pandas (version 2.0.3), Scikit-learn (version 1.3.2), Matplotlib (version 3.10.3), Seaborn (version 0.13.2), OpenCV (version 4.10.0.84), Pillow (version 10.3.0), PyTorch (version 2.4.1), and ect (version 1.0.3). For reproducibility, a global random seed (42) was consistently applied across all random operations. All data and code necessary to reproduce these results are available at: https://github.com/DanChitwood/coca_leaf_CNN
#### Initial Shape Representation and Principal Component Analysis (PCA)
Leaf contours were initially represented by 99 pseudo-landmarks equidistantly placed along the outline and anchored by tip and base points. These points were flattened into 198-dimensional vectors to form the basis of the shape dataset. Principal Component Analysis (PCA) was then applied to this high-dimensional coordinate space to reduce dimensionality while capturing the principal modes of morphological variation. Separate PCA models were trained and saved for each distinct dataset. These pre-trained PCA models were subsequently loaded for inverse transformations, enabling the reconstruction of 2D leaf contours from their PCA scores.
#### Synthetic Data Generation via SMOTE-like Augmentation
To mitigate potential class imbalances and augment the training data, a Synthetic Minority Over-sampling Technique (SMOTE)-like approach was implemented directly in the PCA feature space. For each distinct leaf class, the method proceeded by identifying its k=5 nearest neighbors within the same class in the PCA feature space. A synthetic sample was then generated by randomly selecting an existing sample from the class and interpolating its PCA scores with a randomly selected nearest neighbor. The interpolation factor (α) was a random value between 0 and 1. This process was iteratively repeated until each class contained a target of 400 synthetic samples. The PCA scores of these synthetic samples were then inverse-transformed using the corresponding collection's PCA model to reconstruct their 2D contour points in the original coordinate space. This augmentation strategy was applied independently to each dataset, significantly expanding the training data for robust model development.
#### Euler Characteristic Transform (ECT) Calculation and Image Generation
A primary morphometric feature used as input for the CNN was the Euler Characteristic Transform (ECT), which provides a powerful image-based representation of closed contours. For each original (real) and newly generated (synthetic) leaf shape, the ECT calculation involved the following steps: First, for synthetic samples, PCA scores were inverse-transformed back into the original 198-dimensional flattened coordinate space to reconstruct the 99 (x, y) contour points. These contour points were then used to construct an EmbeddedGraph object. The EmbeddedGraph underwent internal normalization, including centering, a Procrustes-like alignment for standardized orientation and size, and uniform scaling to be contained within a bounding circle of radius 1. Notably, no additional random rotations were applied during this step. Finally, the ECT was calculated using 180 radial directions and 180 linearly spaced distance thresholds from 0 to the bounding radius (1), yielding a 180x180 matrix of ECT values for each leaf shape. The cartesian matrix representation of the ECT was converted to radial coordinates and aligned with the leaf outline shape mask.
To prepare these ECT features for CNN input, the 180x180 ECT matrices were rendered as 256x256 single-channel grayscale images. A global minimum and maximum ECT value, determined across all combined real and synthetic datasets, was used to normalize pixel intensities consistently. The final CNN input consisted of a two-channel image: one channel containing the grayscale ECT image and the second channel containing a 256x256 binary shape mask (indicating leaf presence). For the combined dataset's cultigen classification, input images were downscaled to 64x64 pixels.
#### Convolutional Neural Network (CNN) Architecture and Training
Separate CNN models were trained and evaluated for classification for each analysis. The CNN architecture, implemented in PyTorch, was designed as a sequential model comprising: An input layer accepting two-channel images (256x256 for individual collections, 64x64 for the combined dataset). Three sequential convolutional blocks, each with a Conv2d layer (3x3 kernel, 1-pixel padding, increasing filter counts: 32, 64, 128), followed by BatchNorm2d, ReLU activation, and MaxPool2d (2x2 kernel, stride 2). A classifier head consisting of a Flatten layer, a Linear (dense) layer with 512 units and ReLU activation, a Dropout layer (0.5 probability), and a final Linear layer outputting logits for each unique class.
Model training employed a 5-fold stratified cross-validation strategy. For each fold, the validation set consisted exclusively of real leaf samples, while the training set comprised all synthetic samples combined with real samples from the training split. Class weights, computed using sklearn.utils.class_weight.compute_class_weight('balanced') based on all training labels, were applied to the CrossEntropyLoss criterion. The Adam optimizer was used with a learning rate of 0.001. A ReduceLROnPlateau scheduler monitored validation loss (patience 5 epochs, factor 0.1). Early stopping (patience 10 epochs) ceased training if validation loss did not improve, and the best validation weights were saved. Final predictions were derived by averaging raw logits from the best models of all 5 folds.
#### Model Evaluation and Interpretability
Model performance was evaluated on real samples using accuracy, precision, recall, and F1-score. Classification reports and confusion matrices were generated. Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations were computed by targeting the final convolutional layer of a trained model (Fold 0), averaging heatmaps from 5 randomly selected real samples per class, and overlaying them onto representative ECT images to identify influential regions.

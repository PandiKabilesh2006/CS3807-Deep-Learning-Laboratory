# CNN for Image Classification on CIFAR-10

> Deep Learning Laboratory – Experiment 3  
> Shiv Nadar University Chennai  
> Course: CS3807 – Deep Learning Laboratory

## Overview

This project implements a custom **Convolutional Neural Network (CNN)** from scratch using TensorFlow/Keras to perform multi-class image classification on the **CIFAR-10** dataset. 

In addition to building and training the primary CNN classifier, the experiment covers:
- Visualizing intermediate layer **feature maps** to inspect what features the convolutional layers learn.
- A comparative study of **Max Pooling** vs. **Average Pooling** architectures to evaluate their impact on learning dynamics and accuracy.

---

## Dataset

**CIFAR-10**
- 50,000 Training Images
- 10,000 Testing Images
- 10 Image Categories
- Image Size: **32 × 32 × 3** (RGB color images)

Classes include:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## Project Workflow

```
       Load CIFAR-10 Dataset
                 │
                 ▼
         Data Preprocessing
  (Normalization to [0,1], train/val split)
                 │
                 ▼
         Build Custom CNN
 (3x Conv-MaxPool Blocks + Dense + Dropout)
                 │
                 ▼
     Train & Evaluate Primary CNN
  (20 epochs, Adam, Sparse Crossentropy)
                 │
                 ▼
    Intermediate Feature Map Extraction
(Visualizing outputs of individual Conv layers)
                 │
                 ▼
   Pooling Comparison Experiments
   (Max Pooling vs. Average Pooling)
                 │
                 ▼
    Final Evaluation & Visualizations
```

---

## Model Architecture

The custom CNN architecture consists of three convolutional blocks followed by fully connected layers:

```
Input Layer (32 × 32 × 3)
      │
      ▼
Conv2D (32 filters, 3x3, ReLU, same padding)
      │
      ▼
MaxPooling2D (2x2)
      │
      ▼
Conv2D (64 filters, 3x3, ReLU, same padding)
      │
      ▼
MaxPooling2D (2x2)
      │
      ▼
Conv2D (128 filters, 3x3, ReLU, same padding)
      │
      ▼
MaxPooling2D (2x2)
      │
      ▼
Flatten
      │
      ▼
Dense (128 neurons, ReLU)
      │
      ▼
Dropout (0.5 rate)
      │
      ▼
Dense (10 neurons, Softmax)
```

### Training Configuration:
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** 64
- **Epochs:** 20
- **Validation Split:** 10%

---

## Results

### Custom CNN Performance
- **Test Loss:** `0.8892`
- **Test Accuracy:** `74.34%`

---

## Pooling Comparison Study

To compare the impact of different pooling strategies on spatial dimension reduction, two simplified architectures (1 convolutional layer + pooling + flatten + dense output) were trained for **10 epochs**:

1. **Max Pooling Model:** Achieved **70.94%** training accuracy and **64.60%** validation accuracy.
2. **Average Pooling Model:** Achieved **65.73%** training accuracy and **60.50%** validation accuracy.

### Conclusion
Max Pooling performed superiorly compared to Average Pooling. This is because Max Pooling extracts the most prominent features (edges, corners, sharp transitions), which are highly descriptive for image classification, while Average Pooling tends to smooth out local details.

---

## Visualizations

The experiment generates publication-quality PDF plots (600 DPI) mapping performance:
- **Sample Images:** Grid of CIFAR-10 training images with titles.
- **Class Distribution:** Frequency of each category in the dataset.
- **Training and Validation Curves:** Accuracy and Loss progression over 20 epochs.
- **Confusion Matrix:** Detail of correct vs. misclassified classes on the test set.
- **Predictions:** visual representation of model predictions on test samples.
- **Feature Maps:** Activation maps showing the output of the convolutional layers.
- **Max vs Avg Pooling Comparison:** Plot of training and validation accuracy/loss for both pooling configurations.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Repository Structure

```
Lab 3 CNN for Image Classification/
├── cnn_for_cifar_10.ipynb                 # Jupyter Notebook containing all experiment code
├── 24011101075_DL_Lab_Exercise_3.pdf       # Lab Report with detailed writeup
├── Sample_images.pdf                      # Grid of sample dataset images
├── Class_Distribution.pdf                 # Category frequency distribution
├── Training_and_Validation_Accuracy.pdf   # Accuracy curves
├── Training_and_Validation_Loss.pdf       # Loss curves
├── Feature_Map.pdf                        # Visualization of internal Conv layers
├── Confusion_Matrix.pdf                   # Performance heatmap per class
├── Predictions.pdf                        # Test set predictions visualization
└── Max_vs_Avg_Pooling.pdf                 # Pooling comparison curves
```

---

## Learning Outcomes

- Implementing 2D convolutional layers, max pooling, and dropout using TensorFlow.
- Normalization and preprocessing of multi-channel RGB image datasets.
- Extracting and analyzing intermediate layer feature maps to understand CNN features.
- Evaluating the mathematical and practical differences between Max Pooling and Average Pooling.
- Generating clean, publication-ready data visualizations.

---

## References

1. Ian Goodfellow, Yoshua Bengio, Aaron Courville – *Deep Learning*
2. Stanford CS231n: *Convolutional Neural Networks for Visual Recognition*
3. TensorFlow Keras API Documentation
4. Alex Krizhevsky – *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10 dataset)

---

⭐ If you found this project helpful, consider giving the repository a star!

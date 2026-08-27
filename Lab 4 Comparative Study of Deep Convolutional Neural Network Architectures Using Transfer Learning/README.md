# Transfer Learning & Fine-Tuning with VGG16 on CIFAR-10

> Deep Learning Laboratory – Experiment 4  
> Shiv Nadar University Chennai  
> Course: CS3807 – Deep Learning Laboratory

## Overview

This project performs multi-class image classification on the **CIFAR-10** dataset using **Transfer Learning** and **Fine-Tuning** with a pre-trained **VGG16** network. 

The study is executed in two stages:
1. **Feature Extraction:** The convolutional base of VGG16 (pre-trained on ImageNet) is frozen. A custom classification head is trained on top of these features.
2. **Fine-Tuning:** The weights of the upper convolutional layers (`block5` of VGG16) are unfrozen and trained alongside the classification head using a very low learning rate to adapt features to the target domain.

---

## Dataset

**CIFAR-10**
- 50,000 Training Images
- 10,000 Testing Images
- 10 Image Categories
- Input Preprocessing: The images are upscaled to **96 × 96 × 3** using bilinear interpolation and normalized using VGG16's ImageNet mean subtraction.

Classes include:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

---

## Project Workflow

```
Load & Resize CIFAR-10 to 96x96
              │
              ▼
Load Pretrained VGG16 (ImageNet weights)
              │
              ▼
Freeze Convolutional Base (Trainable = False)
              │
              ▼
Attach Custom Classifier Top
(GAP + Dense 256 + Dropout 0.3 + Softmax 10)
              │
              ▼
Phase 1: Train Classifier (Feature Extraction)
     (10 epochs, LR = 0.001, Adam)
              │
              ▼
Phase 2: Fine-Tuning (Unfreeze block5)
       (5 epochs, LR = 1e-5, Adam)
              │
              ▼
Final Model Evaluation & Metric Logging
```

---

## Model Architecture

```
Input Image (96 × 96 × 3)
      │
      ▼
Pre-trained VGG16 Base (frozen conv blocks 1-4)
      │
      ▼
VGG16 Block 5 (unfrozen during fine-tuning stage)
      │
      ▼
GlobalAveragePooling2D
      │
      ▼
Dense (256 neurons, ReLU)
      │
      ▼
Dropout (0.3 rate)
      │
      ▼
Dense (10 neurons, Softmax)
```

- **Total Parameters:** `14,848,586`
- **Trainable Parameters (Feature Extraction):** `3,858,954`
- **Trainable Parameters (Fine-Tuning):** `11,927,306`

---

## Results

| Phase / Metric | Training Accuracy | Validation/Testing Accuracy | Loss | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Feature Extraction (10 Epochs)** | 89.32% | 82.86% | 0.6378 (val) | 1018.16 s |
| **Fine-Tuning (5 Epochs)** | 97.45% | 88.30% (val) | 0.4986 (val) | 593.99 s |
| **Final Test Set Evaluation** | — | **88.30%** | **0.4986** | **Total: 1612.14 s** |

### Final Evaluation Metrics
- **Test Accuracy:** `88.30%`
- **Macro Precision:** `88.37%`
- **Macro Recall:** `88.30%`
- **Macro F1 Score:** `88.32%`

---

## Key Observations

- **Transfer Learning Efficacy:** Freezing the pre-trained weights and training only the classification head quickly achieved over 82.86% validation accuracy, bypassing the need to train a deep network from scratch.
- **Fine-Tuning Boost:** Unfreezing the `block5` convolutional layers of VGG16 with a lower learning rate (`1e-5`) allowed the model to refine high-level feature extractors for the CIFAR-10 dataset, raising the test accuracy to **88.30%** (a **+5.44%** increase).
- **Overfitting Management:** Dropout (0.3) and Global Average Pooling helped manage overfitting, though training accuracy approached 97.45%, indicating some specialization to the training set.

---

## Visualizations

Plots are saved in publication-quality PDF format (600 DPI, Times New Roman):
- `01_sample_images.pdf`: Grid of resized sample CIFAR-10 images.
- `02_initial_accuracy.pdf` & `02_initial_loss.pdf`: Performance curves during the Feature Extraction stage.
- `03_finetune_accuracy.pdf` & `03_finetune_loss.pdf`: Performance curves during the Fine-Tuning stage.
- `04_confusion_matrix.pdf`: Class-by-class heatmap of final model performance.
- `05_misclassified.pdf`: Samples of images misclassified by the model with true vs. predicted labels.

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
Lab 4 Comparative Study of Deep Convolutional Neural Network Architectures Using Transfer Learning/
├── 24011101075_DL_Lab_Exercise_4.ipynb   # Jupyter Notebook containing all training & evaluation code
├── 24011101075_DL_Lab_Exercise_4.pdf     # Lab Report detailing the study
├── 01_sample_images.pdf                  # Visualized dataset samples
├── 02_initial_accuracy.pdf               # Feature extraction phase validation/train accuracy curves
├── 02_initial_loss.pdf                   # Feature extraction phase validation/train loss curves
├── 03_finetune_accuracy.pdf              # Fine-tuning phase validation/train accuracy curves
├── 03_finetune_loss.pdf                  # Fine-tuning phase validation/train loss curves
├── 04_confusion_matrix.pdf               # Confusion matrix heatmap on test set
└── 05_misclassified.pdf                  # Grid of misclassified samples with true/pred labels
```

---

## Learning Outcomes

- Applying transfer learning workflows using pre-trained networks (VGG16).
- Resizing and preprocessing images to adapt to pre-trained architectures.
- Fine-tuning selective layers (unfreezing `block5` only) while freezing others.
- Using a dual learning rate schedule (higher lr for classifier head, lower lr for fine-tuned blocks).
- Creating comprehensive classification reports (Precision, Recall, F1 Score).

---

## References

1. Karen Simonyan, Andrew Zisserman – *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG16 Paper)
2. TensorFlow Keras Transfer Learning Guide
3. CIFAR-10 Dataset by Alex Krizhevsky

---

⭐ If you found this project helpful, consider giving the repository a star!

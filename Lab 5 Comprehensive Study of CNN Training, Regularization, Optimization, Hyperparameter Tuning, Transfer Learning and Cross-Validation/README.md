# CNN Training, Optimization, and Regularization Study

> Deep Learning Laboratory – Experiment 5  
> Shiv Nadar University Chennai  
> Course: CS3807 – Deep Learning Laboratory

## Overview

This project conducts an exhaustive comparative study of Convolutional Neural Network (CNN) design decisions, optimization methods, and regularization techniques using the **Oxford-IIIT Pet** dataset and a pre-trained **MobileNetV2** backbone. 

The experiment is structured into **10 sub-experiments**, systematically investigating hyperparameters and architecture selections before performing a **5-fold cross-validation** to select the final model for deployment.

---

## Dataset

**Oxford-IIIT Pet Dataset**
- Total Images: **7,349** (approx. 200 images per class)
- Classes: **37 categories** of dog and cat breeds (25 dogs, 12 cats)
- Image Size: Resized to **224 × 224 × 3** (RGB)
- Preprocessing: Min-max normalization and input scaling.
- Data Augmentation: Applied during training (Random Horizontal Flip, Random Rotation up to 0.1, Random Zoom up to 0.1).

---

## Project Workflow & The 10 Sub-Experiments

The project explores optimization across ten distinct tasks:

### 1. Baseline Model
Established a baseline classification head on top of MobileNetV2 with `learning_rate=0.001`, `optimizer=Adam`, `dropout=0.25`, and no L2 regularization or Batch Normalization.

### 2. Weight Initialization
Compared three weight initialization techniques for the dense classifier layers:
- **Random Uniform**
- **Xavier (Glorot) Uniform**
- **He Normal**

### 3. Regularization & Batch Normalization
Evaluated different methods of combating overfitting:
- **No Regularization**
- **L2 Regularization** (`l2_reg=0.001`)
- **Dropout** (`dropout_rate=0.5`)
- **Batch Normalization**

### 4. Optimizer Comparison
Analyzed training trajectory and rate of convergence for:
- **SGD** (Stochastic Gradient Descent)
- **Momentum** (SGD with momentum = 0.9)
- **RMSprop**
- **Adam**

### 5. Learning Rate Tuning
Compared learning rates of `0.001` (standard) vs. `0.0001` (fine).

### 6. Batch Size Tuning
Evaluated the effects of batch size on generalization and stability:
- **Batch Size:** 16, 32, and 64

### 7. Dropout Tuning
Tuned classifier dropout rate:
- **Dropout Rate:** 0.0, 0.25, and 0.5

### 8. Feature Extraction
Trained the model with the MobileNetV2 base completely frozen (trainable features act as static representation vectors).

### 9. Fine-Tuning MobileNetV2
Unfroze MobileNetV2 layers from layer 100 onwards and fine-tuned using a slow learning rate. Unlocked a peak validation accuracy of **77.99%**.

### 10. 5-Fold Cross-Validation
Identified the top 3 configuration candidates from previous trials and tested them under 5-Fold Cross-Validation:
- **C1_Baseline:** `dropout=0.25`, `l2_reg=0.0`, `use_batchnorm=False`, `lr=0.001`, `optimizer=Adam`
- **C2_BatchNorm:** `dropout=0.25`, `l2_reg=0.0`, `use_batchnorm=True`, `lr=0.0001`, `optimizer=Adam`
- **C3_L2_Dropout:** `dropout=0.5`, `l2_reg=0.001`, `use_batchnorm=True`, `lr=0.0001`, `optimizer=Adam`

---

## 5-Fold Cross-Validation Results

| Configuration | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean Accuracy | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1_Baseline** (Best) | 88.72% | 91.03% | 89.13% | 88.18% | 88.32% | **89.08%** | **0.0103** |
| **C2_BatchNorm** | 81.52% | 80.03% | 79.89% | 77.99% | 79.48% | **79.78%** | **0.0113** |
| **C3_L2_Dropout** | 73.51% | 74.59% | 75.95% | 72.96% | 77.04% | **74.81%** | **0.0151** |

---

## Best Configuration & Final Evaluation

The **C1_Baseline** configuration performed significantly better under cross-validation. It was trained on the complete dataset for **8 epochs** for final deployment.

### Final Performance on Test Set
- **Test Loss:** `0.3969`
- **Test Accuracy:** **87.90%**
- **Precision:** **89.21%**
- **Recall:** **87.90%**
- **F1 Score:** **87.83%**

---

## Visualizations

Plots are saved in publication-quality PDF format (600 DPI, Times New Roman, Font size 15):
- `dataset_samples.pdf` & `sample_images.pdf`: Visualizing raw images of cat/dog breeds in the dataset.
- `baseline_mobilenetv2_accuracy.pdf` & `baseline_mobilenetv2_loss.pdf`: Baseline model history.
- `weight_initialization_comparison.pdf`: Trajectories comparing Random Uniform, Xavier, and He initializations.
- `regularization_comparison.pdf` & `dropout_comparison.pdf`: Impact of regularization and dropout rates on training.
- `optimizer_comparison.pdf`: Comparison of convergence speed across SGD, Momentum, RMSprop, and Adam.
- `learning_rate_comparison.pdf`: Tuning curves for lr 0.001 vs 0.0001.
- `batch_size_comparison.pdf`: Validation trends across batch sizes 16, 32, 64.
- `feature_extraction_accuracy.pdf` & `feature_extraction_loss.pdf`: Feature extraction vs base model.
- `fine_tuning_accuracy.pdf` & `fine_tuning_loss.pdf`: Pretrained base tuning history.
- `cross_validation_comparison.pdf`: Boxplot/line comparisons of validation accuracies during 5-Fold Cross-Validation.
- `final_model_training_loss.pdf`: Training loss trajectory of the deployed model.
- `confusion_matrix.pdf`: Class-by-class heatmap of final classifications across all 37 breeds.
- `misclassified_images.pdf`: Grid showing specific misclassifications with predicted/true breeds.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Pandas

---

## Repository Structure

```
Lab 5 Comprehensive Study of CNN Training.../
├── 24011101075_DL_Lab_Exercise_5.ipynb     # Jupyter Notebook containing all 10 experiments & cross-validation
├── 24011101075_DL_Lab_Exercise_5.pdf       # Complete lab report and results analysis
├── baseline_mobilenetv2_accuracy.pdf
├── baseline_mobilenetv2_loss.pdf
├── batch_size_comparison.pdf
├── confusion_matrix.pdf
├── cross_validation_comparison.pdf
├── dataset_samples.pdf
├── dropout_comparison.pdf
├── feature_extraction_accuracy.pdf
├── feature_extraction_loss.pdf
├── final_model_training_loss.pdf
├── fine_tuning_accuracy.pdf
├── fine_tuning_loss.pdf
├── learning_rate_comparison.pdf
├── misclassified_images.pdf
├── optimizer_comparison.pdf
├── regularization_comparison.pdf
├── sample_images.pdf
└── weight_initialization_comparison.pdf
```

---

## Learning Outcomes

- Engineering automated data pipelines using the `tf.data` API (augmentation, batching, caching, prefetching).
- Comparative evaluation of optimization algorithms (SGD, Momentum, RMSprop, Adam).
- Comparative evaluation of regularization (L2, Dropout) and Batch Normalization.
- Designing systematic hyperparameter sweeps for learning rates, batch sizes, and dropouts.
- Implementing K-Fold Cross-Validation pipelines from scratch in TensorFlow.
- Analyzing fine-tuning strategies (freezing and selective unfreezing of pre-trained backbones).

---

## References

1. Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen – *MobileNetV2: Inverted Residuals and Linear Bottlenecks*
2. The Oxford-IIIT Pet Dataset (Parkhi et al.)
3. TensorFlow Core Tutorials: *Transfer Learning and Fine-Tuning*

---

⭐ If you found this project helpful, consider giving the repository a star!

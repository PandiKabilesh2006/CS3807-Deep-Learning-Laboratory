# Multi-Layer Perceptron (MLP) for Fashion-MNIST Classification

> Deep Learning Laboratory – Experiment 2  
> Shiv Nadar University Chennai  
> Course: CS3807 – Deep Learning Laboratory

## Overview

This project implements a **Multi-Layer Perceptron (MLP)** using TensorFlow/Keras to perform multi-class image classification on the **Fashion-MNIST** dataset.

The experiment covers the complete deep learning workflow, including:

- Data preprocessing
- MLP model construction
- Model training and evaluation
- Hyperparameter optimization using **RandomizedSearchCV** with **SciKeras**
- Performance comparison between the baseline and optimized models
- Visualization of training performance

---

## Dataset

**Fashion-MNIST**

- 60,000 Training Images
- 10,000 Testing Images
- 10 Clothing Categories
- Image Size: **28 × 28** grayscale

Classes include:

- T-shirt/Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

---

## Project Workflow

```
Load Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Build Baseline MLP
      │
      ▼
Train Baseline Model
      │
      ▼
Evaluate Baseline
      │
      ▼
Randomized Hyperparameter Search
      │
      ▼
5-Fold Cross Validation
      │
      ▼
Best Hyperparameters
      │
      ▼
Retrain Optimized Model
      │
      ▼
Final Evaluation
```

---

## Data Preprocessing

The following preprocessing steps were performed:

- Flattened each **28 × 28** image into a **784-dimensional vector**
- Normalized pixel values to the range **[0,1]**
- Converted labels into one-hot encoded vectors
- Split the dataset into training and testing sets

---

## Baseline Model Architecture

```
Input Layer (784)

↓

Dense(128, ReLU)

↓

Dense(64, ReLU)

↓

Dense(10, Softmax)
```

Training configuration:

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Batch Size: 32
- Epochs: 20

---

# Hyperparameter Optimization

Automated hyperparameter tuning was performed using:

- **RandomizedSearchCV**
- **SciKeras**
- **5-Fold Cross Validation**

### Search Space

| Hyperparameter | Values |
|---------------|--------|
| Hidden Layers | 1, 2, 3 |
| Hidden Neurons | 32, 64, 128, 256 |
| Learning Rate | 0.1, 0.01, 0.001 |
| Batch Size | 16, 32, 64, 128 |
| Epochs | 10, 20, 30 |
| Optimizer | SGD, Adam, RMSProp |
| Activation | ReLU, Tanh, Sigmoid |
| Dropout | 0.0, 0.2, 0.5 |

---

# Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 3 |
| Hidden Neurons | 128 |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Optimizer | RMSProp |
| Activation | Tanh |
| Dropout | 0.2 |
| Epochs | 30 |

---

# Results

## Baseline Model

| Metric | Score |
|--------|-------|
| Accuracy | **88.14%** |
| Precision | **88.22%** |
| Recall | **88.14%** |
| F1 Score | **88.12%** |

---

## Optimized Model

| Metric | Score |
|--------|-------|
| Cross-Validation Accuracy | **89.12%** |
| Test Accuracy | **87.62%** |
| Precision | **87.58%** |
| Recall | **87.62%** |
| F1 Score | **87.51%** |

---

# Observations

- The MLP successfully learned meaningful image representations from the Fashion-MNIST dataset.
- Hyperparameter optimization identified a configuration with a **higher mean cross-validation accuracy (89.12%)**.
- After retraining and testing, the optimized model achieved a **slightly lower test accuracy (87.62%)** than the baseline model (88.14%).
- This demonstrates that improved cross-validation performance does not always translate to improved performance on unseen test data.

---

# Visualizations

The project includes:

- Sample Images
- Class Distribution
- Training Accuracy vs Epoch
- Validation Accuracy vs Epoch
- Training Loss vs Epoch
- Validation Loss vs Epoch
- Confusion Matrix
- Hyperparameter Search Results
- Baseline vs Optimized Performance Comparison

---

# Technologies Used

- Python
- TensorFlow / Keras
- SciKeras
- Scikit-learn
- NumPy
- Matplotlib
- Pandas

---

# Repository Structure

```
.
├── MLP_(2).ipynb
├── Report.pdf
├── README.md
├── plots/
│   ├── Sample_images
│   ├── Class_Distribution
│   ├── Training_Accuracy_vs_Epoch
│   ├── Validation_Accuracy_vs_Epoch
│   ├── Training_Loss_vs_Epoch
│   ├── Validation_Loss_vs_Epoch
│   ├── Confusion_Matrix
│   ├── Randomized_Search_Results
│   └── Baseline_Vs_Optimized_Model_Accuracy
```

---

# Learning Outcomes

Through this experiment, the following concepts were explored:

- Image preprocessing for deep learning
- Multi-Layer Perceptron architecture
- TensorFlow/Keras model development
- Model evaluation using classification metrics
- Hyperparameter optimization with RandomizedSearchCV
- Cross-validation for model selection
- Performance visualization and analysis

---

## References

1. Ian Goodfellow, Yoshua Bengio, Aaron Courville – *Deep Learning*
2. Christopher Bishop – *Pattern Recognition and Machine Learning*
3. Simon Haykin – *Neural Networks and Learning Machines*
4. Fashion-MNIST Dataset
5. TensorFlow & Keras Documentation

---

⭐ If you found this project helpful, consider giving the repository a star!
# 📓 CS3807 - Deep Learning Laboratory

**SCHOOL NOTEBOOK / LAB RECORD**
* **Course:** CS3807 - Deep Learning Laboratory
* **Academic Corpus:** [CS3807-Deep-Learning-Laboratory](file:///e:/CS3807-Deep%20Learning%20Laboratory/)
* **Platform:** Python, Jupyter, NumPy, TensorFlow

---

## 📋 LABORATORY INDEX (TABLE OF CONTENTS)

| Ex. No. | Date | Experiment Title / Topic | Reference Link | Status |
| :---: | :---: | :--- | :---: | :---: |
| **1** | 2026-07-25 | **Single Layer Perceptron from Scratch**<br>• Custom neural network architecture using only Python & NumPy.<br>• Implementations of activation functions (Sigmoid), loss calculation, feedforward propagation, and gradient descent optimizer.<br>• Solved AND, OR, and NOT logic gates decision boundaries.<br>• Evaluated binary classification accuracy on the Banknote Authentication dataset. | [Lab 1 Directory](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%201%20Single%20Layer%20Perceptron/)<br>[Notebook](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%201%20Single%20Layer%20Perceptron/NN_From_Scratch.ipynb) | 🟢 **Completed** |
| **2** | 2026-07-25 | **Multi Layer Perceptron (MLP)**<br>• Implementation of MLP models using TensorFlow/Keras.<br>• Multi-class image classification on the Fashion MNIST dataset.<br>• Non-linear decision boundaries analysis (solving XOR Gate problem).<br>• Hyperparameter optimization via Randomized Search, tuning dropout rates, learning rates, epochs, and baseline model comparison. | [Lab 2 Directory](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%202%20Multi%20Layer%20Perceptron/)<br>[Notebook](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%202%20Multi%20Layer%20Perceptron/MLP.ipynb) | 🟢 **Completed** |

---

## 🛠️ Environment Setup & Installation

Before running any of the notebooks, configure your Python environment.

### 1. Prerequisites
* **Python 3.8+**
* **Jupyter Notebook / JupyterLab** or **VS Code** with Python and Jupyter extensions.

### 2. Install Core Dependencies
Use the provided [requirements.txt](file:///e:/CS3807-Deep%20Learning%20Laboratory/requirements.txt) to install common libraries:
```bash
pip install -r requirements.txt
```
For Lab 2, since it uses Keras/TensorFlow for the Multi Layer Perceptron, make sure to also install TensorFlow:
```bash
pip install tensorflow
```

---

## 🚀 Lab Run Instructions

### 📓 Lab 1: Single Layer Perceptron from Scratch

1. **Dataset Location:** Ensure that [data_banknote_authentication.txt](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%201%20Single%20Layer%20Perceptron/data_banknote_authentication.txt) is present in the lab folder.
2. **Open Notebook:** Open [NN_From_Scratch.ipynb](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%201%20Single%20Layer%20Perceptron/NN_From_Scratch.ipynb).
3. **Run Notebook:** Select your Python kernel/environment and run all cells sequentially.
4. **Output Visualizations:** The notebook generates:
   - Logic gate decision boundaries (`.pdf` files)
   - Evolution of weights & biases (`.eps` files)
   - Training error vs. Epoch curve (`.eps` files)
   - Banknote classification metrics (Confusion Matrix, scatter plots, etc.)

### 📓 Lab 2: Multi Layer Perceptron (MLP)

1. **Verify TensorFlow Installation:** Ensure `tensorflow` package is correctly installed in your selected environment/kernel.
2. **Open Notebook:** Open [MLP.ipynb](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%202%20Multi%20Layer%20Perceptron/MLP.ipynb).
3. **Run Notebook:** Select the Python kernel/environment containing TensorFlow and run all cells sequentially.
4. **Output Visualizations:** The execution will produce:
   - XOR gate representative decision boundaries (`.pdf` file)
   - Sample images and class distribution (`.eps` files)
   - Hyperparameter search results (Randomized Search)
   - Training vs. Validation Accuracy & Loss curves (`.eps` files)
   - Confusion matrix of the final model on Fashion MNIST dataset


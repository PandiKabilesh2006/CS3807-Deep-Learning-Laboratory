# Lab 1: Single Layer Perceptron (Neural Network From Scratch)

This directory contains the implementation of a basic neural network architecture from scratch using Python and NumPy, applied to the banknote authentication problem.

---

## 🎯 Objectives
*   Implement a neural network (Single Layer Perceptron / Multi-Layer Perceptron) architecture from scratch.
*   Implement feedforward propagation, activation functions (e.g., Sigmoid), loss computation, backpropagation, and gradient descent optimization without using high-level deep learning libraries like TensorFlow or PyTorch.
*   Train and evaluate the model on the Banknote Authentication Dataset.

---

## 📊 Dataset: Banknote Authentication
The dataset [data_banknote_authentication.txt](file:///e:/CS2807-Deep%20Learning%20Laboratory/Single%20Layer%20Perceptron/data_banknote_authentication.txt) contains 1,372 instances with 4 continuous features and 1 binary class label:

1.  **Variance**: Variance of the Wavelet Transformed image.
2.  **Skewness**: Skewness of the Wavelet Transformed image.
3.  **Curtosis**: Curtosis of the Wavelet Transformed image.
4.  **Entropy**: Entropy of the image.
5.  **Class (Target)**: `0` for genuine banknotes, `1` for forged banknotes.

---

## 📂 Directory Contents
*   [NN_From_Scratch (1).ipynb](file:///e:/CS2807-Deep%20Learning%20Laboratory/Single%20Layer%20Perceptron/NN_From_Scratch%20%281%29.ipynb): The Jupyter notebook containing the step-by-step neural network implementation, training loop, and evaluation.
*   [data_banknote_authentication.txt](file:///e:/CS2807-Deep%20Learning%20Laboratory/Single%20Layer%20Perceptron/data_banknote_authentication.txt): The raw dataset file.

---

## 🚀 How to Run
1. Open the [NN_From_Scratch (1).ipynb](file:///e:/CS2807-Deep%20Learning%20Laboratory/Single%20Layer%20Perceptron/NN_From_Scratch%20%281%29.ipynb) notebook in VS Code or Jupyter Notebook.
2. Select your Python kernel.
3. Run the cells sequentially to load the dataset, build the model, and observe the training process.

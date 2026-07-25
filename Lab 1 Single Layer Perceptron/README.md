# 🧠 Single Layer Perceptron from Scratch

> Implementation of a Single Layer Perceptron from scratch using NumPy for binary classification and logic gate learning.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Enabled-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 📖 Overview

This repository contains a complete implementation of a **Single Layer Perceptron** developed entirely from scratch without using any machine learning libraries.

The project demonstrates:

- Artificial Neuron
- Step Activation Function
- Perceptron Learning Algorithm
- Binary Classification
- Weight & Bias Updates
- Decision Boundary Visualization
- Performance Evaluation
- Learning Rate Analysis
- Logic Gate Implementation
- XOR Failure Analysis

---

# ✨ Features

✅ Perceptron implemented completely from scratch

✅ Forward propagation

✅ Step activation function

✅ Perceptron learning rule

✅ Weight initialization

✅ Bias initialization

✅ Epoch-wise training

✅ Decision boundary visualization

✅ Weight evolution plot

✅ Bias evolution plot

✅ Learning rate comparison

✅ Confusion matrix

✅ Logic gate implementation

---

# 📂 Repository Structure

```
Lab 1 Single Layer Perceptron/
│
├── NN_From_Scratch.ipynb
├── README.md
│
├── data_banknote_authentication.txt
│
├── AND_Gate_Decision_Boundaries.pdf
├── OR_Gate_Decision_Boundaries.pdf
├── NOT_Gate_Decision_Thresholds.pdf
├── Decision_Boundary.pdf
├── Decision_Boundary.png
│
├── histogram.eps
├── scatterplot.eps
├── boxplot.eps
├── corr_matrix.eps
├── Training_Error_Vs_Epoch.eps
├── Weight_Evolution.eps
├── Bias_Evolution.eps
├── Learning_Rate_Comparison.eps
├── Confusion_Matrix.eps
```

---

# 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib

---

# 🧮 Perceptron Learning Algorithm

For each training sample,

### Weighted Sum

\[
z=w^Tx+b
\]

### Step Activation

\[
\hat{y}=
\begin{cases}
1,&z\ge0\\
0,&z<0
\end{cases}
\]

### Update Rule

\[
w=w+\eta(y-\hat y)x
\]

\[
b=b+\eta(y-\hat y)
\]

---

# 📊 Visualizations

The notebook automatically generates:

- Feature Histograms
- Scatter Plot
- Correlation Matrix
- Boxplots
- Training Error vs Epoch
- Weight Evolution
- Bias Evolution
- Learning Rate Comparison
- Confusion Matrix
- Decision Boundary

---

# 🔌 Logic Gates

## AND Gate

✔ Successfully Learned

✔ Linearly Separable

✔ Correct Decision Boundary

---

## OR Gate

✔ Successfully Learned

✔ Linearly Separable

✔ Correct Decision Boundary

---

## NOT Gate

✔ Successfully Learned

✔ Correct Threshold

---

## XOR Gate

❌ Failed to Converge

The perceptron repeatedly updated its weights but never converged because the XOR dataset is **not linearly separable**. The decision boundary continuously oscillated throughout training, demonstrating the limitation of a single-layer perceptron.

---

# 📈 Results

The model successfully classified the **Banknote Authentication Dataset** and generated the following evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The project also visualizes how the weights, bias, and training error evolve during learning.

---

# 🚀 How to Run

Clone the repository

```bash
git clone https://github.com/PandiKabilesh2006/CS3807-Deep-Learning-Laboratory.git
```

Navigate to the project

```bash
cd "Lab 1 Single Layer Perceptron"
```

Install dependencies

```bash
pip install numpy matplotlib jupyter
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```
NN_From_Scratch.ipynb
```

Run all cells.

---

# 📚 Learning Outcomes

This project helped understand

- Artificial Neurons
- Linear Classification
- Binary Classification
- Decision Boundaries
- Weight Updates
- Learning Rule
- Linear Separability
- Importance of Bias
- XOR Limitation
- Foundation of Neural Networks

---

# 📌 Future Improvements

- Multi-Layer Perceptron (MLP)
- Backpropagation
- Sigmoid Activation
- ReLU Activation
- Gradient Descent
- Mini-Batch Learning
- MNIST Classification

---

# 📖 References

- Frank Rosenblatt, *The Perceptron (1958)*
- Deep Learning — Goodfellow, Bengio & Courville
- Pattern Recognition and Machine Learning — Christopher Bishop
- UCI Machine Learning Repository

---

# 👨‍💻 Author

**Pandi Kabilesh**

B.Tech Artificial Intelligence & Data Science

Shiv Nadar University Chennai

GitHub: https://github.com/PandiKabilesh2006

⭐ If you found this repository helpful, consider giving it a star!
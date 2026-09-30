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
| **3** | 2026-08-15 | **CNN for Image Classification**<br>• Custom CNN from scratch in TensorFlow/Keras.<br>• Multi-class classification on the CIFAR-10 dataset.<br>• Visualizing intermediate layer feature maps.<br>• Comparative study of Max Pooling vs. Average Pooling architectures. | [Lab 3 Directory](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%203%20CNN%20for%20Image%20Classification/)<br>[Notebook](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%203%20CNN%20for%20Image%20Classification/cnn_for_cifar_10.ipynb) | 🟢 **Completed** |
| **4** | 2026-08-20 | **Transfer Learning with VGG16**<br>• Image classification on upscaled CIFAR-10 (96x96x3).<br>• Comparison of frozen Feature Extraction base vs. selective Fine-Tuning of block5 layers.<br>• Hyperparameter training and classification reports (Precision, Recall, F1). | [Lab 4 Directory](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%204%20Comparative%20Study%20of%20Deep%20Convolutional%20Neural%20Network%20Architectures%20Using%20Transfer%20Learning/)<br>[Notebook](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%204%20Comparative%20Study%20of%20Deep%20Convolutional%20Neural%20Network%20Architectures%20Using%20Transfer%20Learning/24011101075_DL_Lab_Exercise_4.ipynb) | 🟢 **Completed** |
| **5** | 2026-08-27 | **Comprehensive CNN Training & Optimization**<br>• Fine-grained multi-class classification on Oxford-IIIT Pet (37 breeds).<br>• Systematic analysis across 10 sub-experiments: Initializers, regularizers (L2, Dropout), optimizers, learning rates, batch sizes, feature extraction vs. fine-tuning.<br>• 5-Fold Cross-Validation on top configurations to select the best deployed model. | [Lab 5 Directory](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%205%20Comprehensive%20Study%20of%20CNN%20Training,%20Regularization,%20Optimization,%20Hyperparameter%20Tuning,%20Transfer%20Learning%20and%20Cross-Validation/)<br>[Notebook](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%205%20Comprehensive%20Study%20of%20CNN%20Training,%20Regularization,%20Optimization,%20Hyperparameter%20Tuning,%20Transfer%20Learning%20and%20Cross-Validation/24011101075_DL_Lab_Exercise_5.ipynb) | 🟢 **Completed** |
| **7** | 2026-09-17 | **End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders**<br>• End-to-end comparative study of Fully Connected Autoencoder (FC-AE), Convolutional Autoencoder (CAE), Denoising CAE, and Variational Autoencoder (VAE) on MNIST.<br>• Quantitative evaluation of reconstruction fidelity using MSE, MAE, and SSIM.<br>• Noise robustness analysis under Gaussian noise ($\sigma \in \{0.1, 0.2, 0.3\}$) and Salt-and-Pepper corruption.<br>• 2D latent space manifold visualization, random synthetic digit generation ($z \sim \mathcal{N}(0, I)$), linear latent interpolation, and bottleneck capacity ablation ($d_z \in \{2, 8, 16, 32\}$). | [Lab 7 Directory](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%207%20End-to-End%20Study%20of%20Autoencoders,%20Convolutional/)<br>[Report PDF](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%207%20End-to-End%20Study%20of%20Autoencoders,%20Convolutional/24011101075_DL_Lab_Exercise_7_compressed.pdf)<br>[README](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%207%20End-to-End%20Study%20of%20Autoencoders,%20Convolutional/README.md) | 🟢 **Completed** |

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
For deep learning labs, make sure to also install TensorFlow:
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

### 📓 Lab 3: CNN for Image Classification

1. **Verify Dataset:** CIFAR-10 dataset will be automatically downloaded by TensorFlow when running the notebook.
2. **Open Notebook:** Open [cnn_for_cifar_10.ipynb](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%203%20CNN%20for%20Image%20Classification/cnn_for_cifar_10.ipynb).
3. **Run Notebook:** Select the Python kernel/environment containing TensorFlow and run all cells sequentially.
4. **Output Visualizations:** The notebook generates:
   - Sample images and class distribution (`.pdf` files)
   - Training/Validation Accuracy & Loss curves (`.pdf` files)
   - Intermediate layer feature maps (`Feature_Map.pdf`)
   - Max vs Average Pooling comparison (`Max_vs_Avg_Pooling.pdf`)
   - Confusion Matrix and test predictions (`.pdf` files)

### 📓 Lab 4: Transfer Learning with VGG16

1. **Verify Dataset:** CIFAR-10 is automatically loaded and preprocessed.
2. **Open Notebook:** Open [24011101075_DL_Lab_Exercise_4.ipynb](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%204%20Comparative%20Study%20of%20Deep%20Convolutional%20Neural%20Network%20Architectures%20Using%20Transfer%20Learning/24011101075_DL_Lab_Exercise_4.ipynb).
3. **Run Notebook:** Select the Python kernel/environment containing TensorFlow and run all cells sequentially.
4. **Output Visualizations:** The notebook generates:
   - Initial sample images (`01_sample_images.pdf`)
   - Feature Extraction curves (`02_initial_accuracy.pdf`, `02_initial_loss.pdf`)
   - Fine-Tuning curves (`03_finetune_accuracy.pdf`, `03_finetune_loss.pdf`)
   - Heatmap Confusion Matrix (`04_confusion_matrix.pdf`)
   - Grid of misclassified samples (`05_misclassified.pdf`)

### 📓 Lab 5: Comprehensive CNN Optimization & Cross-Validation

1. **Dataset Location:** Ensure the Oxford-IIIT Pet Dataset is available (it will be downloaded automatically via `tensorflow_datasets` or similar package if configured).
2. **Open Notebook:** Open [24011101075_DL_Lab_Exercise_5.ipynb](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%205%20Comprehensive%20Study%20of%20CNN%20Training,%20Regularization,%20Optimization,%20Hyperparameter%20Tuning,%20Transfer%20Learning%20and%20Cross-Validation/24011101075_DL_Lab_Exercise_5.ipynb).
3. **Run Notebook:** Select the Python kernel/environment containing TensorFlow (GPU environment recommended) and run all cells sequentially.
4. **Output Visualizations:** Generates 20 publication-quality comparison charts:
   - Weight Initialization (`weight_initialization_comparison.pdf`)
   - Regularizers (`regularization_comparison.pdf`, `dropout_comparison.pdf`)
   - Optimizers and Learning Rates comparisons (`.pdf` files)
   - Feature Extraction vs. Fine-Tuning comparisons (`.pdf` files)
   - 5-Fold Cross-Validation comparisons (`cross_validation_comparison.pdf`)
   - Final classification report, confusion matrix heatmap, and misclassified examples (`.pdf` files)

### 📓 Lab 7: Autoencoders, Denoising Autoencoders and Variational Autoencoders (VAEs)

1. **Dataset Location:** The MNIST dataset is automatically downloaded and loaded via `tensorflow.keras.datasets.mnist`.
2. **Lab Report & TeX Source:** Review [24011101075_DL_Lab_Exercise_7_compressed.pdf](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%207%20End-to-End%20Study%20of%20Autoencoders,%20Convolutional/24011101075_DL_Lab_Exercise_7_compressed.pdf) or compile [Experiment_7.tex](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%207%20End-to-End%20Study%20of%20Autoencoders,%20Convolutional/Experiment_7.tex). Detailed analyses and architecture breakdowns are provided in the [Lab 7 README](file:///e:/CS3807-Deep%20Learning%20Laboratory/Lab%207%20End-to-End%20Study%20of%20Autoencoders,%20Convolutional/README.md).
3. **Execution Pipeline:**
   - Normalize $28 \times 28 \times 1$ grayscale images to $[0, 1]$.
   - Train Fully Connected Autoencoder ($784 \to 128 \to 32 \to 16 \to 32 \to 128 \to 784$).
   - Train Convolutional Autoencoder (Conv2D + MaxPool / UpSampling2D) preserving 2D spatial locality.
   - Train Denoising CAE with Gaussian corruption ($\sigma=0.2$) targeting clean uncorrupted targets.
   - Train Variational Autoencoder with reparameterization trick ($z = \mu + \sigma \odot \epsilon$) and joint reconstruction + KL divergence objective.
   - Run latent bottleneck capacity study ($d_z \in \{2, 8, 16, 32\}$).
4. **Output Visualizations:** Generates 23 publication-quality comparison charts:
   - Sample images and FC-AE reconstructions (`01_mnist_samples.pdf`, `02_fc_ae_original_vs_reconstructed.pdf`, `03_fc_ae_training_validation_loss.pdf`)
   - CAE reconstructions and FC-AE vs. CAE comparison (`04_cae_reconstruction.pdf`, `05_fc_ae_vs_cae_reconstruction.pdf`)
   - Denoising progression and noise sensitivity curves (`06_clean_noisy_denoised.pdf`, `07_denoising_cae_loss.pdf`, `08_noise_level_vs_mse.pdf`, `09_noise_level_vs_mae.pdf`, `10_noise_level_vs_ssim.pdf`)
   - VAE losses (Reconstruction, KL, Total) and reconstructions (`11_vae_reconstruction_loss.pdf`, `12_vae_kl_loss.pdf`, `13_vae_total_loss.pdf`, `14_vae_original_vs_reconstructed.pdf`)
   - 2D latent space cluster manifold, 25 generated sample digits, and smooth latent interpolation (`15_vae_latent_space.pdf`, `16_vae_generated_25.pdf`, `17_vae_interpolation.pdf`)
   - Per-image reconstruction error distributions and top 5 high-error sample inspections (`18_vae_reconstruction_error.pdf`, `19_vae_top5_high_error.pdf`, `20_combined_reconstruction_error.pdf`, `21_fc_top5_high_error.pdf`)
   - Latent dimension sensitivity curves (`22_latent_dimension_vs_mse.pdf`, `23_latent_dimension_vs_ssim.pdf`)


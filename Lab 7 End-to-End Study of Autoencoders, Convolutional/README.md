# End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders, and Variational Autoencoders

> Deep Learning Laboratory – Experiment 7  
> Shiv Nadar University Chennai  
> Course: CS3807 – Deep Learning Laboratory  
> Branch: B.Tech Artificial Intelligence & Data Science (Semester V, AY: 2026–27)

---

## Overview

This project presents an end-to-end comparative study of **Autoencoders (AEs)** and their deep variants for unsupervised image representation, spatial feature reconstruction, noise suppression (denoising), and generative modeling using the **MNIST** benchmark dataset.

The investigation progresses systematically across four foundational architectures:
1. **Fully Connected Autoencoder (FC-AE):** Dimensionality reduction and bottleneck reconstruction using dense multi-layer perceptron networks.
2. **Convolutional Autoencoder (CAE):** Exploiting 2D convolutional feature hierarchies and spatial locality with upsampling.
3. **Denoising Convolutional Autoencoder (Denoising CAE):** Reconstructing clean ground-truth images from corrupted inputs under varying noise levels.
4. **Variational Autoencoder (VAE):** Enforcing a continuous, regularized probabilistic latent distribution using the reparameterization trick for synthetic image generation and manifold interpolation.

Additionally, extensive ablations evaluate the **latent bottleneck dimension** ($d_z \in \{2, 8, 16, 32\}$), evaluate **noise sensitivity** ($\sigma \in \{0.1, 0.2, 0.3\}$), map the **2D latent manifold geometry**, and analyze the **per-image reconstruction error distribution** to identify outlier failure modes.

---

## Dataset & Experimental Protocol

### MNIST Handwritten Digit Dataset
- **Image Dimensions:** $28 \times 28 \times 1$ (Grayscale)
- **Pixel Intensity Rescaling:** Normalization from $[0, 255]$ to $[0.0, 1.0]$ via min-max scaling.
- **Experimental Split:**
  - **Laboratory Training Subset:** 10,000 images (9,000 training, 1,000 validation).
  - **Held-Out Test Set:** 2,000 images reserved exclusively for final evaluation.
- **Unsupervised Paradigm:** Ground-truth labels are excluded during optimization. The target is the original uncorrupted image:
  $$\boxed{x \longrightarrow \text{Encoder} \longrightarrow z \longrightarrow \text{Decoder} \longrightarrow \hat{x}}$$

---

## Model Architectures & Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                EXPERIMENTAL PIPELINE                                   │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [Input Image x] (28x28x1)
        │
        ├──> [Study 1: Flatten to 784] ──> Dense(128) ──> Dense(32) ──> Latent z(16)
        │                                                                     │
        │                                  Output x̂ (784) <── Dense(128) <── Dense(32)
        │
        ├──> [Study 2: Spatial (28x28x1)] ──> Conv(32) ──> MaxPool ──> Conv(64) ──> MaxPool
        │                                                                               │
        │                                  Output x̂ (28x28x1) <── UpSample <── Conv(64)
        │
        ├──> [Study 3: Gaussian Noise] ──> Noisy x̃ (σ=0.2) ──> Denoising CAE ──> Clean x̂
        │
        └──> [Study 4: Probabilistic] ──> Conv Base ──> μ, log σ² ──> z = μ + σ ⊙ ε
                                                                           │
                                           Reconstruction / Generation <── ConvTranspose
```

### Detailed Network Specifications

| Parameter / Layer | Fully Connected AE | Convolutional AE | Denoising CAE | Variational AE (VAE) |
| :--- | :--- | :--- | :--- | :--- |
| **Input Shape** | $784$ (flattened) | $28 \times 28 \times 1$ | $28 \times 28 \times 1$ | $28 \times 28 \times 1$ |
| **Encoder Architecture** | Dense(128, ReLU)<br>Dense(32, ReLU) | Conv2D(32, 3x3, ReLU)<br>MaxPool2D(2x2)<br>Conv2D(64, 3x3, ReLU)<br>MaxPool2D(2x2) | Conv2D(32, 3x3, ReLU)<br>MaxPool2D(2x2)<br>Conv2D(64, 3x3, ReLU)<br>MaxPool2D(2x2) | Conv2D(32, 3x3, ReLU, s=2)<br>Conv2D(64, 3x3, ReLU, s=2)<br>Flatten $\to$ Dense(16, ReLU) |
| **Latent Space ($z$)** | Dense($16$, Linear) | Conv2D($64$, $3 \times 3$, same) | Conv2D($64$, $3 \times 3$, same) | $\mu \in \mathbb{R}^2$, $\log \sigma^2 \in \mathbb{R}^2$<br>Sample $z \in \mathbb{R}^2$ via Reparameterization |
| **Decoder Architecture** | Dense(32, ReLU)<br>Dense(128, ReLU) | Conv2D(64, 3x3, ReLU)<br>UpSampling2D(2x2)<br>Conv2D(32, 3x3, ReLU)<br>UpSampling2D(2x2) | Conv2D(64, 3x3, ReLU)<br>UpSampling2D(2x2)<br>Conv2D(32, 3x3, ReLU)<br>UpSampling2D(2x2) | Dense($7 \times 7 \times 64$, ReLU)<br>Conv2DTranspose(64, 3x3, s=2)<br>Conv2DTranspose(32, 3x3, s=2) |
| **Output Layer** | Dense(784, Sigmoid) | Conv2D(1, 3x3, Sigmoid) | Conv2D(1, 3x3, Sigmoid) | Conv2D(1, 3x3, Sigmoid) |
| **Total Parameters** | **211,040** | **74,497** | **74,497** | **485,957** |
| **Loss Function** | Binary Cross-Entropy | Binary Cross-Entropy | Binary Cross-Entropy | Binary Cross-Entropy + KL Divergence |
| **Optimizer & LR** | Adam ($\eta = 10^{-3}$) | Adam ($\eta = 10^{-3}$) | Adam ($\eta = 10^{-3}$) | Adam ($\eta = 10^{-3}$) |
| **Batch Size & Epochs**| $128$ / $20$ epochs | $128$ / $20$ epochs | $128$ / $20$ epochs | $128$ / $20$ epochs |

---

## Mathematical Formulation

### 1. Reconstruction Objective
For input $x_i$ and reconstructed image $\hat{x}_i \in [0, 1]^D$:
- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \| x_i - \hat{x}_i \|_2^2$$
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \| x_i - \hat{x}_i \|_1$$
- **Structural Similarity Index Measure (SSIM):**
  $$\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + C_1)(2\sigma_{x\hat{x}} + C_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + C_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + C_2)}$$

### 2. Denoising Mapping
Corrupted inputs $\tilde{x} = x + n$, with additive white Gaussian noise $n \sim \mathcal{N}(0, \sigma^2)$, are mapped back to uncorrupted targets $x$:
$$\mathcal{L}_{\text{Denoise}} = \mathbb{E}_{x \sim p(x), n \sim \mathcal{N}(0, \sigma^2)} \left[ \mathcal{L}_{\text{BCE}}(x, g_\phi(f_\theta(\tilde{x}))) \right]$$

### 3. Variational Autoencoder Loss & Reparameterization
- **Reparameterization Trick:** Enables backpropagation through stochastic sampling:
  $$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
- **Total VAE Loss ($\mathcal{L}_{\text{VAE}}$):**
  $$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{rec}} + D_{\text{KL}}(q_\phi(z|x) \,\|\, p(z))$$
- **Analytical KL Divergence against Standard Normal Prior $\mathcal{N}(0, I)$:**
  $$D_{\text{KL}}(q_\phi(z|x) \,\|\, p(z)) = -\frac{1}{2} \sum_{j=1}^{d_z} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

---

## Experimental Results & Comparison

### Consolidated Model Performance on Held-Out Test Set (2,000 Images)

| Model Architecture | Test MSE | Test MAE | Mean SSIM | Total Parameters | Training Time | Convergence Trajectory |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Fully Connected AE (FC-AE)** | $0.022110$ | $0.058131$ | $0.737771$ | $211,040$ | $27.21\text{ s}$ | Loss: $0.3559 \to 0.1236$ (Train), $0.2523 \to 0.1303$ (Val) |
| **Convolutional AE (CAE)** | **0.002616** | **0.015094** | **0.973534** | **74,497** | $685.09\text{ s}$ | Final Train Loss: $0.0684$, Val Loss: $0.0695$ |
| **Denoising CAE ($\sigma=0.2$)** | $0.004969$ | $0.022047$ | $0.941478$ | **74,497** | $672.30\text{ s}$ | Final Train Loss: $0.0759$, Val Loss: $0.0783$ |
| **Variational AE (VAE, 2D latent)**| $0.043799$ | $0.102828$ | $0.485724$ | $485,957$ | $356.05\text{ s}$ | Recon Loss: $148.99$, KL Loss: $5.73$, Total: $154.72$ |

### VAE Detailed Loss Components

| Metric Component | Held-Out Test Value | Description |
| :--- | :---: | :--- |
| **Reconstruction Loss ($\mathcal{L}_{\text{rec}}$)** | $148.990936$ | Unreduced binary cross-entropy across all $784$ pixels |
| **Kullback-Leibler Divergence ($\mathcal{L}_{\text{KL}}$)** | $5.725443$ | Latent distribution penalty regularizing $q(z\|x)$ toward $\mathcal{N}(0, I)$ |
| **Total VAE Loss ($\mathcal{L}_{\text{VAE}}$)** | **154.716400** | Joint objective balancing fidelity against prior conformity |

---

## Detailed Study Analyses

### 1. FC-AE vs. Convolutional AE: Impact of Spatial Inductive Bias
- **Spatial Topology Preservation:** While the FC-AE flattens 2D matrices into 1D vectors—destroying pixel neighborhood relationships—the CAE applies translation-equivariant kernels that preserve spatial locality.
- **Quantitative Superiority:** The CAE achieves an **88.2% reduction in MSE** ($0.002616$ vs. $0.022110$) and boosts structural similarity (**SSIM: $0.9735$ vs. $0.7378$**) while requiring **64.7% fewer trainable parameters** ($74,497$ vs. $211,040$).

### 2. Denoising Robustness Under Varying Noise Corruptions
Evaluating the Denoising CAE on images corrupted with zero-mean Gaussian noise across multiple standard deviations $\sigma$:

| Gaussian Noise Level ($\sigma$) | Test MSE | Test MAE | Mean SSIM | Reconstruction Integrity |
| :---: | :---: | :---: | :---: | :--- |
| $\sigma = 0.1$ | $0.003855$ | $0.018465$ | **0.954884** | High stroke fidelity; minor edge softness |
| $\sigma = 0.2$ (Training level) | $0.005016$ | $0.022132$ | **0.940780** | Complete background noise suppression, sharp boundaries |
| $\sigma = 0.3$ | $0.007599$ | $0.029860$ | **0.878881** | Major digit topology preserved; slight stroke thinning |

*Inference:* Even when noise power is tripled, SSIM remains above $0.878$, verifying that the convolutional manifold projection filters out high-frequency stochastic noise while preserving semantic digit manifolds.

### 3. Latent Bottleneck Capacity Study ($d_z \in \{2, 8, 16, 32\}$)
Evaluating the compression-fidelity trade-off in the basic autoencoder:

| Latent Dimension ($d_z$) | Test MSE | Mean SSIM | Trainable Parameters | Compression Ratio |
| :---: | :---: | :---: | :---: | :---: |
| **2** | $0.047606$ | $0.431449$ | $210,130$ | $392 : 1$ |
| **8** | $0.030645$ | $0.645589$ | $210,520$ | $98 : 1$ |
| **16** | $0.024599$ | $0.708026$ | $211,040$ | $49 : 1$ |
| **32** | **0.021787** | **0.739935** | $212,080$ | $24.5 : 1$ |

*Inference:* The steepest marginal gain in reconstruction fidelity occurs between $d_z = 2$ and $d_z = 8$ ($\Delta \text{SSIM} = +0.214$). Beyond $d_z = 16$, returns diminish as the model captures progressively finer idiosyncratic nuances rather than primary structural modes.

### 4. VAE Latent Geometry, Generation & Interpolation
- **2D Latent Space Manifold:** Visualizing the 2D latent coordinates ($z_1, z_2$) reveals class-specific clustering where visually similar digits (e.g., $3, 5, 8$ or $4, 9, 7$) reside in adjacent neighborhoods with continuous boundaries.
- **Synthetic Sample Generation:** Sampling 25 random vectors $z \sim \mathcal{N}(0, I)$ directly through the decoder synthesizes realistic, novel handwritten digits without class condition inputs.
- **Linear Latent Interpolation:** Decoded traversals along $z(\alpha) = (1-\alpha)z_A + \alpha z_B$ for $\alpha \in [0, 1]$ demonstrate smooth topological morphing (e.g., smoothly closing a loop to transform digit $7$ into digit $0$) without abrupt artifacts.

---

## Reconstruction Error Distribution & Outlier Diagnostics

Computing the per-image squared reconstruction error $e_i = \frac{1}{D} \sum_{j=1}^D (x_{ij} - \hat{x}_{ij})^2$ highlights the difficulty spectrum:

```
Reconstruction Error Frequency Distribution:
  CAE: [Peak at 0.001 - 0.003] ──> Heavy right-skew tail (Concentrated precision)
  VAE: [Peak at 0.035 - 0.050] ──> Broader variance due to Gaussian prior regularization
```

### Top 5 High-Error Outlier Analysis

| Model | Rank | Test Sample Index | Digit Class | Per-Image MSE | Diagnostic Cause |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FC-AE** | 1 | 1782 | 8 | $0.071842$ | Asymmetric double-loop with unusually heavy stroke ink |
| | 2 | 654 | 5 | $0.067009$ | Highly slanted stroke and disconnected top horizontal bar |
| | 3 | 1801 | 9 | $0.064957$ | Narrow loop resembling a vertical bar |
| | 4 | 876 | 8 | $0.064668$ | Off-center, irregular aspect ratio |
| | 5 | 338 | 8 | $0.064561$ | Fragmented bottom loop with unusual curvature |
| **VAE** | 1 | 1352 | 2 | $0.122466$ | Unconventional looped base with sharp angular stroke |
| | 2 | 1325 | 8 | $0.117107$ | Extremely thick, overlapping brush contours |
| | 3 | 1847 | 5 | $0.114303$ | Compressed vertical ascender |
| | 4 | 1790 | 2 | $0.112336$ | Straight-line base without curl |
| | 5 | 8 | 5 | $0.103874$ | Wide loop and atypical stroke width |

*Inference:* Both models fail predominantly on digits featuring atypical stroke thickness, fragmented loops, or unusual slant angles that deviate from the central distribution modes.

---

## Output Visualizations

The experiment generates **23 publication-quality PDF plots** (600 DPI):

| Figure File | Description | Key Insight |
| :--- | :--- | :--- |
| `01_mnist_samples.pdf` | Sample MNIST images from the dataset | Visualizes representative raw inputs across classes 0–9 |
| `02_fc_ae_original_vs_reconstructed.pdf` | Original vs. FC-AE reconstructed digits | Captures general digit shapes but exhibits blurred edges |
| `03_fc_ae_training_validation_loss.pdf` | FC-AE training and validation loss curves | Demonstrates smooth convergence without overfitting |
| `04_cae_reconstruction.pdf` | Original vs. CAE reconstructed digits | Sharp stroke edges and excellent structural restoration |
| `05_fc_ae_vs_cae_reconstruction.pdf` | Direct side-by-side comparison: Original vs. FC-AE vs. CAE | Highlights spatial convolution superiority over flattening |
| `06_clean_noisy_denoised.pdf` | Clean, noisy ($\sigma=0.2$), and denoised digits | Verifies noise suppression while preserving digit strokes |
| `07_denoising_cae_loss.pdf` | Denoising CAE training and validation loss curves | Confirms stable optimization under corrupted inputs |
| `08_noise_level_vs_mse.pdf` | Gaussian noise level ($\sigma$) vs. Reconstruction MSE | Quantifies monotonic error increase as corruption grows |
| `09_noise_level_vs_mae.pdf` | Gaussian noise level ($\sigma$) vs. Reconstruction MAE | Tracks mean absolute deviations across noise levels |
| `10_noise_level_vs_ssim.pdf` | Gaussian noise level ($\sigma$) vs. Reconstruction SSIM | Evaluates structural degradation from $\sigma=0.1$ to $0.3$ |
| `11_vae_reconstruction_loss.pdf` | VAE reconstruction loss curve over epochs | Captures primary reconstruction learning phase |
| `12_vae_kl_loss.pdf` | VAE KL divergence loss trajectory | Tracks latent manifold regularization toward $\mathcal{N}(0, I)$ |
| `13_vae_total_loss.pdf` | VAE total combined loss curve | Shows balanced optimization of fidelity and prior penalty |
| `14_vae_original_vs_reconstructed.pdf` | Original vs. VAE reconstructed test digits | Illustrates smooth, continuous reconstructions |
| `15_vae_latent_space.pdf` | 2D VAE latent space colored by digit class | Reveals distinct class clusters and continuous boundaries |
| `16_vae_generated_25.pdf` | Grid of 25 novel digits generated by sampling $z \sim \mathcal{N}(0, I)$ | Proves generative capability of learned latent space |
| `17_vae_interpolation.pdf` | Sequential latent space linear interpolation ($\alpha=0 \to 1$) | Demonstrates smooth topological digit transformation |
| `18_vae_reconstruction_error.pdf` | Histogram of per-image reconstruction error for VAE | Characterizes error distribution and variance |
| `19_vae_top5_high_error.pdf` | 5 worst-reconstructed test images by VAE | Identifies high-difficulty structural outliers |
| `20_combined_reconstruction_error.pdf` | Comparative reconstruction error histograms for all models | Shows CAE distribution tightly centered near zero |
| `21_fc_top5_high_error.pdf` | 5 worst-reconstructed test images by FC-AE | Diagnoses failure modes in dense bottleneck models |
| `22_latent_dimension_vs_mse.pdf` | Latent dimension ($d_z \in \{2, 8, 16, 32\}$) vs. MSE | Highlights compression-fidelity trade-off curve |
| `23_latent_dimension_vs_ssim.pdf` | Latent dimension ($d_z \in \{2, 8, 16, 32\}$) vs. SSIM | Identifies optimal bottleneck capacity threshold |

---

## Repository Structure

```
Lab 7 End-to-End Study of Autoencoders, Convolutional/
├── 24011101075_DL_Lab_Exercise_7.pdf             # Complete academic lab report (17.4 MB)
├── 24011101075_DL_Lab_Exercise_7_compressed.pdf  # Compressed lab report (656 KB)
├── Experiment_7.tex                             # Comprehensive LaTeX source code
├── README.md                                    # Experiment documentation and results analysis
├── 01_mnist_samples.pdf
├── 02_fc_ae_original_vs_reconstructed.pdf
├── 03_fc_ae_training_validation_loss.pdf
├── 04_cae_reconstruction.pdf
├── 05_fc_ae_vs_cae_reconstruction.pdf
├── 06_clean_noisy_denoised.pdf
├── 07_denoising_cae_loss.pdf
├── 08_noise_level_vs_mse.pdf
├── 09_noise_level_vs_mae.pdf
├── 10_noise_level_vs_ssim.pdf
├── 11_vae_reconstruction_loss.pdf
├── 12_vae_kl_loss.pdf
├── 13_vae_total_loss.pdf
├── 14_vae_original_vs_reconstructed.pdf
├── 15_vae_latent_space.pdf
├── 16_vae_generated_25.pdf
├── 17_vae_interpolation.pdf
├── 18_vae_reconstruction_error.pdf
├── 19_vae_top5_high_error.pdf
├── 20_combined_reconstruction_error.pdf
├── 21_fc_top5_high_error.pdf
├── 22_latent_dimension_vs_mse.pdf
└── 23_latent_dimension_vs_ssim.pdf
```

---

## Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras:** Model definition, custom VAE loss computation, reparameterization trick, training workflows.
- **NumPy:** Matrix operations, noise generation (Gaussian and Salt-and-Pepper), latent interpolation.
- **Scikit-Image:** SSIM structural similarity metric calculation.
- **Matplotlib & Seaborn:** Publication-quality visualizations (600 DPI).
- **LaTeX / TikZ:** Academic report compilation and architectural block diagrams.

---

## Learning Outcomes

- Formulated and trained encoder-bottleneck-decoder architectures for unsupervised image reconstruction.
- Demonstrated the critical importance of spatial inductive bias (convolution and transposed convolution) over fully connected layers for image data.
- Built noise-robust feature extractors capable of recovering clean signals from noisy inputs.
- Derived and implemented the Variational Autoencoder framework, combining reconstruction loss with analytical KL divergence.
- Applied the reparameterization trick to enable gradient-based optimization through stochastic latent nodes.
- Evaluated reconstruction fidelity using multi-dimensional quantitative metrics (MSE, MAE, SSIM) alongside qualitative error distribution histograms.

---

## References

1. Ian Goodfellow, Yoshua Bengio, Aaron Courville – *Deep Learning*, MIT Press, 2016.
2. Diederik P. Kingma, Max Welling – *Auto-Encoding Variational Bayes*, International Conference on Learning Representations (ICLR), 2014.
3. Pascal Vincent, Hugo Larochelle, Yoshua Bengio, Pierre-Antoine Manzagol – *Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion*, JMLR, 2010.
4. Yann LeCun, Corinna Cortes, Christopher J.C. Burges – *The MNIST Database of Handwritten Digits*.
5. Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, Eero P. Simoncelli – *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE Transactions on Image Processing, 2004.

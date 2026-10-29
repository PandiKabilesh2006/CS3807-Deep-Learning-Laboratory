# Effect of Attention Type and Attention Position in a Pretrained ResNet50

> Deep Learning Laboratory – Experiments 8 & 9  
> Shiv Nadar University Chennai  
> Department of Computer Science & Engineering  
> Course: CS3807 – Deep Learning Laboratory  
> Branch: B.Tech Artificial Intelligence & Data Science (Semester V, AY: 2026–27)  
> Student Registration: 24011101075  

---

## 📌 Executive Summary & Motivation

In deep convolutional neural networks (CNNs), standard convolutional filters treat all spatial locations and all channel feature maps uniformly. **Attention mechanisms** introduce dynamic, input-dependent reweighting functions that allow the network to selectively emphasize informative features and suppress irrelevant background clutter.

However, an attention mechanism is not an isolated mathematical block whose behavior is invariant to its host architecture. The effectiveness of attention is governed by five interdependent factors:

$$\boxed{\text{Attention Effectiveness} = f(\text{Attention Type}, \text{Attention Position}, \text{Feature Resolution}, \text{Feature Semantics}, \text{Model Complexity})}$$

This laboratory project conducts a systematic, controlled experimental investigation into:
1. **Which attention mechanism should be used?** (Channel / Squeeze-and-Excitation, Spatial, Sequential Channel–Spatial, Self-Attention, Combined, ECA, CBAM, and Coordinate Attention).
2. **Where should attention be inserted within the feature hierarchy?** Early-stage ($P_1$: high-resolution/low-level features), intermediate-stage ($P_2, P_3$: parts/textures), or late-stage ($P_4$: compact semantic representations).
3. **Does stacking attention across multiple stages help or hurt generalization?** Single-stage vs. multi-stage attention ($P_3, P_4, P_3+P_4, P_2+P_3+P_4$).
4. **Does sequential ordering matter?** Channel $\to$ Spatial vs. Spatial $\to$ Channel.
5. **How does receptive field affect spatial attention?** Kernel size scaling $k \in \{3, 5, 7\}$.
6. **Interpretability & Visualization:** Disentangling **internal forward-pass attention maps** from **post-hoc gradient attribution (Grad-CAM)**.

---

## 🔬 Dataset & Experimental Protocol

### 1. Dataset: TensorFlow Flowers (`tf_flowers`)
The benchmark evaluation is performed on the natural RGB flower dataset comprising **5 distinct classes**:
- `daisy`
- `dandelion`
- `roses`
- `sunflowers`
- `tulips`

The dataset exhibits realistic computer vision challenges: severe background clutter (foliage, grass, vases, outdoor scenes), scale variation, intra-class color differences, viewpoint shifts, and partial occlusions.

### 2. Preprocessing & Data Augmentation
- **Input Dimensions:** Resized to $224 \times 224 \times 3$.
- **Normalization:** ImageNet-standard zero-centering and scaling ($z = \frac{x - \mu}{\sigma}$).
- **Augmentation Pipeline:** Maintained strictly identical across all ablations to prevent split/augmentation bias:
  - Random horizontal flipping
  - Small affine rotations ($\pm 15^\circ$)
  - Random zoom ($\pm 10\%$)
  - Random translation ($\pm 10\%$)
  - Mild contrast jitter

### 3. Stratified Partitioning Protocol
To guarantee that empirical differences stem strictly from model architectural variations rather than data distribution shifts, a fixed random seed is enforced:
- **Training Partition (70%):** $\approx 2,569$ images ($442$ daisy, $629$ dandelion, $449$ roses, $490$ sunflowers, $559$ tulips)
- **Validation Partition (15%):** $\approx 550$ images
- **Held-Out Test Partition (15%):** $551$ images evaluated on all final metrics

---

## 🏛️ Backbone Architecture & ResNet50 Feature Hierarchy

All experiments utilize a **ResNet50** backbone pretrained on ImageNet-1k ($I \in \mathbb{R}^{224 \times 224 \times 3}$). As the depth increases, the spatial resolution contracts via strided convolutions while channel capacity expands:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RESNET50 STAGE-WISE HIERARCHY                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

 [Input: 224x224x3]
         │
    [Stem + MaxPool]  ───────> 56 × 56 × 64   (Low-level edges, pixel gradients)
         │
     [conv2_x Stage]  ───────> 56 × 56 × 256  ◄─── Position P1 (High spatial detail, colors, textures)
         │
     [conv3_x Stage]  ───────> 28 × 28 × 512  ◄─── Position P2 (Local patterns, petal arrangements)
         │
     [conv4_x Stage]  ───────> 14 × 14 × 1024 ◄─── Position P3 (Object parts, intermediate semantics)
         │
     [conv5_x Stage]  ───────> 7 × 7 × 2048   ◄─── Position P4 (High-level class-discriminative semantics)
         │
    [Global Avg Pool] ───────> 2048-D Vector
         │
     [Dropout (0.5)]
         │
    [Dense Softmax]   ───────> 5 Class Probabilities
```

### Stage Summary & Insertion Points:

| Stage Identifier | Output Tensor Shape ($H \times W \times C$) | Candidate Insertion Position | Dominant Feature Representation |
| :--- | :---: | :---: | :--- |
| **Stem + MaxPool** | $56 \times 56 \times 64$ | — | Raw edges, simple gradients, color boundaries |
| **conv2_x** | $56 \times 56 \times 256$ | **$P_1$** | Fine textures, high-resolution edge groupings |
| **conv3_x** | $28 \times 28 \times 512$ | **$P_2$** | Mid-level petal clusters, floral motifs, geometric corners |
| **conv4_x** | $14 \times 14 \times 1024$ | **$P_3$** (Primary) | Distinctive floral structures (stamens, corollas), balanced spatial-semantic |
| **conv5_x** | $7 \times 7 \times 2048$ | **$P_4$** | Global semantic concepts, class-separating feature manifolds |

---

## 📐 Mathematical Formulation of Attention Mechanisms

Given an intermediate convolutional feature tensor $F \in \mathbb{R}^{H \times W \times C}$:

### 1. Channel Attention (Squeeze-and-Excitation / SE)
Answers: *"What feature channels are most informative?"*
- **Squeeze Step (Global Average Pooling):**
  $$z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W F(i, j, c), \quad z \in \mathbb{R}^C$$
- **Excitation Step (Two-layer MLP Bottleneck with reduction ratio $r=16$):**
  $$M_c(F) = \sigma\left(W_2 \cdot \delta(W_1 z)\right)$$
  where $W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$, $W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$, $\delta = \text{ReLU}$, and $\sigma = \text{Sigmoid}$.
- **Feature Recalibration:**
  $$F_c = F \odot M_c(F)$$

---

### 2. Spatial Attention
Answers: *"Where are the informative regions located?"*
- **Channel-Wise Aggregation:** Computes mean and max projections along the channel axis:
  $$F_{\text{avg}} = \text{Mean}_c(F) \in \mathbb{R}^{H \times W \times 1}, \quad F_{\text{max}} = \text{Max}_c(F) \in \mathbb{R}^{H \times W \times 1}$$
  $$F_s = \left[ F_{\text{avg}} \,\|\, F_{\text{max}} \right] \in \mathbb{R}^{H \times W \times 2}$$
- **Convolutional Filter & Sigmoid Gating:**
  $$M_s(F) = \sigma\left( \text{Conv}_{k \times k}(F_s) \right) \in \mathbb{R}^{H \times W \times 1}, \quad k \in \{3, 5, 7\}$$
- **Spatial Modulation:**
  $$F'_s = F \odot M_s(F)$$

---

### 3. Sequential Channel–Spatial Attention (CBAM Style)
Sequentially applies channel attention followed by spatial attention:
$$F_c = M_c(F) \odot F$$
$$F_{cs} = M_s(F_c) \odot F_c$$
This paradigm first recalibrates "what is important" and then isolates "where it is located".

---

### 4. Lightweight Spatial Self-Attention (Non-Local Matrix Interaction)
Models pairwise spatial interactions across distant coordinates.
- **Flattening:** $X \in \mathbb{R}^{N \times C}$ where $N = H \times W$.
- **Linear Projections:**
  $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad (W_Q, W_K \in \mathbb{R}^{C \times d}, \; W_V \in \mathbb{R}^{C \times C})$$
- **Scaled Dot-Product Attention:**
  $$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) \in \mathbb{R}^{N \times N}$$
- **Residual Integration:**
  $$Y = A V \in \mathbb{R}^{N \times C}, \quad F_{\text{SA}} = F + \gamma \cdot \text{reshape}(Y)$$
  where $\gamma$ is an initialized learnable scalar.

---

### 5. Efficient Channel Attention (ECA)
Avoids channel dimensionality reduction to preserve direct channel-to-channel communication. Uses a fast 1D convolution with an adaptive kernel size $k$:
$$k = \psi(C) = \left| \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} \right|_{\text{odd}}$$
$$M_{\text{ECA}}(F) = \sigma\left( \text{Conv1D}_k(\text{GAP}(F)) \right)$$
Introduces only **5 trainable parameters** at $P_3$.

---

### 6. Coordinate Attention
Decomposes channel attention into two 1D direction-aware pooling operations (horizontal $H \times 1$ and vertical $1 \times W$) to capture long-range spatial dependencies along one spatial axis while preserving precise positional coordinate information along the other.

---

### 7. Post-Hoc Interpretability: Grad-CAM
Gradient-weighted Class Activation Mapping computes the gradient of the predicted class logit $y^c$ with respect to feature maps $A^k$:
$$\alpha_k^c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{ij}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
*Crucial distinction:* **Internal attention maps** are forward-pass multiplicative gates ($M_c, M_s$) that modify network weights directly; **Grad-CAM** is a post-hoc backward-pass explanation method derived from gradients without altering model representations.

---

## 📊 Comprehensive Experimental Results

### Experiment 1: Attention-Type Ablation (Fixed at Position $P_3$)

All models evaluate identical training/validation splits with backbone frozen initially and fine-tuned uniformly.

| Model Index | Attention Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Total Parameters | Extra Attn Params | Inference Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | **Baseline ResNet50** | $94.92\%$ | $95.05\%$ | $94.81\%$ | $0.9489$ | $23,518,277$ | $0$ (Baseline) | **3.0702 ms** |
| **A1** | **Channel Attention (SE)** | $94.74\%$ | $94.88\%$ | $94.60\%$ | $0.9471$ | $23,650,437$ | $+132,160$ | $3.2828\text{ ms}$ |
| **A2** | **Spatial Attention ($k=7$)** | **95.10%** | **95.14%** | **95.11%** | **0.9512** | $23,518,375$ | **+98** | $3.3299\text{ ms}$ |
| **A3** | **Channel + Spatial ($P_3$)** | $94.19\%$ | $94.45\%$ | $94.04\%$ | $0.9414$ | $23,650,535$ | $+132,258$ | $3.3278\text{ ms}$ |
| **A4** | **Self-Attention ($P_3$)** | $93.47\%$ | $93.45\%$ | $93.44\%$ | $0.9342$ | $24,830,790$ | $+1,312,513$ | $3.5036\text{ ms}$ |
| **A5** | **Combined ($C+S+\text{SA}$)** | $94.92\%$ | $95.39\%$ | $94.67\%$ | $0.9493$ | $24,963,048$ | $+1,444,771$ | $3.5254\text{ ms}$ |

---

### Experiment 2: Attention-Position Ablation (Channel + Spatial Module)

Evaluates the exact same sequential Channel+Spatial attention module inserted at different depths:

| Position ID | ResNet50 Stage | Spatial Res ($H \times W$) | Channels ($C$) | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Module Parameters |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$P_1$** | After `conv2_x` | $56 \times 56$ | $256$ | **94.92%** | $94.80\%$ | **94.94%** | **0.9485** | **8,562** |
| **$P_2$** | After `conv3_x` | $28 \times 28$ | $512$ | $93.47\%$ | $93.72\%$ | $93.24\%$ | $0.9339$ | $33,410$ |
| **$P_3$** | After `conv4_x` | $14 \times 14$ | $1024$ | $94.19\%$ | $94.45\%$ | $94.04\%$ | $0.9414$ | $132,258$ |
| **$P_4$** | After `conv5_x` | $7 \times 7$ | $2048$ | $94.56\%$ | $94.68\%$ | $94.46\%$ | $0.9454$ | $526,562$ |

---

### Experiment 3: Single vs. Multiple Attention Positions

Tests whether cascading attention across multiple stages provides compounding benefits:

| Configuration | Stages Applied | Test Accuracy | Macro F1 | Attention Parameters | Total Model Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single $P_3$** | `conv4_x` | $94.19\%$ | $0.9414$ | $132,258$ | $23,650,535$ |
| **Single $P_4$** | `conv5_x` | **94.56%** | **0.9454** | $526,562$ | $24,044,839$ |
| **Dual $P_3 + P_4$** | `conv4_x` + `conv5_x` | **94.56%** | $0.9445$ | $658,820$ | $24,177,097$ |
| **Triple $P_2 + P_3 + P_4$** | `conv3_x` + `conv4_x` + `conv5_x` | $93.83\%$ | $0.9377$ | $692,230$ | $24,210,507$ |

*Insight:* Multi-stage insertion does not improve performance beyond a single late stage ($P_4$). Aggressively inserting attention across 3 stages ($P_2+P_3+P_4$) reduces macro F1 to $0.9377$ due to compounding optimization interference and over-regularization.

---

### Experiment 4: Channel vs. Spatial Execution Ordering (at $P_3$)

| Ordering | Pipeline Flow | Test Accuracy | Macro F1 | Inference Time |
| :--- | :---: | :---: | :---: | :---: |
| **Channel $\to$ Spatial** | $F \to M_c \to M_s$ | **94.19%** | **0.9414** | **3.3278 ms** |
| **Spatial $\to$ Channel** | $F \to M_s \to M_c$ | $94.01\%$ | $0.9394$ | $3.3315\text{ ms}$ |

*Insight:* Prioritizing channel selection ("which features to attend to") before spatial filtering ("where in the image to look") yields slightly superior feature selection compared to the reverse order.

---

### Experiment 5: Spatial Attention Receptive Field (Kernel Size $k$)

| Convolution Kernel Size | Effective Receptive Field | Test Accuracy | Macro F1 | Inference Time |
| :---: | :---: | :---: | :---: | :---: |
| **$3 \times 3$** | Compact local context | **95.46%** | **0.9544** | **3.3185 ms** |
| **$5 \times 5$** | Medium context | $94.92\%$ | $0.9489$ | $3.3297\text{ ms}$ |
| **$7 \times 7$** | Large spatial context | $95.10\%$ | $0.9512$ | $3.3299\text{ ms}$ |

*Insight:* $k=3$ achieves the highest test accuracy ($95.46\%$) and macro F1 ($0.9544$). Flower parts (e.g., petals, stamen tips) are spatially localized; smaller kernels avoid blurring boundary contours into adjacent background foliage.

---

### Experiment 6: Benchmark Comparison of Established Attention Modules (at $P_3$)

| Mechanism | Accuracy | Precision | Recall | Macro F1 | Total Parameters | Extra Attn Params | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ResNet50** | $94.92\%$ | $95.05\%$ | $94.81\%$ | $0.9489$ | $23,518,277$ | $0$ | **3.0702 ms** |
| **Channel / SE** | $94.74\%$ | $94.88\%$ | $94.60\%$ | $0.9471$ | $23,650,437$ | $+132,160$ | $3.2828\text{ ms}$ |
| **Spatial ($k=7$)** | $95.10\%$ | $95.14\%$ | $95.11\%$ | $0.9512$ | $23,518,375$ | $+98$ | $3.3299\text{ ms}$ |
| **ECA (Efficient Channel)** | **95.46%** | **95.51%** | **95.30%** | **0.9537** | $23,518,282$ | **+5** | $3.3348\text{ ms}$ |
| **CBAM** | $93.28\%$ | $93.62\%$ | $93.03\%$ | $0.9320$ | $23,650,535$ | $+132,258$ | $3.3386\text{ ms}$ |
| **Self-Attention** | $93.47\%$ | $93.45\%$ | $93.44\%$ | $0.9342$ | $24,830,790$ | $+1,312,513$ | $3.5036\text{ ms}$ |
| **Coordinate Attention** | $94.56\%$ | $94.80\%$ | $94.40\%$ | $0.9452$ | $23,618,725$ | $+100,448$ | $3.3219\text{ ms}$ |

---

## 📈 Accuracy vs. Complexity Trade-Off Analysis

```
       Accuracy vs. Complexity Trade-off Landscape:
  0.956 ┤
        │        ★ ECA (+5 params, 95.46%)
  0.952 ┤            ★ Spatial k=7 (+98 params, 95.10%)
        │
  0.948 ┤  ● Baseline (23.52M, 94.92%)
        │         ● SE (23.65M, 94.74%)             ● Combined (24.96M, 94.92%)
  0.944 ┤              ● Coordinate (23.62M, 94.56%)
        │         ● Channel+Spatial (23.65M, 94.19%)
  0.940 ┤
        │
  0.936 ┤                                           ▼ Self-Attention (+1.31M params, 93.47%)
        │         ▼ CBAM (23.65M, 93.28%)
  0.932 ┴───────────────┬───────────────────────────┬───────────────────────────────►
                      23.6M                       24.2M                           25.0M
                                       Total Parameters
```

### Key Analytical Takeaways:
1. **The Efficiency Champion — ECA:**
   - Adds merely **5 parameters** to the 23.52M baseline.
   - Yields the highest accuracy ($95.46\%$) and highest macro F1 ($0.9537$).
   - Completely avoids the dimensionality reduction bottleneck that limits standard Squeeze-and-Excitation.
2. **The Pitfall of Heavy Self-Attention in Transfer Learning:**
   - Spatial self-attention introduces **$1,312,513$ additional parameters** and exhibits the slowest inference time ($3.5036\text{ ms}$).
   - Achieves the lowest accuracy ($93.47\%$) among standalone mechanisms.
   - On moderate-sized transfer learning datasets ($\approx 3,600$ images), full spatial $N \times N$ self-attention matrices overfit easily, requiring heavy pretraining data (e.g., JFT-300M / ImageNet-21k) to generalize effectively.
3. **Spatial Attention is Extremely Cost-Effective:**
   - Spatial attention alone ($k=7$) requires only **98 parameters** while surpassing the baseline ($95.10\%$ vs. $94.92\%$).

---

## 🎯 Confusion Matrix & Class-Wise Performance

Evaluating the 551 held-out test set images across flower classes:

### Per-Class F1 Score Breakdown Across Key Models:

| Model | `daisy` | `dandelion` | `roses` | `sunflowers` | `tulips` | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ResNet50** | $0.96$ | $0.94$ | $0.92$ | $0.96$ | $0.96$ | $0.9489$ |
| **Spatial ($k=3$) @ $P_3$** | $0.96$ | **0.96** | **0.94** | **0.97** | $0.95$ | **0.9544** |
| **ECA @ $P_3$** | $0.95$ | **0.96** | **0.94** | $0.96$ | **0.96** | **0.9537** |
| **Spatial ($k=7$) @ $P_3$** | $0.96$ | $0.94$ | $0.93$ | $0.96$ | $0.95$ | $0.9512$ |
| **Combined ($C+S+\text{SA}$)** | **0.97** | $0.95$ | $0.93$ | $0.96$ | $0.94$ | $0.9493$ |
| **Channel (SE) @ $P_3$** | $0.95$ | $0.95$ | $0.93$ | $0.96$ | $0.95$ | $0.9471$ |
| **Coordinate Attention** | $0.96$ | **0.96** | $0.91$ | **0.97** | $0.92$ | $0.9452$ |
| **Channel+Spatial @ $P_3$** | $0.94$ | $0.94$ | $0.93$ | $0.95$ | $0.94$ | $0.9414$ |
| **Self-Attention @ $P_3$** | $0.95$ | $0.95$ | $0.92$ | $0.94$ | $0.91$ | $0.9342$ |
| **CBAM @ $P_3$** | $0.94$ | $0.94$ | $0.91$ | $0.94$ | $0.93$ | $0.9320$ |

### Error Distribution & Confusion Insights:
- **Hardest Class:** `roses` consistently exhibits the lowest F1-score across all models ($0.91 - 0.94$). Roses share significant color and petal overlap with `tulips` (8–9 rose test samples were misclassified as tulips).
- **Easiest Classes:** `sunflowers` and `daisy` demonstrate high separability ($F1 \ge 0.96$) owing to distinct concentric geometry, large centers, and characteristic ray florets.
- **Attention's Targeted Benefit:** ECA and Spatial ($k=3$) boosted `dandelion` F1 from $0.94$ to $0.96$ and `roses` F1 from $0.92$ to $0.94$ by sharpening focus on petal structures while ignoring lawn backgrounds.

---

## 🔍 Qualitative Interpretability & Visual Diagnostics

### 1. Internal Attention Maps vs. Post-Hoc Grad-CAM
- **Internal Channel Attention:** Weight distributions across $C=1024$ channels reveal sparse selection: the top-20 channels carry disproportionately large modulation weights ($m_c \approx 0.90 - 1.00$), acting as semantic feature selectors.
- **Internal Spatial Attention:** Spatial maps $M_s(F) \in [0, 1]^{14 \times 14}$ produce a high-activation central mask that suppresses boundary pixels and background vegetation before feature pooling.
- **Grad-CAM Explanations:** For correct predictions (e.g., sunflowers and daisies), Grad-CAM heatmaps concentrate tightly on the central disk and petals. For misclassified samples, activations become diffuse or improperly fixate on background foliage.

### 2. Depth Evolution of Spatial Attention ($P_2 \to P_3 \to P_4$)
- **At $P_2$ ($56 \times 56$):** Focuses on fine textural boundaries, edge contours, and individual color transitions.
- **At $P_3$ ($14 \times 14$):** Isolates intermediate object structures (whole flower heads, petal clusters, stems).
- **At $P_4$ ($7 \times 7$):** Forms a broad semantic blob covering the entire salient foreground object.

---

## 📁 Repository Structure & Visual Artifacts

```
Lab 8 Attention/
├── 24011101075_DL_Lab_Exercise_8.pdf        # Complete academic lab report (36 pages)
├── Academics_SNUC (2).pdf                   # Lab instruction manual & syllabus
├── README.md                                # Comprehensive experiment documentation
├── 01_sample_images.pdf                     # Sample flower inputs across all 5 classes
├── 02_class_distribution.pdf                # Per-split dataset class balance histogram
├── 03_acc_curves.pdf                        # Training & validation accuracy trajectories
├── 04_loss_curves.pdf                       # Cross-entropy loss convergence curves
├── 05_bar_accuracy.pdf                      # Classification accuracy comparison across attention types
├── 06_bar_macroF1.pdf                       # Macro F1 comparison across attention models
├── 07_bar_position_acc.pdf                  # Attention position ablation accuracy (P1 to P4)
├── 08_bar_multi_position_f1.pdf             # Single vs. multi-stage attention macro F1
├── 09_bar_order.pdf                         # Attention execution order comparison (C->S vs S->C)
├── 10_bar_kernel.pdf                        # Spatial attention kernel size ablation (k=3, 5, 7)
├── 11_scatter_acc_vs_params.pdf             # Accuracy vs. Model Parameters trade-off scatter plot
├── 12_scatter_f1_vs_time.pdf                # Macro F1 vs. Inference Time per image scatter plot
├── 13_position_curve.pdf                    # Performance curve across network depth positions
├── 14_confusion_counts.pdf                  # Raw prediction count confusion matrices
├── 14_confusion_row-normalised.pdf          # Row-normalized confusion matrices
├── 15_confusion_difference.pdf              # Confusion difference heatmap (Attention minus Baseline)
├── 16_per_class_f1_heatmap.pdf              # 2D heatmap of per-class F1 across all evaluated models
├── 17_feature_maps_before_after.pdf         # Feature activations before and after attention modulation
├── 18_channel_attention_weights.pdf         # Learned channel importance weights distribution
├── 19_spatial_attention_A2_spatial.pdf      # Spatial attention maps and overlays for model A2
├── 19_spatial_attention_A3_cs_P3.pdf        # Spatial attention maps for Channel+Spatial model A3
├── 20_attention_position_heatmap.pdf        # Heatmaps across stages P2, P3, and P4
├── 20_attention_position_overlay.pdf        # Overlays across stages P2, P3, and P4
├── 21_self_attention_responses.pdf          # Spatial self-attention query-key interaction maps
├── 22_gradcam_baseline_vs_attention.pdf     # Grad-CAM comparison: Baseline vs. Attention model
├── 23_qualitative_comparison.pdf            # Multi-model qualitative comparison (correct + misclassified)
└── 24_gradcam_extended_comparison.pdf       # Extended Grad-CAM panel across all attention mechanisms
```

---

## 💡 Key Theoretical Questions & Answers

1. **What is the difference between an internal attention map and Grad-CAM?**  
   *An internal attention map is generated by the attention module directly in the forward pass to reweight intermediate features during inference. Grad-CAM is a post-hoc diagnostic calculated after prediction using backward gradients to highlight where the model looked, without modifying model activations.*

2. **Why does ECA outperform SE and CBAM despite having fewer parameters?**  
   *SE uses a dimension-reducing MLP ($C \to C/r \to C$) that disrupts direct correspondence between channels. ECA replaces this bottleneck with a local 1D convolution over uncompressed channel features, avoiding dimensionality reduction while introducing only 5 parameters.*

3. **Why is spatial self-attention computationally expensive on high-resolution maps?**  
   *For an image feature map with spatial dimensions $H \times W = N$, the self-attention matrix has size $N \times N$. If spatial resolution doubles, $N$ quadruples and the attention matrix memory/compute scales sixteen-fold ($\mathcal{O}(N^2)$).*

4. **Does applying attention at multiple stages always improve classification performance?**  
   *No. While applying attention at both $P_3$ and $P_4$ maintained $94.56\%$ accuracy, inserting attention at three stages ($P_2 + P_3 + P_4$) degraded accuracy to $93.83\%$. Repeated gating compounds gradient attenuation and over-constrains representations.*

5. **Why did spatial kernel $k=3$ outperform $k=7$?**  
   *A smaller $3 \times 3$ kernel captures tight, localized discriminative regions (e.g., petal contours and anthers) without diluting the spatial attention map with background lawn or stems.*

---

## 🛠️ Technologies Used

- **Deep Learning Framework:** TensorFlow 2.x / Keras
- **Pretrained Architecture:** ImageNet ResNet50
- **Attention Modules Implemented:** Squeeze-and-Excitation (SE), Spatial Attention, CBAM, ECA, Coordinate Attention, Scaled Dot-Product Self-Attention
- **Interpretability & Visualization:** Grad-CAM, Matplotlib, Seaborn
- **Metrics & Evaluation:** Scikit-Learn (Classification Report, Confusion Matrix, Macro F1)
- **Documentation & Typesetting:** LaTeX / TeX, Markdown

---

## 📚 References

1. K. He, X. Zhang, S. Ren, and J. Sun, *"Deep Residual Learning for Image Recognition,"* CVPR, pp. 770–778, 2016.
2. J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, *"Squeeze-and-Excitation Networks,"* IEEE TPAMI, vol. 42, no. 8, pp. 2011–2023, 2020.
3. S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, *"CBAM: Convolutional Block Attention Module,"* ECCV, pp. 3–19, 2018.
4. Q. Wang, B. Wu, P. Zhu, P. Li, W. Zuo, and Q. Hu, *"ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks,"* CVPR, pp. 11534–11542, 2020.
5. Q. Hou, D. Zhou, and J. Feng, *"Coordinate Attention for Efficient Mobile Network Design,"* CVPR, pp. 13713–13722, 2021.
6. A. Vaswani et al., *"Attention Is All You Need,"* NeurIPS, vol. 30, 2017.
7. R. R. Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,"* ICCV, pp. 618–626, 2017.


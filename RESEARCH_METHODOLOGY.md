# Research Methodology: AI_LUNG High-Performance CT Analysis

This document outlines the scientific and technical methodologies implemented in the AI_LUNG pipeline. It is designed to serve as a foundation for academic publication and technical documentation.

---

## 🔬 1. Stage 1: High-Fidelity 2.5D Denoising

### 1.1 Neural Architecture
The denoiser utilizes a **U-Net** topology optimized for high-resolution medical imaging.
*   **Encoder (Backbone):** **EfficientNet-B5** pretrained on ImageNet is used as the feature extractor. This provides a deep, multi-scale receptive field while maintaining parameter efficiency.
*   **Attention Mechanism:** **Convolutional Block Attention Modules (CBAM)** are integrated at each skip-connection bottleneck. CBAM performs sequential Channel and Spatial attention, allowing the model to "focus" on clinically relevant structures (e.g., small nodules and vessels) while ignoring random noise.
*   **Dimensionality:** To capture spatial context across slices without the computational cost of 3D convolutions, we use a **2.5D Input**. The central target slice is concatenated with **4 context slices** above and below ($2 \times 4 + 1 = 9$ input channels).

### 1.2 Hybrid Loss Function
The objective function is a weighted combination of three distinct loss components to ensure structural integrity:
$$Loss = \alpha L_1 + \beta (1 - SSIM) + \gamma \nabla Loss$$
*   **L1 Loss:** Ensures pixel-wise Hounsfield Unit (HU) accuracy.
*   **SSIM Loss:** Direct optimization for human-perceived structural similarity.
*   **Gradient Loss ($\nabla$):** Penalizes the loss of sharp edges, preventing the "waxy" or "blurred" appearance common in standard denoising.

---

## ⚡ 2. High-Accuracy Physics Simulation

The project rejects simple Gaussian noise in favor of **Fidelity-First CT Physics**. The low-dose simulation follows the physical photon-count reality:
1.  **Forward Projection:** The normal-dose image is transformed into a **Sinogram** using the **Radon Transform**.
2.  **Quantum Noise Injection:** **Poisson Noise** is injected into the sinogram space, simulating the reduced photon count of a low-dose scan.
3.  **Filtered Back-Projection (FBP):** The noisy sinogram is reconstructed back into the image domain.
*   **Scientific Impact:** This ensures the neural network learns the specific "streak artifacts" and "correlated noise" patterns unique to CT scanners, rather than generic digital noise.

---

## 🏎️ 3. "Turbo" Pipeline: High-Throughput Delivery

To enable training on massive sets (166,000+ slices) in resource-constrained environments like Google Colab, several I/O optimizations were developed:
*   **Local NVMe Synchronization:** A "Best-Effort" syncing algorithm copies data from high-latency Cloud Storage (Google Drive) to the local VM NVMe (SSD). It respects an **80% Disk Safety Limit** to prevent VM crashes while maximizing throughput.
*   **Grouped Series Sampler:** To mitigate the memory bottleneck of large 3D volumes, a custom Sampler groups all slices of a single patient into consecutive batches. This ensures each volume is loaded into the **LRU Cache** exactly once, achieving near **100% cache efficiency**.
*   **O(1) Header Mapping:** Instead of expensive directory crawling, the system uses a **JSON-precomputed mapping** to find `.npy` files instantly, reducing the startup metadata scan from minutes to seconds.

---

## 🛡️ 4. Resiliency & "Invincible" Training

Since high-accuracy physics and full-volume training lead to long epoch times (~12–15 hours), the system implements a robust fault-tolerance mechanism:
*   **Mid-Epoch Checkpointing:** The "brain" state (model weights, optimizer state, batch index) is saved to Google Drive every **500 batches** (~1 hour of training time).
*   **Deterministic Resumption:** Upon reconnection, the system identifies the last saved batch and **resumes training precisely from that point**. This eliminates progress loss due to internet disconnections or GPU provider timeouts.

---

## 📊 5. Evaluation Metrics
Models are evaluated using clinical-standard metrics:
*   **PSNR (Peak Signal-to-Noise Ratio):** Measuring the clarity of simulated low-dose restoration.
*   **SSIM (Structural Similarity Index):** Ensuring the preservation of lung anatomy and diagnostic features.
*   **Malignancy Detection:** Area Under the ROC Curve (**AUC**), Sensitivity (Recall), and Specificity.

---
**Last Updated:** March 2026
**Status:** Peak Performance Configuration (Physics Mode + Full Sampling)

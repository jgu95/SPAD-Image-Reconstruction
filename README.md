# SPAD-Image-Reconstruction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**PyTorch implementation of a Two-stage SE-ResUNet for Single-Photon Avalanche Diode (SPAD) image reconstruction. Features a GPU-optimized burst processing pipeline tailored for photon-limited sensors.**

---

## 📸 Overview

Single-Photon Avalanche Diodes (SPADs) offer extreme low-light sensitivity but suffer from heavy signal-dependent Poisson noise. This project implements a **physics-aware deep learning pipeline** to reconstruct high-fidelity images from raw 48-channel SPAD burst sequences.

**Key Technical Highlights:**
* **Two-Stage Architecture:** Cascaded **SE-ResUNet** (Squeeze-and-Excitation Residual U-Net) to balance coarse denoising and fine texture recovery.
* **H100 Optimization:** Addressed system bottlenecks (I/O latency & VRAM bandwidth) using multi-threaded prefetching and mixed-precision (AMP) training.
* **Physics-Based:** Designed to handle extreme shot noise inherent to photon counting, outperforming Gaussian-based heuristics.

![Result Preview](https://via.placeholder.com/800x200?text=Place+Your+Result+Comparison+Image+Here)
*(Figure: Comparison between Raw SPAD Input, Baseline, and Our Reconstruction)*
% -------------------------------------------------------------------
\section{Experiments and Results}
\subsection{Quantitative Performance}
\begin{table}[h!]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Model Stage} & \textbf{PSNR (dB)} & \textbf{SSIM} & \textbf{L1 Loss} \
\hline
Stage 1 (Base) & 23.77 & 0.6867 & 0.0487 \
Stage 2 (EMA) & 30.56 & 0.8734 & 0.0182 \
Stage 2 (EMA2 Hybrid) & \textbf{31.45} & \textbf{0.8807} & \textbf{0.0165} \
\hline
\end{tabular}
\end{table}
---

## 💾 Downloads (Models & Dataset)

Due to GitHub's file size limits, the pre-trained weights and the full training dataset are hosted externally. Please download them and place them in the `./weights` and `./data` directories respectively.

| File Name | Description | Size | Download Link |
| :--- | :--- | :--- | :--- |
| **`stage1_coarse_ema.pth`** | **Stage 1 Model**: Coarse reconstruction & noise suppression | ~125 MB | https://drive.google.com/file/d/13d5Eit6cy2E1sysZ36UIwunSa3d18ghJ/view?usp=sharing |
| **`stage2_refine_ema.pth`** | **Stage 2 Model**: Fine texture refinement (EMA optimized) | ~125 MB | https://drive.google.com/file/d/17F-JVEiBGSgoX6X0QIYylzxXZA6DWf3p/view?usp=sharing |
| **`spad_trainset.zip`** | **Training Data**: Pre-processed SPAD burst sequences | ~1.6 GB | https://drive.google.com/file/d/1BpLto98TPUqskZyphN-FK_V4sCl8AbMl/view?usp=sharing |

---

## 🛠️ Project Structure

```text
SPAD-Image-Reconstruction/
├── weights/               # Place downloaded .pth files here
│   ├── stage1_coarse_ema.pth
│   └── stage2_refine_ema.pth
├── data/                  # Place unzipped dataset here
├── model_arch.py          # SE-ResUNet architecture definition
├── utils.py               # Raw SPAD decoding & preprocessing logic
├── inference.py                # Inference demo script
├── requirements.txt       # Python dependencies
├── Technical_Report.pdf   # Full project documentation
└── README.md

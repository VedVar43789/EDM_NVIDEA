# Steering Diffusion Models: EDM Reproduction and Classifier-Free Guidance Analysis on CIFAR-10 and MNIST

**Author:** Vedant Vardhaan, DSC 180A Section D32, UCSD, Fall 2026

**Mentor:** Prof. Yusu Wang

## Overview

This repository holds the Quarter 1 capstone for DSC 180A. The project studies the EDM paper (Karras et al., NeurIPS 2022) and classifier-free guidance. Experiments use the NVIDIA pretrained EDM models on CIFAR-10 and a small U-Net on MNIST.

Sampling covers the unconditional and class-conditional settings, followed by a sweep over the guidance weight. Recorded metrics include accuracy, FID, and class-conditional FID. Quarter 2 is planned to extend this setup into diffusion steering through an NA-RFM reproduction.

## Repository Structure

| Path | Contents |
| --- | --- |
| `notebooks/` | Lab notebooks. Place `cifar10_edm_lab.ipynb` and `guidance_mnist.ipynb` here. |
| `scripts/` | Optional helper scripts used outside the notebooks. |
| `figures/` | Plots and sample grids exported for the report. |
| `results/` | Metric files written by the notebooks, including CFG sweep JSON. |
| `report/` | LaTeX source `main.tex` and the compiled PDF `main.pdf`. |
| `README.md` | Project overview, setup, data notes, and the figure map. |
| `NOTES.md` | Predictions, discussion notes, and open questions. |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | MIT license. |

## Setup

- Python 3.10 or later.
- Install dependencies with `pip install -r requirements.txt`.
- A GPU with CUDA 11 or later is required for CIFAR-10 training. CPU is fine for MNIST.
- DSMLP and Google Colab both work.

## Data

CIFAR-10 has 50,000 training images and 10,000 test images, each 32x32, across 10 classes. MNIST has 60,000 training images, each 28x28, across 10 classes. Both datasets are auto-downloaded by the notebooks on first run.

## Reproducing Figures and Tables

| Item | Content | Notebook | Cell |
| --- | --- | --- | --- |
| Figure 1 | Unconditional CIFAR-10 samples | `notebooks/cifar10_edm_lab.ipynb` | Part 1 |
| Figure 2 | Denoiser trajectory | `notebooks/cifar10_edm_lab.ipynb` | Part 1, trajectory cell |
| Figure 3 | Conditional class grid | `notebooks/cifar10_edm_lab.ipynb` | Part 2 |
| Figure 4 | CFG sweep plot: accuracy, FID, FID_class vs w | `notebooks/cifar10_edm_lab.ipynb` | Part 5 |
| Table 1 | CFG metrics per w | `notebooks/cifar10_edm_lab.ipynb` | Part 5, writes `results/cfg_sweep_n1000.json` |
| Figure 5 | Loss curve and FID across snapshots for small-subset training | `notebooks/cifar10_edm_lab.ipynb` | Part 6 |
| Figure 6 | Nearest training image comparison | `notebooks/cifar10_edm_lab.ipynb` | Part 6 |
| Figure 7 | MNIST CFG grid across w | `notebooks/guidance_mnist.ipynb` | Part 2 |
| Table 2 | MNIST target match, confidence, pixel spread per w | `notebooks/guidance_mnist.ipynb` | Part 2 |

## Report

The report source is `report/main.tex`. The compiled PDF belongs at `report/main.pdf` and is meant to be committed with the repository.

## References

- Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022. https://arxiv.org/abs/2206.00364
- NVIDIA EDM code: https://github.com/NVlabs/edm

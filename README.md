# Steering Diffusion Models: EDM Reproduction and Classifier-Free Guidance Analysis on CIFAR-10 and MNIST

**Author:** Vedant Vardhaan, DSC 180A Section D32, UCSD, Fall 2026
**Mentor:** Prof. Yusu Wang

## Overview

This repository holds the Quarter 1 capstone for DSC 180A. The project studies the EDM paper (Karras et al., NeurIPS 2022) and classifier-free guidance (CFG). Experiments use the NVIDIA pretrained EDM models on CIFAR-10 and a small class-conditional U-Net on MNIST.

Sampling covers the unconditional and class-conditional settings, followed by a sweep over the guidance weight w. Recorded metrics include label accuracy, FID, and per-class FID. Quarter 2 is planned to extend this setup into diffusion steering through a reproduction of noise alignment and recursive feature machine (NA-RFM) steering by Wang, Belkin, and Wang.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Open `notebooks/cifar10_edm_lab.ipynb` on a GPU runtime and run every cell top to bottom (about 1 hour on an NVIDIA A100 using Google Collab Pro).
3. Open `notebooks/guidance_mnist.ipynb` on CPU or GPU and run every cell top to bottom (about 60 to 75 minutes on CPU, 10 to 20 minutes on GPU).
4. All figures export to `figures/`, metrics write to `results/`, and generated samples cache in `outputs/`.
5. Compile the report at `report/main.tex` to produce the submission PDF.

## Repository Structure

| Path | Contents |
| --- | --- |
| `notebooks/` | Lab notebooks: `cifar10_edm_lab.ipynb` and `guidance_mnist.ipynb` |
| `scripts/` | Optional helper scripts used outside the notebooks |
| `figures/` | Plots and sample grids exported for the report |
| `results/` | Metric files written by the notebooks, including CFG sweep JSON |
| `report/` | LaTeX source `main.tex` and the compiled PDF `main.pdf` |
| `downloads/` | Auto-created by notebooks for pretrained weights and data (~720 MB) |
| `outputs/` | Auto-created by notebooks for cached generated samples |
| `README.md` | Project overview, setup, data notes, and the figure map |
| `NOTES.md` | Predictions, discussion notes, and open questions |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT license |

## Setup

**Requirements:**

- Python 3.10 or later
- A CUDA 11 or later GPU for CIFAR-10 training (A100, T4, or equivalent)
- CPU is sufficient for MNIST (GPU recommended for speed)
- About 2 GB free disk space for downloads and cached outputs

**Install:**

- cd report
- pdflatex main.tex
- bibtex main
- pdflatex main.tex
- pdflatex main.tex

Overleaf also compiles the report directly.

## References

- Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022. https://arxiv.org/abs/2206.00364
- NVIDIA EDM code: https://github.com/NVlabs/edm
- Ho, J., and Salimans, T. Classifier-Free Diffusion Guidance. NeurIPS 2021 Workshop on Deep Generative Models.
- CIFAR-10 ResNet-56 classifier: https://github.com/chenyaofo/pytorch-cifar-models

## License

MIT. See `LICENSE`.
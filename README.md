# Steering Diffusion Models: EDM Reproduction and Classifier-Free Guidance Analysis on CIFAR-10 and MNIST

**Author:** Vedant Vardhaan, DSC 180A Section D32, UCSD, Fall 2026
**Mentor:** Prof. Yusu Wang

## Overview

This repository holds the Quarter 1 capstone for DSC 180A. The project studies the EDM paper (Karras et al., NeurIPS 2022) and classifier-free guidance (CFG). Experiments use the NVIDIA pretrained EDM models on CIFAR-10 and a small class-conditional U-Net on MNIST.

Sampling covers the unconditional and class-conditional settings, followed by a sweep over the guidance weight w. Recorded metrics include label accuracy, FID, and per-class FID. For Quarter 2, after a meeting this week with Prof. Yusu Wang, our team is pivoting from image steering to continuous diffusion for text generation. The EDM design-space view transfers directly to the continuous embedding space used by methods like Diffusion-LM, DiffuSeq, and CDCD.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Open `notebooks/cifar10_edm_lab.ipynb` on a GPU runtime and run every cell top to bottom (about 1 hour on an NVIDIA A100 using Google Colab Pro).
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

```
pip install -r requirements.txt
```

**Tested platform:** Google Colab Pro with A100 runtime. DSMLP GPU containers should also work.

## Data

**CIFAR-10:** 50,000 training and 10,000 test images, 32x32 RGB, 10 balanced classes. Auto-downloaded from the Hugging Face mirror on first run. If `cifar-10-python.tar.gz` is already in `downloads/`, that copy is used instead.

**MNIST:** 60,000 training images, 28x28 grayscale, 10 balanced classes. Auto-downloaded by the MNIST notebook on first run.

**Pretrained weights (auto-downloaded, hash-checked):**

- NVIDIA EDM CIFAR-10 unconditional and conditional checkpoints
- CIFAR-10 Inception features reference
- CIFAR-10 ResNet-56 classifier (chenyaofo/pytorch-cifar-models)
- Inception-v3 FID detector
- Course MNIST U-Net generator and digit evaluator

Total download size: about 720 MB. Interrupted downloads resume on rerun.

## Reproducing Figures and Tables

Every figure and table in `report/main.pdf` is produced by running the two notebooks top to bottom. Save displayed figures into `figures/` as noted.

**CIFAR-10 Notebook (`notebooks/cifar10_edm_lab.ipynb`):**

| Item | Content | Section | Output |
| --- | --- | --- | --- |
| Figure 1 | Unconditional CIFAR-10 samples | Part 1 | `figures/uncond_grid.png` |
| Figure 2 | Denoiser trajectory, unconditional | Part 1 | `figures/uncond_trajectory.png` |
| Figure 3 | Conditional samples, one class per row | Part 2 | `figures/cond_grid.png` |
| Figure 4 | CFG trajectory, w=1 vs w=4 | Part 3, "Your turn" cell | `figures/cfg_trajectory_w1.png`, `figures/cfg_trajectory_w4.png` |
| Table 1 | CFG metrics per w | Part 5 | `results/cfg_sweep_n1000.json` |
| Figure 5 | CFG sweep plot: accuracy, FID, FID_class vs w | Part 5 | `figures/cfg_sweep.png` |
| Figure 6 | Training loss and FID across snapshots | Part 6 | `figures/training_loss_fid.png` |
| Figure 7 | Nearest training image comparison | Part 6 | `figures/nearest_100.png`, `figures/nearest_1000.png` |

**MNIST Notebook (`notebooks/guidance_mnist.ipynb`):**

| Item | Content | Section | Output |
| --- | --- | --- | --- |
| Figure 8 | MNIST CFG grid across w (target digit 3, seed 42) | Part 3, "Your comparison" cell | `figures/mnist_your_w_sweep.png` |
| Table 2 | MNIST target match, confidence, NFE per w | Part 3 | Printed inline, recorded in `NOTES.md` |

**Important:** the MNIST notebook writes a helper file `mnist_guidance_lab.py` next to the notebook. Run cells top to bottom so this import succeeds.

## Observed Results Reference

CIFAR-10 CFG sweep (1,000 images per w, FID against the full 50,000-image CIFAR-10 reference):

| w | Label Accuracy | FID | FID_class |
| --- | --- | --- | --- |
| 0 | 11.5% | 31.18 | - |
| 1 | 95.3% | 31.33 | 112.1 |
| 1.5 | 98.7% | 31.45 | 107.6 |
| 2 | 98.8% | 33.05 | 108.9 |
| 3 | 99.5% | 36.06 | 112.5 |
| 4 | 98.9% | 39.06 | 117.6 |

Published EDM FIDs (1.97 unconditional, 1.79 class-conditional) use 50,000 samples per setting and are not directly comparable to these.

## Report

Source: `report/main.tex`. Compiled PDF: `report/main.pdf`.

To compile locally:

```
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Overleaf compiles the report directly without the local commands.

## References

- Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022. https://arxiv.org/abs/2206.00364
- NVIDIA EDM code: https://github.com/NVlabs/edm
- Ho, J., and Salimans, T. Classifier-Free Diffusion Guidance. NeurIPS 2021 Workshop on Deep Generative Models.
- CIFAR-10 ResNet-56 classifier: https://github.com/chenyaofo/pytorch-cifar-models

## Contribution Statement: Human and AI

This project was carried out by Vedant Vardhaan with mentorship from Prof. Yusu Wang. AI assistants (primarily Claude) were used throughout as a working tool. The sections below summarize the split; a longer appendix with specific prompts, outputs, and verification steps is included at the end of the report.

**My own work:**

- Running every experiment end to end on Google Colab A100 (CIFAR-10 and MNIST notebooks)
- Writing the Part 3 and Part 6 cells that the notebooks ask students to fill in
- Interpreting every figure and table, writing all answers to notebook questions in my own words
- Deciding what the report should argue and in what order
- Writing the first-pass of every section (abstract, intro, data, results, takeaways) based on my own reading of the EDM paper and the course slides
- Taking initiative to set up a meeting with Prof. Yusu Wang this week to discuss the Q2 project direction, which led to the pivot from image steering to continuous diffusion for text generation
- Reviewing every AI output for accuracy and rewriting anything that did not match what I actually did

**Where AI assisted:**

- Breaking down the assignment PDF and course page into a concrete timeline and deliverable list
- Producing quick plain-language summaries of the EDM paper, Diffusion-LM, DiffuSeq, and CDCD so I could scope the Q2 reading without reading all four end to end on day one
- Debugging notebook errors and explaining what cells were doing step by step
- Restructuring my draft prose into tighter sections, fixing LaTeX issues (citations rendering as "?", figure path mismatches, subcaption layout)
- Drafting the README structure, the AI use appendix format, and the NOTES.md scaffolding
- Rephrasing sections of the report into more natural, concise language once I had written the first-pass content

**Verification I did on AI output:**

- Every number in the report (accuracies, FID values, loss values) was copied directly from my own notebook runs, not from an AI estimate
- Every claim about what EDM, CFG, or the three Q2 papers do was cross-checked against the actual paper or the course slides
- Every code cell AI drafted was executed before accepting it
- Every citation key in `reference.bib` was checked against the real paper on arXiv or the publication venue

## License

MIT. See `LICENSE`.
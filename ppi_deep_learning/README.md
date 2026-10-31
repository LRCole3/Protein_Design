# Deep Learning Architectures for Protein–Protein Interactions

A progressive, hands-on series that extends the NIGMS Sandbox module
[**Protein-Protein Interactions using ML**](https://github.com/NIGMS/Protein-Protein-Interactions-using-ML)
from random forests, logistic regression and MLPs to the architectures behind modern
protein modelling: **variational autoencoders, GANs, diffusion models and transformers**.

Every notebook uses the same real data as the NIGMS module: molecular dynamics (MD) of the
SARS-CoV and SARS-CoV-2 receptor-binding domains (RBD) bound to human ACE2, described by
4,886 inverse distances between interface residue pairs. Each architecture is introduced
**because the previous one fails at something measurable**, and a shared scorecard makes
those failures visible.

## Learning path

| # | Notebook | Architecture | Question it answers | Limitation that motivates the next step | Status |
|---|---|---|---|---|---|
| 01 | [Deep learning foundations](01_deep_learning_foundations.ipynb) | PyTorch MLP, gradient saliency, integrated gradients | Which residue pairs separate the two variants? | A classifier models p(y \| x) only; it cannot describe or generate conformations | available |
| 02 | [Autoencoders → VAEs](02_autoencoders_and_VAEs.ipynb) | PCA, autoencoder, VAE, conditional VAE | What does the conformational landscape look like? Can we sample it? | Gaussian likelihood gives over-smoothed samples | available |
| 03 | [GANs](03_GANs.ipynb) | GAN, WGAN-GP, projection-discriminator cGAN | Can we generate sharp, realistic frames? | Unstable training, mode collapse, no likelihood | available |
| 04 | [Diffusion models](04_diffusion_models.ipynb) | DDPM, v-prediction, DDIM, classifier-free guidance | Can we get quality *and* coverage with stable training? | Flat-vector MLPs ignore residue structure | available |
| 05 | Attention & transformers | Multi-head attention, residue tokens, pair-biased cross-attention | Which residues talk to which partners? | Discriminative only | coming next |
| 06 | Capstone: diffusion transformer | adaLN diffusion transformer, flow matching; map to ESM, AlphaFold 3, RFdiffusion | How do today's protein design models fit together? | — | coming next |

## What's different from the NIGMS module

* **Honest train/test splits.** Consecutive MD frames are nearly identical, so a random split
  leaks information. Notebook 01 measures the trajectory's autocorrelation and introduces a
  *block* split (50-frame blocks, every 5th held out) and a strict *run* split (train on run 1,
  test on run 2).
* **A shared generative scorecard** (`ppi_dl/metrics.py`): precision and coverage
  (Naeem *et al.*, 2020), MMD, a classifier two-sample test, a frozen "judge" classifier for
  conditional models, and a memorization check. Each notebook appends its rows to
  `results/scorecard.csv`, so later notebooks compare against earlier ones.
* **Explicit PyTorch training loops**, so the same five-line loop can be followed from MLPs
  to diffusion transformers.
* **A data-loading fix.** The upstream Submodule 3 notebook loads `cov2_run2` from
  `cropped_cov1_run2.npy`. `ppi_dl.data` loads each run from its own file.

## Getting started

```bash
git clone https://github.com/LRCole3/Protein_Design.git
cd Protein_Design/ppi_deep_learning
pip install -r requirements.txt
jupyter lab
```

Run the notebooks **in order, from inside the `ppi_deep_learning/` folder**. On first use
`ppi_dl.download_data()` fetches the NIGMS ML input files (about 78 MB) into `data/`, which is
git-ignored. If you already cloned the NIGMS repository, skip the download with:

```bash
export PPI_DATA_DIR=/path/to/Protein-Protein-Interactions-using-ML/tutorial_files/ML_input_files
```

No GPU is needed. Each notebook runs in roughly 1–10 minutes on a laptop CPU, and a GPU is
used automatically if present.

## Repository layout

```
ppi_deep_learning/
├── 01_deep_learning_foundations.ipynb
├── 02_autoencoders_and_VAEs.ipynb
├── 03_GANs.ipynb
├── 04_diffusion_models.ipynb
├── ppi_dl/
│   ├── data.py       # download, 1/d transform, standardization, splits, residue maps
│   ├── diffusion.py  # DDPM schedule, v-prediction training, DDIM sampler (NB04)
│   ├── metrics.py    # shared generative-model scorecard
│   └── plotting.py   # per-residue importance and sample-comparison plots
├── results/
│   └── scorecard.csv # rows appended by each generative notebook
└── requirements.txt
```

Each notebook follows the NIGMS format: overview, learning objectives, worked sections,
**Challenge Problems** with a solution cell immediately below, a summary and references.

## Data and credit

The MD-derived features and residue-pair files are from the NIGMS Sandbox module
*Protein-Protein Interactions using ML*. That module is based on:

> Pavlova A. *et al.* (2021) Machine learning reveals the critical interactions for SARS-CoV-2
> spike protein binding to ACE2. *J. Phys. Chem. Lett.* 12, 5494–5502.
> [doi:10.1021/acs.jpclett.1c01494](https://doi.org/10.1021/acs.jpclett.1c01494)

Please cite the original work and follow the NIGMS repository's license when reusing the data.

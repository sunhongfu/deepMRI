# deepMRI: Deep Learning Methods for MRI

**Authors:** Yang Gao, Zhuang Xiong, Hongfu Sun

This repository is a collection of deep learning tools for MRI reconstruction and quantitative mapping. Each method lives in its own standalone repository (linked below). Each subfolder here contains a short README that links directly to the corresponding standalone repo.

> **Clinicians / non-coders:** iQSM+ is available as a browser-based web app — no MATLAB or command line needed. See the [iQSM+ repo](https://github.com/sunhongfu/iQSM_Plus) for Docker and conda instructions.

---

## Projects

| Method | Task | Paper |
|--------|------|-------|
| [iQSM+](#iqsm-plus) | Orientation-adaptive single-step QSM | [MIA 2024](https://doi.org/10.1016/j.media.2024.103160) |
| [iQSM](#iqsm) | Single-step instant QSM | [NeuroImage 2022](https://www.sciencedirect.com/science/article/pii/S1053811922005274) |
| [xQSM](#xqsm) | QSM dipole inversion | [NMR Biomed 2021](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/nbm.4461) |
| [DCRNet](#dcrnet) | QSM + R2* acceleration | [NeuroImage 2021](https://www.sciencedirect.com/science/article/pii/S1053811921006790) |
| [BFRnet](#bfrnet) | QSM background field removal | [arXiv 2022](https://arxiv.org/abs/2204.02760) |
| [AFTER-QSM](#after-qsm) | QSM for oblique/anisotropic scans | [NeuroImage 2022](https://www.sciencedirect.com/science/article/pii/S1053811922009636) |
| [MoDIP](#modip) | Model-based deep image prior QSM | [NeuroImage 2024](https://www.sciencedirect.com/science/article/pii/S1053811924000788) |
| [DIP-UP](#dip-up) | Deep image prior phase unwrapping | — |
| [DeepRelaxo](#deeprelaxo) | Fast brain R2* mapping | [MRM 2024](https://github.com/sunhongfu/DeepRelaxo) |

---

## How to Get the Code

Each method lives in its own **standalone GitHub repository**. Clone the one you need directly — links are in the [Projects](#projects) table above. For example:

```bash
git clone https://github.com/sunhongfu/iQSM.git
```

Each subfolder in this deepMRI repo contains a README that links to the corresponding standalone repository.

---

## Requirements

Each method has its own dependencies — see the individual repo for details. General requirements:

- Python 3.7+ with PyTorch 1.8+
- MATLAB R2019a+ (for MATLAB wrappers, not required for web app)
- NVIDIA GPU recommended

---

[⬆ top](#deepmri-deep-learning-methods-for-mri)

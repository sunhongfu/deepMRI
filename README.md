# deepMRI: Deep Learning Methods for MRI

**Authors:** Yang Gao, Zhuang Xiong, Hongfu Sun

This repository is a collection of deep learning tools for MRI reconstruction and quantitative mapping. Each method lives in its own standalone repository (linked below). Each subfolder here contains a short README that links directly to the corresponding standalone repo.

> **Clinicians / non-coders:** iQSM+ is available as a browser-based web app — no MATLAB or command line needed. See the [iQSM+ repo](https://github.com/sunhongfu/iQSM_Plus) to get started.

---

## Projects

| Method | Task | Paper |
|--------|------|-------|
| [iQSM+](https://github.com/sunhongfu/iQSM_Plus) | Orientation-adaptive single-step QSM | [MIA 2024](https://doi.org/10.1016/j.media.2024.103160) |
| [iQSM](https://github.com/sunhongfu/iQSM) | Single-step instant QSM | [NeuroImage 2022](https://doi.org/10.1016/j.neuroimage.2022.119327) |
| [xQSM](https://github.com/sunhongfu/xQSM) | QSM dipole inversion | [NMR Biomed 2021](https://doi.org/10.1002/nbm.4461) |
| [DCRNet](https://github.com/sunhongfu/DCRNet) | QSM + R2* acceleration | [NeuroImage 2021](https://doi.org/10.1016/j.neuroimage.2021.118771) |
| [BFRnet](https://github.com/sunhongfu/BFRnet) | QSM background field removal | [arXiv 2022](https://arxiv.org/abs/2204.02760) |
| [AFTER-QSM](https://github.com/sunhongfu/AFTER-QSM) | QSM for oblique/anisotropic scans | [NeuroImage 2022](https://doi.org/10.1016/j.neuroimage.2022.119824) |
| [MoDIP](https://github.com/sunhongfu/MoDIP) | Model-based deep image prior QSM | [NeuroImage 2024](https://doi.org/10.1016/j.neuroimage.2024.120540) |
| [DIP-UP](https://github.com/sunhongfu/DIP-UP) | Deep image prior phase unwrapping | — |
| [DeepRelaxo](https://github.com/sunhongfu/DeepRelaxo) | Fast brain R2* mapping | [MRM 2024](https://doi.org/10.1002/mrm.70405) |

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

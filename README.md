# deepMRI: Deep Learning Methods for MRI

**Authors:** Yang Gao, Zhuang Xiong, Hongfu Sun

This repository is a collection of deep learning tools for MRI reconstruction and quantitative mapping. Each method lives in its own standalone repository (linked below) and is also accessible here as a git submodule.

> **Clinicians / non-coders:** see [Quick Start](#-quick-start--no-matlab-required) for a browser-based web app — no installation needed.

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

---

## How the Code is Organised

Each method in this collection lives in its own **standalone GitHub repository**. This deepMRI repo links to all of them as [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) — meaning each subfolder (e.g. `iQSM/`, `xQSM/`) is a pointer to the corresponding standalone repo, not a copy of the files.

**You have two options depending on your needs:**

| Goal | What to do |
|------|------------|
| Use or contribute to **one specific method** | Go directly to its standalone repo (links in the table above) and clone that |
| Get **everything** in one place | Clone this deepMRI repo with `--recurse-submodules` (see below) |

---

## Cloning

### Option 1 — Clone a single method (recommended if you only need one)

Go to the standalone repo for that method and clone it directly, e.g.:

```bash
git clone https://github.com/sunhongfu/iQSM.git
```

Standalone repos for all methods are linked in the [Projects](#projects) table above.

### Option 2 — Clone everything

```bash
git clone --recurse-submodules https://github.com/sunhongfu/deepMRI.git
```

> **Important:** A plain `git clone` (without `--recurse-submodules`) will give you **empty subfolders**. You must include that flag to get the actual code.

If you already cloned without the flag, run:

```bash
git submodule update --init --recursive
```

### Keeping your clone up to date

Each subproject is maintained independently. To pull the latest changes from all subprojects into your deepMRI clone:

```bash
git submodule update --remote --recursive
```

Or for a single subproject:

```bash
git submodule update --remote iQSM
```

---

## Requirements

- Python 3.7+ with PyTorch 1.8+
- MATLAB R2019a+ (for MATLAB wrappers)
- NVIDIA GPU recommended

Tested on: CentOS 7.8 (Tesla V100), macOS 12 / Windows 10 / Ubuntu 19.10 (RTX 4090).

---

## Quick Start – No MATLAB Required

The iQSM+ method is available as a **browser-based web app** powered by [Gradio](https://gradio.app). No MATLAB, no conda setup, no command line needed.

### Option A – Docker (recommended for clinicians)

```bash
# 1. Install Docker Desktop: https://docs.docker.com/get-docker/
# 2. Clone this repo
git clone --recurse-submodules https://github.com/sunhongfu/deepMRI.git
cd deepMRI

# 3. Build and launch (GPU – needs NVIDIA Container Toolkit)
docker compose up

# 3b. CPU-only (slower, works on any machine)
#     Edit docker-compose.yml: remove the 'deploy' block, then:
docker compose up

# 4. Open browser: http://localhost:7860
```

### Option B – Conda

```bash
conda env create -f environment.yml
conda activate deepmri
python app/app.py   # opens http://localhost:7860
```

### Option C – pip

```bash
pip install -r requirements.txt
python app/app.py
```

**Web UI features:**
- Upload phase NIfTI (`.nii` / `.nii.gz`) or DICOM
- Enter echo time(s) in seconds
- Optionally upload magnitude and brain mask
- Click **Run Reconstruction**
- Download QSM result NIfTI — view in FSLeyes / ITK-SNAP / 3D Slicer

---

[⬆ top](#deepmri-deep-learning-methods-for-mri)

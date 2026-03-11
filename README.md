# deepMRI: Deep learning methods for MRI

**Authors:** Yang Gao, Zhuang Xiong, Hongfu Sun

- This repo is devloped based on Pytorch (1.8 or later) and matlab (R2019a or later).

- The codes in this repo were tested on Centos 7.8 with Nvdia Tesla V100 and macos12.0.1/win10/ubuntu19.10 with NViDia 4090.

- Major update, 19, March, 2025: We now have a new and more user-friendly matlab wrapper for iQSM+/iQSM/iQFM/xQSM/xQSM+ reconstuctions (with simpler syntaxes); see repo for iQSM+ ([iQSM+](#head5)) for more details.

---

## 🚀 Quick Start – No MATLAB required

The iQSM+ method is now available as a **browser-based web app** powered by
[Gradio](https://gradio.app). No MATLAB, no conda setup, no command line needed.

### Option A – Docker (recommended for clinicians)

```bash
# 1. Install Docker Desktop (https://docs.docker.com/get-docker/)
# 2. Clone this repo
git clone https://github.com/sunhongfu/deepMRI.git
cd deepMRI

# 3. Build and launch (GPU version – needs NVIDIA Container Toolkit)
docker compose up

# 3b. CPU-only (slower but works on any machine)
#     Edit docker-compose.yml: remove the 'deploy' block, then:
docker compose up

# 4. Open in your browser
#    http://localhost:7860
```

### Option B – Conda (for researchers)

```bash
conda env create -f environment.yml   # one-time setup
conda activate deepmri
python app/app.py                     # opens http://localhost:7860
```

### Option C – pip

```bash
pip install -r requirements.txt
python app/app.py
```

**What to expect in the web UI:**
- Upload phase NIfTI (`.nii` / `.nii.gz`)
- Enter echo time(s) in seconds
- Optionally upload magnitude and brain mask
- Click **Run Reconstruction**
- Download the QSM result NIfTI and view in FSLeyes / ITK-SNAP / 3D Slicer

---

- Major Update, 31st, Jan, 2023: Delete old-version iQSM checkpoints. Will upload latest codes and checkpoints shortly.

&nbsp;
&nbsp;
&nbsp;

# Projects
[iQSM+ for orientation-adaptive single-step QSM reconstruction](#head5)

[iQSM for single-step instant QSM](#head4)

[DCRNet for QSM and R2* acceleration](#head3)

[xQSM for QSM dipole inversion](#head2)

[BFRnet for QSM background field removal](#head1)

&nbsp;
&nbsp;
&nbsp;

# <span id="head5"> iQSM+ for orientation-adaptive single-step QSM reconstruction </span>
**Instant tissue field and magnetic susceptibility mapping from MRI raw phase using Laplacian enabled deep neural networks**

[source code (github)](https://github.com/sunhongfu/deepMRI/tree/master/iQSM_Plus) &nbsp;  | &nbsp;   [arXiv (pre-print)](https://arxiv.org/abs/2311.07823) &nbsp;  |  &nbsp;  [MIA (full paper)](https://doi.org/10.1016/j.media.2024.103160)

![Whole Framework](https://github.com/sunhongfu/deepMRI/blob/master/iQSM_Plus/figs/fig1.png)

# <span id="head4"> iQSM for single-step instant QSM </span>
**Instant tissue field and magnetic susceptibility mapping from MRI raw phase using Laplacian enabled deep neural networks**

[source code (github)](https://github.com/sunhongfu/deepMRI/tree/master/iQSM) &nbsp;  | &nbsp;  [data & checkpoints (dropbox)](https://www.dropbox.com/sh/9kmbytgf3jpj7bh/AACUZJ1KlJ1AFCPMIVyRFJi5a?dl=0) &nbsp;  | &nbsp;  [arXiv (pre-print)](https://arxiv.org/abs/2111.07665) &nbsp;  |  &nbsp;  [NeuroImage (full paper)](https://www.sciencedirect.com/science/article/pii/S1053811922005274)

![Whole Framework](https://www.dropbox.com/s/7bxkyu1utxux76k/Figs_1.png?raw=1)

&nbsp;
&nbsp;
&nbsp;

# <span id="head3"> DCRNet for QSM and R2* acceleration </span>
**Accelerating quantitative susceptibility and R2\* mapping using incoherent undersampling and deep neural network reconstruction**

[source code (github)](https://github.com/sunhongfu/deepMRI/tree/master/DCRNet) &nbsp;  | &nbsp;  [data & checkpoints (dropbox)](https://www.dropbox.com/sh/p9k9rq8zux2bkzq/AADSgw3bECQ9o1dPpIoE5b85a?dl=0) &nbsp;  | &nbsp;  [arXiv (pre-print)](https://arxiv.org/abs/2103.09375) &nbsp;  |  &nbsp;  [NeuroImage (full paper)](https://www.sciencedirect.com/science/article/pii/S1053811921006790)

![Whole Framework](https://www.dropbox.com/s/f729s5l2xvpwjfx/Figs_1.png?raw=1)

&nbsp;
&nbsp;
&nbsp;

# <span id="head2"> xQSM for QSM dipole inversion </span>
**xQSM: quantitative susceptibility mapping with octave convolutional and noise-regularized neural networks**

[source code (github)](https://github.com/sunhongfu/deepMRI/tree/master/xQSM) &nbsp;  | &nbsp;  [data & checkpoints (dropbox)](https://www.dropbox.com/sh/weps2m849svsh93/AAAAPqqKcLkL10Arqhy-3h_Fa?dl=0) &nbsp;  | &nbsp;  [arXiv (pre-print)](https://arxiv.org/abs/2004.06281) &nbsp;  | &nbsp;  [NMR in Biomed (full paper)](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/nbm.4461)

![Whole Framework](https://www.dropbox.com/s/bq7gsc540gy2kgc/Fig1.png?raw=1)

&nbsp;
&nbsp;
&nbsp;

# <span id="head1"> BFRnet for QSM background field removal </span>
**BFRnet: A deep learning-based MR background field removal method for QSM of the brain containing significant pathological susceptibility sources**

[source code (github)](https://github.com/sunhongfu/deepMRI/tree/master/BFRnet) &nbsp;  | &nbsp;  [data & checkpoints (dropbox)](https://www.dropbox.com/sh/q678oapc65evrfa/AADh2CGeUzhHh6q9t3Fe3fVVa?dl=0) &nbsp;  | &nbsp;  [arXiv (pre-print)](https://arxiv.org/abs/2204.02760) &nbsp;  | &nbsp;  

![Whole Framework](https://www.dropbox.com/s/fe408itfqdh61lx/Picture1.tif?raw=1)

&nbsp;
&nbsp;
&nbsp;
[⬆ top](#readme)

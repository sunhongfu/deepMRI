# xQSM – Quantitative Susceptibility Mapping with Octave Convolutional Neural Networks

- Major update, 19, March, 2025: We now have new and more user-friendly matlab wrappers for xQSM/xQSM+/iQSM+/iQSM/iQFM reconstuctions (with simpler syntaxes);  see our repo for deepMRI/iQSM_plus for more details. 
* This repository is for a octave convolutional neural network based QSM dipole inversion. 

- Both Matlab (2019a later) implementation and pytorch (1.4 or later) based implementation were provided in this repo. 
(2022.Nov.5. Corrected minor bugs in python version codes and checkpoints.) 

* This code was built and tested on Centos 7.8 with Nvdia Tesla V100 and windows 10/ubuntu 19.10 with GTX 1060. 

# Content
- [ Overview](#head1)
	- [(1) Overall Framework](#head2)
    - [(2) Representative Results](#head3)
- [ Manual](#head4)
	- [Requirements](#head5)
	- [Quick Start (on example data)](#head6)
	- [Reconstruction on your own data](#head7)
	- [Train new xQSM](#head8)

For a quick MATLAB demo for testing, download the repository, navigate MATLAB to the '**matlab/eval**' folder and run the script '**run_demo.m**'
This demo compares four different neural networks for QSM dipole inversion.
A COSMOS map is used as the groud truth/label for quantitative assessment.

# <span id="head1"> Overview </span>

## <span id="head2"> Overall Framework </span>

![Whole Framework](https://www.dropbox.com/s/bq7gsc540gy2kgc/Fig1.png?raw=1)
Fig. 1. Overview of the proposed xQSM method. The top row demonstrates the preparation process with the in vivo training datasets. Octave convolution is shown in the middle row, which introduces an X-shaped operation for communication between feature maps of different resolutions. Training input patches pass through a noise-adding layer (yellow) during each iteration step. The bottom row illustrates the xQSM network architecture based on the U-net backbone。 

# <span id="head2"> Representative Results </span>

![Representative Results](https://www.dropbox.com/s/qlb9b7wjlwipf90/Fig2.png?raw=1)
Fig. 2. Comparison of different QSM methods on 10 in vivo local field maps (five 0.6 mm isotropic from 7 T and five 1 mm isotropic from 3 T). Average QSM maps and DGM zoomed-in images are shown in the top four rows. Yellow arrows point to the apparent DGM susceptibility contrast loss with respect to the Calculation Of Susceptibility through Multiple Orientation Sampling (COSMOS). 

# <span id="head4"> Manual </span>

## <span id="head5"> Requirements </span>

    - Python 3.7 or later  
    - NVDIA GPU (CUDA 10.0)  
    - Anaconda Navigator (4.6.11) for Pytorch Installation
    - Pytorch 1.4 or later
    - MATLAB 2017b or later 

## <span id="head6"> Quick start on demo data </span>

we have provided two demo scripts in folder 'matlab/eval/run_demo.m' and 'python/eval/run_demo.ipynb' for matlab and pytorch version xQSM quick testing on our demo data, respectively 

the demo data are avaiale at https://www.dropbox.com/sh/weps2m849svsh93/AAAAPqqKcLkL10Arqhy-3h_Fa?dl=0 

## <span id="head7"> Reconstruction on your own data </span>

Prepare your data in NIFTI format, and replace them with the demo data. 

## <span id="head8"> Train new xQSM networks </span>

The training codes are provided in folder 'matlab/training' and 'python/training' for matlab and pytorch implementation, respectively 

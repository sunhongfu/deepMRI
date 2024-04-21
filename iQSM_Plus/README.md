# Plug-and-Play Latent Feature Editing for Orientation-Adaptive Quantitative Susceptibility Mapping Neural Networks

- This repository is for our iQSM+ method, which enables a direct QSM reconstruction from MRI raw phases acquired at arbitray orientations (https://doi.org/10.1016/j.media.2024.103160). 

* This code was built and tested on Win11 with RTX 4090, A4000, MacOS with M1 pro max, and a Centos 7.8 platform with Nvdia Tesla V100. 

# Content

- [ Overview](#head1)
  - [(1) Overall Framework](#head2)
  - [(2) Representative Results](#head3)
- [ Manual](#head4)
  - [Requirements](#head5)
  - [Quick Start](#head6)
  - [Q&A about z_prjs](#head6)

# <span id="head1"> Overview </span>

## <span id="head2">(1) Overall Framework </span>

![Whole Framework](https://github.com/sunhongfu/deepMRI/blob/master/iQSM_Plus/figs/fig1.png)
Fig. 1: The overall structure of the proposed (a) Orientation-Adaptive Neural Network, which is constructed by incorporating (b) Plug-and-Play Orientation-Adaptive Latent Feature Editing (OA-LFE) blocks onto conventional deep neural networks. The proposed OA-LFE can learn the encoding of acquisition orientation vectors and seamlessly integrate them into the latent features of deep networks.

## <span id="head3">(2) Representative Results </span>

![Representative Results](https://github.com/sunhongfu/deepMRI/blob/master/iQSM_Plus/figs/fig3.png)
Fig. 2: Comparison of the original iQSM, iQSM-Mixed, and the proposed iQSM+ methods on (a) two simulated brains with different acquisition orientations, and (b) four in vivo brains scanned at multiple 3T MRI platforms. 

# <span id="head4"> Manual </span>

## <span id="head5"> Requirements </span>

    - Python 3.7 or later
    - NVDIA GPU (CUDA 10.0)
    - Anaconda Navigator (4.6.11) for Pytorch Installation
    - Pytorch 1.8 or later
    - MATLAB 2017b or later
    - BET tool from FSL tool box

## <span id="head6"> Quick Start (on demo data) </span>

It is assume that all your data should be saved in NIFTI format (single or double type). 

Inputs: 
PhasePath: path for raw phase data;
params: reconstruction parameters including TE, vox, B0, and z_prjs;
MaskPath (optional): path for bet mask;
MagPath(optional): path for magnitude;
ReconDir (optional): path for reconstruction saving;

see more details in the matlab code

example usage:
```
    Recon_iQSM_plus('ph.nii', 'params.mat', './BET_mask.nii', 'mag.nii','./');
```

## <span id="head7"> How to calculate variable "z_prjs" </span>

suppose that you have read the dico_info from your dicom files with dicominfo.m (matlab func)
then:

```
% angles!!! (z projections)
Xz = dicom_info.ImageOrientationPatient(3);
Yz = dicom_info.ImageOrientationPatient(6);
Zz = sqrt(1 - Xz^2 - Yz^2);
Zxyz = cross(dicom_info.ImageOrientationPatient(1:3),dicom_info.ImageOrientationPatient(4:6));
Zz = Zxyz(3);
z_prjs = [Xz, Yz, Zz];
```


[⬆ top](#readme)

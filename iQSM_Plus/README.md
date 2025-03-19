# Plug-and-Play Latent Feature Editing for Orientation-Adaptive Quantitative Susceptibility Mapping Neural Networks

- This repository is for our iQSM+ method, which enables a direct QSM reconstruction from MRI raw phases acquired at arbitray orientations (https://doi.org/10.1016/j.media.2024.103160). 

- This code was built and tested on Win11 with RTX 4090, A4000, MacOS with M1 pro max, and a Centos 7.8 platform with Nvdia Tesla V100. 

- Major update, 19, March, 2025: We now have new and more user-friendly matlab wrappers for iQSM+/iQSM/iQFM/xQSM/xQSM+ reconstuctions; 

- minor Update: For windows users: You will have to run iQSM_fcns/ConfigurePython.m first; modify variable "pyExec" (default: 'C:\Users\CSU\anaconda3\envs\Pytorch\python.exe', % conda environment path (windows)),   update the path with yours;
- minor Update: see [Q&A about z_prjs](#head6) for how to calculate vairbal zprjs; 

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

It is assume that you have imported/converted your data input matlab matrices. 

The MATLAB function to run is 'iQSM_plus'. The function returns the iQSM+ result and saves its NIFTI format 'iQSM_plus.nii' and its matlab matrix 'iQSM.mat' in user-defined output folder (default location is current working directory)

Compulsory Inputs are:

1. phase: GRE (gradient echo) MRI phase data;
organized as a 3D (single-echo, e.g., a 256 x 256 x 128 numerical volume)
or 4D volume (multi-echo data,  e.g., a data volume of size 256 x 256 x 128 x 8);
2. TE: Echo Time; Here are two example inputs for
  i. a single-echo data: TE = 20 * 1e-3; (unit: seconds);
  ii. a n-echo data (1xn vector): TE = [4, 8, 12, 16, 20, 24, 28, ...] * 1e-3; (unit: seconds);

Optional Inputs are:

3. mag: magnitude data, which is a numerical volume of the same size as the
   phase input; default: ones;
4. mask: Brain Mask ROI, whose size is the same as the phase input (1-st
   echo); default: ones;
5. voxel_size: image resolution; default: [1 1 1] mm isotropic;
6. B0_dir: B0 field direction; the same as B0_dir in MEDI toolbox;
   default: [0 0 1] for pure axial head orientation;
7. B0: B0 field strength; detault: 3 (unit: Tesla);
8. eroded_rad: a radius for brain mask erosion control;
   default: 3 (3-voxel erosion);
9. output_dir: directory/folder for output of temporary and final results
   default: pwd (current working directory)
********************************

see more details in the matlab code

example usage:
```
QSM = iQSM_plus(phase, TE, 'mag', mag, 'mask', mask, 'voxel_size', [1, 1, 1], 'B0', 3, 'B0_dir', [0, 0, 1], 'eroded_rad', 3, 'output_dir', pwd);

```

for xQSM+ reconstruction, the syntax is similar:

```
QSM = xQSM_plus(lfs, 'mask', mask, 'B0_dir', [0, 0, 1], 'voxel_size', [1, 1, 1], 'output_dir', pwd);  % for xQSM;
```

## <span id="head7"> How to calculate variable "B0_dir" </span>

suppose that you have read the dico_info from your dicom files with dicominfo.m (matlab func)
then:

```
% angles!!! (z projections)
Xz = dicom_info.ImageOrientationPatient(3);
Yz = dicom_info.ImageOrientationPatient(6);
Zz = sqrt(1 - Xz^2 - Yz^2);
Zxyz = cross(dicom_info.ImageOrientationPatient(1:3),dicom_info.ImageOrientationPatient(4:6));
Zz = Zxyz(3);
B0_dir = [Xz, Yz, Zz];
```


[⬆ top](#readme)

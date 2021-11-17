%% This demo shows the complete reconstruction pipeline for iQSM on single-echo MRI phase data
%% Assume your raw phase data is in NIFTI format

% you can download demo data here: ******* seperate data and code *******
% github repo for deepMRI is here: https://github.com/sunhongfu/deepMRI


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set your own data paths and parameters
deepMRI_root = '~/Downloads/deepMRI-master'; % where deepMRI git repo is cloned to
PhasePath = '~/Downloads/DemonData_SingleEcho/ph_single_echo.nii';  % where raw phase data is (in NIFTI format)
ReconDir = '~/Downloads/demo_recon';  %% where to save reconstruction output
Eroded_voxel = 3;  %  set number of voxels for brain mask erosion; 0 means no erosion;
TE = 20e-3; % set Echo Time (in second)
B0 = 3; % set B0 field (in Tesla)
vox = [1 1 1]; % set voxel size a.k.a image resolution (in millimeter)

%% optional data paths to be set, simply comment out if not available
MaskPath = '~/Downloads/DemonData_SingleEcho/mask_single_echo.nii'; %% Path for brain mask which can be extracted by FSL-BET; this can be set to one.
%MagPath = '~/Downloads/DemonData_SingleEcho/mag_single_echo.nii'; %magnitude image; this can be set to one for single-echo data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% add MATLAB paths
addpath(genpath([deepMRI_root,'/iQSM/iQSM_fcns/']));  % add necessary utility function for saving data and echo-fitting;
addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;


%% 1. read in data
nii = load_nii(PhasePath);
phase = nii.img;

if ~ exist('MagPath','var') || isempty(MagPath)
    mag = ones(size(phase));
else
    nii = load_nii(MagPath);
    mag = nii.img;
end

if ~ exist('MaskPath','var') || isempty(MaskPath)
    mask = ones(size(phase));
else
    nii = load_nii(MaskPath);
    mask = nii.img;
end

%% mkdir for output folders
if ~exist(ReconDir, 'dir')
    mkdir(ReconDir)
end


[phase, pos] = ZeroPadding(phase, 16);
[mask, pos] = ZeroPadding(mask, 16);

%% 2. save all information (B0, TE, phase) as .mat file for Network Reconstruction
mask_eroded = Save_Input(phase, mask, TE, B0, Eroded_voxel, ReconDir);

%% Call Python script to conduct the reconstruction; use python API to run iQSM on the demo data
PythonRecon([deepMRI_root, '/iQSM/PythonCodes/Evaluation/Inference.py'], [ReconDir,'/Network_Input.mat'], ReconDir)

%% load reconstruction data and save as NIFTI
load([ReconDir,'/iQSM.mat']);
load([ReconDir,'/iQFM.mat']);

pred_chi = ZeroRemoving(pred_chi, pos);
pred_lfs = ZeroRemoving(pred_lfs, pos);
mask_eroded = ZeroRemoving(mask_eroded , pos);

nii = make_nii(pred_chi .* mask_eroded, vox);
save_nii(nii, [ReconDir,'/iQSM.nii']);

nii = make_nii(pred_lfs .* mask_eroded, vox);
save_nii(nii, [ReconDir,'/iQFM.nii']);







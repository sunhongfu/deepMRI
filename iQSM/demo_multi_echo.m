%% This demo shows the complete reconstruction pipeline for iQSM on Multi-echo MRI phase data
%% Assume your raw phase data is in NIFTI format

% you can download demo data here: ******* seperate data and code *******
% github repo for deepMRI is here: https://github.com/sunhongfu/deepMRI


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set your own data paths and parameters
deepMRI_root = '~/Downloads/deepMRI-master'; % where deepMRI git repo is cloned to
PhasePath = '~/Downloads/DemonData_MultiEcho/ph_multi_echo.nii';  % where raw phase data is (in NIFTI format)
ReconDir = '~/Downloads/demo_recon/';  %% where to save reconstruction output
Eroded_voxel = 0;  %  set number of voxels for brain mask erosion; 0 means no erosion;
TE = [ 0.0032, 0.0065, 0.0098, 0.0131, 0.0164, 0.0197, 0.0231, 0.0264]; % set Echo Time (in second)
B0 = 3; % set B0 field (in Tesla)
vox = [1 1 1]; % set voxel size a.k.a image resolution (in millimeter)

%% optional data paths to be set, simply comment out if not available
MaskPath = '~/Downloads/DemonData_MultiEcho/mask_multi_echo.nii'; %% Path for brain mask which can be extracted by FSL-BET; this can be set to one.
MagPath = '~/Downloads/DemonData_MultiEcho/mag_multi_echo.nii'; % magnitude image; this can be set to one.
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

for i = 1 : length(TE)
    
    %% 2. save all information (B0, TE, phase) as .mat file for Network Reconstruction echo by echo
    
    tmp_TE = TE(i);
    tmp_phase = phase(:,:,:,i);
    
    [tmp_phase, pos] = ZeroPadding(tmp_phase, 16);
    [mask, pos] = ZeroPadding(mask, 16);
    
    mask = Save_Input(tmp_phase, mask, tmp_TE, B0, Eroded_voxel, ReconDir);
    
    %% Call Python script to conduct the reconstruction; use python API to run iQSM on the demo data
    PythonRecon('./PythonCodes/Evaluation/Inference.py', [ReconDir, '/Network_Input.mat'], ReconDir)
    
    %% load reconstruction data
    rec_chi_path = [ReconDir,'iQSM.mat'];
    rec_lfs_path = [ReconDir,'iQFM.mat'];
    
    load(rec_chi_path);
    load(rec_lfs_path);
    
    pred_chi = ZeroRemoving(pred_chi, pos);
    pred_lfs = ZeroRemoving(pred_lfs, pos);
    mask = ZeroRemoving(mask, pos);
    
    chis(:,:,:,i) = TE(i) .* pred_chi;
    lfss(:,:,:,i) = TE(i) .* pred_lfs;
    
    clear tmp_phase; 
end

%% magnitude weighted echo-fitting and save as NIFTI

[chi_fitted, res] = echofit(chis, mag, TE);
nii = make_nii(chi_fitted .* mask, [1 1 1]);
save_nii(nii, [ReconDir, 'iQSM_echo_fitted.nii']);


[chi_fitted, res] = echofit(lfss, mag, TE);
nii = make_nii(chi_fitted .* mask, [1 1 1]);
save_nii(nii, [ReconDir, 'iQFM_echo_fitted.nii']);




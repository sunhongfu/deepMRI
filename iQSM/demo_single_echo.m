%% This demo shows the complete reconstruction pipeline for iQSM on single-echo MRI phase data
%% Assume your raw phase data is in NIFTI format


% (1) download or clone github repo for deepMRI: https://github.com/sunhongfu/deepMRI
% (2) download demo data and checkpoints here: https://www.dropbox.com/sh/9kmbytgf3jpj7bh/AACUZJ1KlJ1AFCPMIVyRFJi5a?dl=0


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set your own data paths and parameters
deepMRI_root = '~/Downloads/deepMRI'; % where deepMRI git repo is downloaded/cloned to
checkpoints  = '~/Downloads/iQSM_data/checkpoints';
PhasePath    = '~/Downloads/iQSM_data/demo/ph_single_echo.nii';  % where raw phase data is (in NIFTI format)
ReconDir     = '~/Downloads/iQSM_data/demo_recon';  %% where to save reconstruction output
Eroded_voxel = 3;  %  set number of voxels for brain mask erosion; 0 means no erosion
TE           = 20e-3; % set Echo Time (in second)
B0           = 3; % set B0 field (in Tesla)
vox          = [1 1 1]; % set voxel size a.k.a image resolution (in millimeter)
NetworkType  = 0; %% network type: 0 for original iQSM, 1 for networks trained with data fidelity, 
                  % 2 for networks trained with learnable Lap-Layer (15 learnable kernels) and data fidelity;  

%% optional mask path to be set, simply comment out if not available
MaskPath = '~/Downloads/iQSM_data/demo/mask_single_echo.nii'; %% brain mask; set to one will skip brain masking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% add MATLAB paths
addpath(genpath([deepMRI_root,'/iQSM/iQSM_fcns/']));  % add necessary utility function for saving data and echo-fitting;
addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;


%% 1. read in data
nii = load_nii(PhasePath);
phase = nii.img;

% interpolate the images to isotropic
imsize = size(phase);
imsize2 = round(imsize.*vox/min(vox));
vox2 = imsize.*vox/imsize2;
interp_flag = ~isequal(imsize,imsize2);

if interp_flag
    phase = angle(imresize3(exp(1j*phase),imsize2));
end

% load mask if available
if ~ exist('MaskPath','var') || isempty(MaskPath)
    mask = ones(size(phase));
else
    nii = load_nii(MaskPath);
    mask = nii.img;
    % interpolate the mask to isotropic
    if interp_flag
        mask = imresize3(mask,imsize2(1:3));
    end
end

% mkdir for output folders
if ~exist(ReconDir, 'dir')
    mkdir(ReconDir)
end

% zero padding to 16 dividable
[phase, pos] = ZeroPadding(phase, 16);
mask = ZeroPadding(mask, 16);


%% 2. save all information (B0, TE, phase) as .mat file for Network Reconstruction
mask_eroded = Save_Input(phase, mask, TE, B0, Eroded_voxel, ReconDir);

% Call Python script to conduct the reconstruction; use python API to run iQSM on the demo data

switch NetworkType 
    case 0
        InferencePath = [deepMRI_root, '/iQSM/PythonCodes/Evaluation/Inference.py']; 
    case 1
        InferencePath = [deepMRI_root, '/iQSM/PythonCodes/Evaluation/DataFidelityVersion/Inference.py'];
    case 2
        InferencePath = [deepMRI_root, '/iQSM/PythonCodes/Evaluation/LearnableLapLayer/Inference.py'];
end 

PythonRecon(InferencePath, [ReconDir,'/Network_Input.mat'], ReconDir, checkpoints);

%% load reconstruction data and save as NIFTI
load([ReconDir,'/iQSM.mat']);
load([ReconDir,'/iQFM.mat']);

pred_chi = ZeroRemoving(pred_chi, pos);
pred_lfs = ZeroRemoving(pred_lfs, pos);

if interp_flag

    nii = make_nii(pred_chi, vox2);
    save_nii(nii, [ReconDir,'/iQSM_interp.nii']);

    nii = make_nii(pred_lfs, vox2);
    save_nii(nii, [ReconDir,'/iQFM_interp.nii']);


    % back to original resolution if anisotropic
    pred_chi = imresize3(pred_chi,imsize);
    pred_lfs = imresize3(pred_lfs,imsize);

end

nii = make_nii(pred_chi, vox);
save_nii(nii, [ReconDir,'/iQSM.nii']);

nii = make_nii(pred_lfs, vox);
save_nii(nii, [ReconDir,'/iQFM.nii']);

delete([ReconDir,'/Network_Input.mat']);
delete([ReconDir,'/iQFM.mat']);
delete([ReconDir,'/iQSM.mat']);
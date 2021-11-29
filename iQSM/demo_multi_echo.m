%% This demo shows the complete reconstruction pipeline for iQSM on Multi-echo MRI phase data
%% Assume your raw phase data is in NIFTI format


% (1) download or clone github repo for deepMRI: https://github.com/sunhongfu/deepMRI
% (2) download demo data and checkpoints here: https://www.dropbox.com/sh/9kmbytgf3jpj7bh/AACUZJ1KlJ1AFCPMIVyRFJi5a?dl=0

clear 
clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data preparation guide: 

% 1. phase evolution type:
% The relationship between the phase data and filed pertubation (delta_B) 
% is assumed to satisfy the following equation: 
% "phase = -delta_B * gamma * TE" 
% Therefore, if your phase data is in the format of "phase = delta_B * gamma * TE;" 
% it will have to be preprocessed by multiplication by -1; 

% 2. For Ultra-high resolutin data:
% it is recommended that the phase data of ultra-high resolution (higher
% than 0.7 mm) should be interpoloated into 1 mm for better reconstruction results.  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set your own data paths and parameters
deepMRI_root = '~/Downloads/deepMRI'; % where deepMRI git repo is downloaded/cloned to
checkpoints  = '~/Downloads/iQSM_data/checkpoints';
PhasePath    = '~/Downloads/iQSM_data/demo/ph_multi_echo.nii';  % where raw phase data is (in NIFTI format)
ReconDir     = '~/Downloads/iQSM_data/demo_recon/';  %% where to save reconstruction output
Eroded_voxel = 0;  %  set number of voxels for brain mask erosion; 0 means no erosion;
TE           = [0.0032, 0.0065, 0.0098, 0.0131, 0.0164, 0.0197, 0.0231, 0.0264]; % set Echo Times (in second)
B0           = 3; % set B0 field (in Tesla)
vox          = [1 1 1]; % set voxel size a.k.a image resolution (in millimeter)
NetworkType  = 0; % network type: 0 for original iQSM, 1 for networks trained with data fidelity,
                  % 2 for networks trained with learnable Lap-Layer (15 learnable kernels) and data fidelity;

%% optional data paths to be set, simply comment out if not available
MaskPath = '~/Downloads/iQSM_data/demo/mask_multi_echo.nii'; %% brain mask; set to one will skip brain masking
MagPath = '~/Downloads/iQSM_data/demo/mag_multi_echo.nii'; % magnitude image; set to one will skip magnitude weights in echo fitting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% add MATLAB paths
addpath(genpath([deepMRI_root,'/iQSM/iQSM_fcns/']));  % add necessary utility function for saving data and echo-fitting;
addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;


%% 1. read in data
nii = load_nii(PhasePath);
phase = nii.img;

% interpolate the phase to isotropic
imsize = size(phase);
imsize2 = [round(imsize(1:3).*vox/min(vox)), imsize(4)];
vox2 = imsize(1:3).*vox/imsize2(1:3);
interp_flag = ~isequal(imsize,imsize2);

if interp_flag
    for echo_num = 1:imsize(4)
        phase2(:,:,:,echo_num) = angle(imresize3(exp(1j*phase(:,:,:,echo_num)),imsize2(1:3)));
    end
    phase = phase2;
    clear phase2
end

if ~ exist('MagPath','var') || isempty(MagPath)
    mag = ones(imsize2);
else
    nii = load_nii(MagPath);
    mag = nii.img;
    % interpolate the mag to isotropic
    if interp_flag
        for echo_num = 1:imsize(4)
            mag2(:,:,:,echo_num) = imresize3(mag(:,:,:,echo_num),imsize2(1:3));
        end
        mag = mag2;
        clear mag2
    end
end

if ~ exist('MaskPath','var') || isempty(MaskPath)
    mask = ones(imsize2(1:3));
else
    nii = load_nii(MaskPath);
    mask = nii.img;
    % interpolate the mask to isotropic
    if interp_flag
        mask = imresize3(mask,imsize2(1:3));
    end
end

%% mkdir for output folders
if ~exist(ReconDir, 'dir')
    mkdir(ReconDir)
end


[mask, pos] = ZeroPadding(mask, 16);

%% set inference.py path; 
switch NetworkType 
    case 0
        InferencePath = [deepMRI_root, '/iQSM/PythonCodes/Evaluation/Inference.py']; 
        checkpoints = [checkpoints, '/iQSM_and_iQFM'];
    case 1
        InferencePath = [deepMRI_root, '/iQSM/PythonCodes/Evaluation/DataFidelityVersion/Inference.py'];
        checkpoints = [checkpoints, '/iQSM_iQFM_DataFidelity']; 
    case 2
        InferencePath = [deepMRI_root, '/iQSM/PythonCodes/Evaluation/LearnableLapLayer/Inference.py'];
        checkpoints = [checkpoints, '/iQSM_learnableKernels']; 
end 

for echo_num = 1 : imsize(4)
    
    %% 2. save all information (B0, TE, phase) as .mat file for Network Reconstruction echo by echo
    tmp_TE = TE(echo_num);
    tmp_phase = phase(:,:,:,echo_num);
    
    tmp_phase = ZeroPadding(tmp_phase, 16);
    
    mask_eroded = Save_Input(tmp_phase, mask, tmp_TE, B0, Eroded_voxel, ReconDir);
    
    % Call Python script to conduct the reconstruction; use python API to run iQSM on the demo data
    PythonRecon(InferencePath, [ReconDir,'/Network_Input.mat'], ReconDir, checkpoints);
    
    %% load reconstruction data and save as NIFTI
    load([ReconDir,'/iQSM.mat']);
    load([ReconDir,'/iQFM.mat']);
    
    pred_chi = ZeroRemoving(pred_chi, pos);
    pred_lfs = ZeroRemoving(pred_lfs, pos);
    
    chi(:,:,:,echo_num) = TE(echo_num) .* pred_chi;
    lfs(:,:,:,echo_num) = TE(echo_num) .* pred_lfs;
    
    clear tmp_phase;
end



%% save results of all echoes before echo fitting
nii = make_nii(chi, vox2);
save_nii(nii, [ReconDir, 'iQSM_all_echoes.nii']);

nii = make_nii(lfs, vox2);
save_nii(nii, [ReconDir, 'iQFM_all_echoes.nii']);




%% magnitude weighted echo-fitting and save as NIFTI

chi_fitted = echofit(chi, mag, TE);
lfs_fitted = echofit(lfs, mag, TE);

if interp_flag
    
    nii = make_nii(chi_fitted, vox2);
    save_nii(nii, [ReconDir, 'iQSM_interp_echo_fitted.nii']);
    
    nii = make_nii(lfs_fitted, vox2);
    save_nii(nii, [ReconDir, 'iQFM_interp_echo_fitted.nii']);
    
    
    % back to original resolution if anisotropic
    chi_fitted = imresize3(chi_fitted,imsize(1:3));
    lfs_fitted = imresize3(lfs_fitted,imsize(1:3));
    
end

nii = make_nii(chi_fitted, vox);
save_nii(nii, [ReconDir,'/iQSM_echo_fitted.nii']);

nii = make_nii(lfs_fitted, vox);
save_nii(nii, [ReconDir,'/iQFM_echo_fitted.nii']);


delete([ReconDir,'/Network_Input.mat']);
delete([ReconDir,'/iQFM.mat']);
delete([ReconDir,'/iQSM.mat']);

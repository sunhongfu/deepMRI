%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This demo shows the complete reconstruction pipeline for DCRNet on single-channel MRI phase data
%% Assume your raw phase data is in .mat format

% (1) download or clone github repo for deepMRI: https://github.com/sunhongfu/deepMRI
% (2) download demo data and checkpoints here: https://www.dropbox.com/sh/p9k9rq8zux2bkzq/AADSgw3bECQ9o1dPpIoE5b85a?dl=0

clear
clc

% It is assumed that the kspace is saved as the size of ky * kz * kx [* echo_numbers], 
% where kx is the fully sampled readout direction. You probably need to permute your data!!!
% For our pretrained networks, the subsampling is supposed to happen in the coronal plane.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set your own data paths and parameters
deepMRI_root = '~/Downloads/deepMRI'; % where deepMRI git repo is downloaded/cloned to;
checkpoints  = '~/Downloads/DCRNet_data/checkpoints'; % where the network is stored;
ksp_path     = '~/Downloads/DCRNet_data/demo/single_channel/kspace_sub_AF4.mat';  % where the subsampled kspace data is (in ".mat" format)
ReconDir     = '~/Downloads/DCRNet_data/demo_recon/single_channel';  %% where to save reconstruction output
vox          = [1 1 1]; % voxel size;
AF           = 4; % accelerating factors. 4 or 8; set it consistent with your network;
dc_weights   = 1; % data consistency weights set between 0 to 1. e.g., 0 means no data consistency;
                  % rec_dc(k) = (1 - dc_weights) * rec(k) * mask + dc_weights * k_sub + (1 - mask) * rec(k);

% optional inputs
MaskPath     = '~/Downloads/DCRNet_data/demo/single_channel/mask_sub_AF4.mat'; %% subsampling mask;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% add MATLAB paths
addpath(genpath([deepMRI_root,'/DCRNet/DCRNet_fcns/']));  % add necessary utility function for saving data and echo-fitting;
addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load data
if ~ exist('ksp_path','var') || isempty(ksp_path)
    error('Please specify the kspace data path!')
else
    inp = load(ksp_path);
    f = fields(inp);
    ksp = inp.(f{1});  % load kspace data;
end

[n_ky, n_kz, n_kx, n_echo]  = size(ksp);

% load mask if available
if ~ exist('MaskPath','var') || isempty(MaskPath)
    disp('Please specify the subsampling mask path!')
    mask = abs(ksp(:,:,round(n_kx/2)),1) > 1e-9;
else
    inp = load(MaskPath);
    f = fields(inp);
    mask = inp.(f{1}); % load mask data;
end

%% mkdir for output folders
if ~exist(ReconDir, 'dir')
    mkdir(ReconDir)
end

%% reconstruction setting;
InferencePath = [deepMRI_root, '/DCRNet/PythonCodes/Evaluation/single_channel/Inference.py'];
switch AF
    case 4     
        network_path = [checkpoints, '/DCRNet_AF4.pth'];
    case 8
        network_path = [checkpoints, '/DCRNet_AF8.pth'];
end



for echo_num = 1 : n_echo
    
    temp_k = ksp(:,:,:,echo_num);
    
    %% save network inputs
    Amp_Nor_factors = Save_Input_Data_For_DCRNet(temp_k, mask, ReconDir, true);
    
    %% Call Python script to conduct the reconstruction;
    
    PythonRecon(InferencePath, [ReconDir,'/NetworkInput.mat'], ReconDir, network_path);
    
    %% load python reconstruction data;
    
    recon_r_path = [ReconDir,'/rec_real.mat'];
    recon_i_path = [ReconDir,'/rec_imag.mat'];
    
    load(recon_r_path);
    load(recon_i_path);
    
    
    %% postprocessing starts;
    recs = recons_r + 1j * recons_i;
        
    recs_new = Amp_Nor_factors * recs * 30; % inverse the amplitude normlization for each echo;
    
    mag(:,:,:,echo_num) = abs(recs_new);
    ph(:,:,:,echo_num) = angle(recs_new);

        
    recon_r_nodc_path = [ReconDir,'/rec_real_nodc.mat'];
    recon_i_nodc_path = [ReconDir,'/rec_imag_nodc.mat'];

    load(recon_r_nodc_path);
    load(recon_i_nodc_path);
    
    
    %% postprocessing starts;
    recs = recons_r + 1j * recons_i;
        
    recs_new = Amp_Nor_factors * recs * 30; % inverse the amplitude normlization for each echo;
    
    mag_nodc(:,:,:,echo_num) = abs(recs_new);
    ph_nodc(:,:,:,echo_num) = angle(recs_new);
end


nii = make_nii(mag, vox);
save_nii(nii, [ReconDir,'/mag_rec.nii']);
nii = make_nii(ph, vox);
save_nii(nii, [ReconDir,'/ph_rec.nii']);

nii = make_nii(mag_nodc, vox);
save_nii(nii, [ReconDir,'/mag_rec_nodc.nii']);
nii = make_nii(ph_nodc, vox);
save_nii(nii, [ReconDir,'/ph_rec_nodc.nii']);

delete([ReconDir,'/NetworkInput.mat']);
delete([ReconDir,'/rec_real.mat']);
delete([ReconDir,'/rec_imag.mat']);
delete([ReconDir,'/rec_real_nodc.mat']);
delete([ReconDir,'/rec_imag_nodc.mat']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% use iQSM for QSM reconstruction;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This demo shows the complete reconstruction pipeline for DCRNet on single-channel MRI phase data
%% Assume your raw phase data is in .mat format

% (1) download or clone github repo for deepMRI: https://github.com/sunhongfu/deepMRI
% (2) download demo data and checkpoints here: https://www.dropbox.com/sh/p9k9rq8zux2bkzq/AADSgw3bECQ9o1dPpIoE5b85a?dl=0

clear
clc

% It is assumed that the kspace is saved as the size of ky * kz * kx
% echo_numbers, where kx is the fully sampled readout direction. For our
% pretrained networks, the subsampling is supposed to happen in the coronal plane.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set your own data paths and parameters
deepMRI_root = '~/Downloads/deepMRI'; % where deepMRI git repo is downloaded/cloned to;
checkpoints  = '~/Downloads/DCRNet_data/checkpoints'; % where the network is stored;
ksp_path     = '~/Downloads/DCRNet_data/demo/kspace_example.mat';  % where the subsampled kspace data is (in ".mat" format)
ReconDir     = '~/Downloads/DCRNet_data/demo_recon';  %% where to save reconstruction output
MaskPath     = '~/Downloads/DCRNet_data/demo/Real_Mask_Acc4_256_by_128.mat'; %% subsampling mask; you can generate
%yours with the function "Gen_Sampling_Mask" in "MatlabCode" folder
vox          = [1 1 1]; % voxel size;
AF           = 4; % accelerating factors. 4 or 8; set it consistent with your network;
dc_weights   = 1;  % data consistency weights subject to [0, 1]. 0 means no data consistency;
% rec_dc(k) = (1 - dc_weights) * rec(k) * mask + dc_weights * k_sub + (1 - mask) * rec(k);
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

% load mask if available
if ~ exist('MaskPath','var') || isempty(MaskPath)
    disp('Please specify the subsampling mask path!')
    tmp = ksp(:,:,1);
    mask = tmp > 1e-9;
else
    % 3D laplacian kernel
    inp = load(MaskPath);
    f = fields(inp);
    mask = inp.(f{1}); % load mask data;
end

%% mkdir for output folders
if ~exist(ReconDir, 'dir')
    mkdir(ReconDir)
end

%% reconstruction setting;
switch AF
    case 4
        InferencePath = [deepMRI_root, '/DCRNet/PythonCodes/Evaluation/single_channel/Inference.py'];
        network_path = [checkpoints, '/DCRNet_AF4.pth'];
    case 8
        InferencePath = [deepMRI_root, '/DCRNet/PythonCodes/Evaluation/single_channel/Inference.py'];
        network_path = [checkpoints, '/DCRNet_AF8.pth'];
end

imsize  = size(ksp);

for echo_num = 1 : imsize(4)
    
    temp_k = ksp(:,:,:,echo_num);
    
    %% save network inputs
    Amp_Nor_factors = Save_Input_Data_For_DCRNet(temp_k, mask, ReconDir, true);
    
    %% Call Python script to conduct the reconstruction;
    
    PythonRecon(InferencePath, [ReconDir,'/NetworkInput.mat'], ReconDir, network_path);
    
    %% load python reconstruction data;
    
    recon_r_path = [ReconDir,'/rec_real.mat'];
    recon_i_path = [ReconDir,'/rec_imag.mat'];
    
    if ~dc_weights
        recon_r_path = [ReconDir,'/rec_real_nodc.mat'];
        recon_i_path = [ReconDir,'/rec_imag_nodc.mat'];
    end
    
    load(recon_r_path);
    load(recon_i_path);
    
    %% load amplitude normlaization factors;
    % load(['Amp_Nor_factors_', num2str(FileNo),'.mat']);
    
    %% postprocessing starts;
    recs = recons_r + 1j * recons_i;
    
    disp('PostProcessing Starts')
    
    recs_new = Amp_Nor_factors * recs * 30; % inverse the amplitude normlization for each echo;
    
    mag(:,:,:,echo_num) = abs(recs_new);
    ph(:,:,:,echo_num) = angle(recs_new);
end

nii = make_nii(mag, vox);
save_nii(nii, [ReconDir,'/mag_rec.nii']);

nii = make_nii(ph, vox);
save_nii(nii, [ReconDir,'/ph_rec.nii']);

delete([ReconDir,'/NetworkInput.mat']);
delete([ReconDir,'/rec_real.mat']);
delete([ReconDir,'/rec_imag.mat']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% use iQSM for QSM reconstruction;



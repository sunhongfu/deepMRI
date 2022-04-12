
clear 
clc 

% requires MATLAB deep learning toolbox
% (1) download or clone github repo for deepMRI: https://github.com/sunhongfu/deepMRI
% (2) download demo data and checkpoints here: https://www.dropbox.com/sh/q678oapc65evrfa/AADh2CGeUzhHh6q9t3Fe3fVVa?dl=0

deepMRI_root = '~/Downloads/deepMRI'; % where deepMRI git repo is cloned to; change as yours;
ReconDir     = '~/Downloads/BFRnet_data/demo_recon/';  %% where to save reconstruction output
checkpoints = '~/Downloads/BFRNet_data/checkpoints/BFRnet_L2_64PS_24BS_45Epo_NewHCmix.mat';  %% give your network path here. 

addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;
addpath(genpath([deepMRI_root,'/BFRnet/Eval']));

nii = load_nii('~/Downloads/BFRNet_data/demo/tfs.nii'); % load the total field map here, and replace the file name with yours.
tfs = double(nii.img);
mask = tfs ~= 0; % brain tissue mask

% note the size of the field map input needs to be divisibel by 8
% otherwise 0 padding should be done first
imSize = size(tfs);
if mod(imSize, 8)
    [tfs, pos] = ZeroPadding(tfs, 8);
end

%% Load the BFRnet and process reconstruction
if canUseGPU()
    [bkg] = MyPredictGPU(tfs, checkpoints); % Recon using GPU
else
    [bkg] = MyPredictCPU(tfs, checkpoints); % Recon using CPU
end

bkg = double(bkg .* mask); % The reconstructio  result is background field map.
lfs = tfs .* mask - bkg;

%% mkdir for output folders
if ~exist(ReconDir, 'dir')
    mkdir(ReconDir)
end

nii = make_nii(double(lfs));
save_nii(nii, [ReconDir '/lfs_BFRnet.nii']);

%% Evaluation, use default pnsr and ssim
% Load ground truth of local field
nii = load_nii('~/Downloads/BFRNet_data/demo/lfs_label.nii');
lfs_label = double(nii.img);

error = lfs - lfs_label;
nii = make_nii(error);
save_nii(nii, [ReconDir '/error_BFRnet.nii']);

lfs_msk = lfs(lfs~=0);
lfs_ver = lfs_msk(:);  % mask the zero-region

lfs_label_msk = lfs_label(lfs_label~=0);
lfs_label_ver = lfs_label_msk(:);

PSNR_recon = psnr(lfs_ver, lfs_label_ver);
fprintf('PSNR of local field recon is %f\n', PSNR_recon);
SSIM_recon = ssim(lfs_ver, lfs_label_ver);
fprintf('SSIM of local field recon is %f\n',SSIM_recon);

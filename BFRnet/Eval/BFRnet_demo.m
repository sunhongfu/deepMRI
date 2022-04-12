
clear 
clc 

%% add NIFTI matlab toolbox for read and write NIFTI format
deepMRI_root = '~/Downloads/deepMRI-master'; % where deepMRI git repo is cloned to; change as yours;
addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;


NetPath = 'BFRnet.mat';  %% give your network path here. 

nii = load_nii('../../tfs.nii'); % load the total field map here, and replace the file name with yours.
tfs = double(nii.img);
mask = tfs ~= 0; % brain tissue mask

% note the size of the field map input needs to be divisibel by 8
% otherwise 0 padding should be done first
imSize = size(tfs);
if mod(imSize, 8)
    [tfs, pos] = ZeroPadding(tfs, 8);
end

%% Load the BFRnet and process reconstruction

add path '../' % load the network  %%%%%%%%%%%ZXY

if canUseGPU()
    [bkg] = MyPredictGPU(tfs, NetPath); % Recon using GPU
else
    [bkg] = MyPredictCPU(tfs, NetPath); % Recon using CPU
end

bkg = double(bkg .* mask); % The reconstructio  result is background field map.
lfs = tfs .* mask - bkg;

nii = make_nii(double(lfs));
save_nii(nii, '../lfs_BFRnet.nii');

%% Evaluation, use default pnsr and ssim
% Load ground truth of local field
load_untouch_nii('../../lfs_label.nii');
lfs_label = ans.img;

Errormap = lfs_BFRnet - lfs_label;
nii = make_nii(Errormap);
save_nii(nii, '../../Errormap_BFRnet.nii');

PSNR_recon = psnr(QSM_recon, single(lfs_label));
fprintf('PSNR of %s is %f\n', PSNR_recon);
SSIM_recon = ssim(QSM_recon, single(lfs_label));
fprintf('SSIM of %s is %f\n\n',SSIM_recon);

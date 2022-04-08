%% navigate MATLAB to the 'eval' folder
% cd('~/deepMRI/xQSM/matlab/eval');

%% add NIFTI matlab toolbox for read and write NIFTI format
deepMRI_root = '~/Downloads/deepMRI-master'; % where deepMRI git repo is cloned to; change as yours; 
addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;

nii = load_nii('../../total_field_input.nii'); % replace the file name with yours. 
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
[bkg] = MyPredictCPU(tfs); % Recon using CPU

% [bkg] = MyPredictGPU_New(tfs); % Recon using GPU

bkg = double(bkg .* mask_input); % The reconstructio  result is background field map.
lfs = tfs .* mask - bkg;

nii = make_nii(double(lfs));
save_nii(nii, '../lfs_BFRnet.nii');

%% Evaluation, use default pnsr and ssim 
% Load ground truth of local field
load_untouch_nii('../../lfs_label.nii');
lfs_label = ans.img;

% If there's no lfs_label, we could generate from COSMOS using forward calculation.
% load_untouch_nii('../../cosmos.nii');
% qsm_label = ans.img;
% lfs = forward_field_calc_HS(qsm_label);

PSNR_recon = psnr(QSM_recon, single(lfs_label));
fprintf('PSNR of %s is %f\n', PSNR_recon);
SSIM_recon = ssim(QSM_recon, single(lfs_label));
fprintf('SSIM of %s is %f\n\n',SSIM_recon);

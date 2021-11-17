%% navigate MATLAB to the 'eval' folder
% cd('~/deepMRI/xQSM/matlab/eval');

%% add NIFTI matlab toolbox for read and write NIFTI format
deepMRI_root = '~/Downloads/deepMRI-master'; % where deepMRI git repo is cloned to; change as yours; 
addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;

%% read in field map and COSMOS map (3rd dimension is z/B0 direction)
nii = load_nii('../../field_input.nii'); % replace the file name with yours. 
field = double(nii.img);
mask = field ~= 0; % brain tissue mask

% note the size of the field map input needs to be divisibel by 8
% otherwise 0 padding should be done first
imSize = size(field);
if mod(imSize, 8)
    [field, pos] = ZeroPadding(field, 8);
end

% illustration of one central axial slice of the input field 
figure;
imagesc(field(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.05, 0.05]);
title('Slice 80 of the Input Field Map (ppm)');
drawnow;

%% read label (for evaluation purpose)
nii = load_nii('../../cosmos_label.nii'); % replace the file name with yours. 
label = double(nii.img);

%% label image normalization (mean of brain tissue region set to 0) for later comparison;
label = label - sum(label(:)) / sum(mask(:));
label = label .* mask;

% illustration of one central axial slice of the COSMOS label 
figure; 
imagesc(label(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2]);
title('Slice 80 of the COSMOS Label (ppm)');
drawnow;

%% start recons
recon_methods_list = {'xQSM_invivo', 'xQSM_syn', 'xQSM_invivo_withNoiseLayer', 'Unet_invivo', 'Unet_syn'};

for k = 1:length(recon_methods_list)
    recon_method = recon_methods_list{k};
    fprintf('Reconstructing QSM using %s\n', recon_method);

    if canUseGPU()
        % (1) if your MATLAB is configured with CUDA GPU acceleration
        QSM_recon = Eval(field, recon_method, 'gpu');
    else
        % (2) otherwise if CUDA is not available, use CPU instead, this is much slower
        QSM_recon = Eval(field, recon_method, 'cpu');
    end

    % if zeropadding was performed, then do zero-removing before next step;
    if mod(imSize, 8)
        QSM_recon = ZeroRemoving(QSM_recon, pos);
    end

    %% image normalization (mean of brain tissue region set to 0)
    QSM_recon = QSM_recon - sum(QSM_recon(:)) / sum(mask(:));
    QSM_recon = QSM_recon .* mask; 

    %% illustration of one central axial slice of the four different reconstructions; 
    figure,
    subplot(121), imagesc(QSM_recon(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
    title(['Slice 80 of the ', recon_method], 'Interpreter', 'none');
    err  = QSM_recon - label;
    subplot(122), imagesc(err(:,:,80)'); colormap gray; axis equal tight; colorbar; caxis([-0.1, 0.2])
    title('Error');
    drawnow;
    
    %% use default pnsr and ssim 
    PSNR_recon = psnr(QSM_recon, single(label));
    fprintf('PSNR of %s is %f\n', recon_method, PSNR_recon);
    SSIM_recon = ssim(QSM_recon, single(label));
    fprintf('SSIM of %s is %f\n\n', recon_method, SSIM_recon);

    %% save the files for ROI measurements; 
    nii = make_nii(QSM_recon, [1, 1, 1]);
    save_nii(nii, ['Chi_', recon_method, '.nii']);
end


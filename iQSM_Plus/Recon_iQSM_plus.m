function Recon_iQSM_plus(PhasePath, paramspath, MaskPath, MagPath, ReconDir)
%% iQSM-series reconstruction 
% this is a universal matlab API for our xQSM and iQFM, iQSM, and iQSM+ recons. 
% Inputs: 
% PhasePath: path for raw phase data;
% params: reconstruction parameters including TE, vox, B0, and z_prjs;
% MaskPath (optional): path for bet mask;
% MagPath(optional): path for magnitude;
% ReconDir (optional): path for reconstruction saving;

% example usage: single-step (iQSM series): Recon_iQSM_series('ph.nii', 'params.mat', './mask.nii', './mag.nii', './'); 
% Dipole inversion (xQSM series): Recon_iQSM_series('lfs.nii', 'params.mat', './mask.nii', '', './'); 

%------------- Assume all your data is in NIFTI format--------------------%
%
% for more deep learning based algorithms for background removal and dipole
% inversion, plese
% (1) download or clone github repo for deepMRI: https://github.com/sunhongfu/deepMR
% (2) download demo data and checkpoints here: https://www.dropbox.com/sh/9kmbytgf3jpj7bh/AACUZJ1KlJ1AFCPMIVyRFJi5a?dl=0
%
% for more conventional algorithms, e.g., phase combination, phase unwrapping, please
% download or clone github repo for Hongfu's QSM toolbox: https://github.com/sunhongfu/QSM

%  Authors: Yang Gao [1,2]
%  yang.gao@csu.edu.cn / yang.gao@uq.edu.au
%  [1]: Central South University, China, Lecturer
%  [2]: University of Queensland, Australia, Honorary Fellow
%  22 Mar, 2023

%  Reference:
%  [1] Gao Y, et al. Instant tissue field and magnetic susceptibility mapping
%  from MRI raw phase using Laplacian enhanced deep neural networks. Neuroimage. 2022
%  doi: 10.1016/j.neuroimage.2022.119410. Epub 2022 Jun 23. PMID: 35753595.
%  [2] Gao, Y, Zhu, X, Moffat, BA, et al. xQSM: quantitative susceptibility mapping 
%  with octave convolutional and noise-regularized neural networks. 
%  NMR in Biomedicine. 2021; 34:e4461. https://doi.org/10.1002/nbm.4461. 


%------------------- data preparation guide ------------------------------%

% 1. phase evolution type:
% The relationship between the phase data and filed pertubation (delta_B)
% is assumed to satisfy the following equation:
% "phase = -delta_B * gamma * TE"
% Therefore, if your phase data is in the format of "phase = delta_B * gamma * TE;"
% it will have to be preprocessed by multiplication by -1;

% created 11.08, 2022
% last modified 01.25, 2022
% latest 0506 2023

if ~exist('PhasePath','var') || isempty(PhasePath)
    error('Please input the path for raw phase data!')
end

if ~exist('ReconDir','var') || isempty(ReconDir)
    ReconDir = dir('./').folder;  %% where to save reconstruction output
else
    %% mkdir for output folders
    if ~exist(ReconDir, 'dir')
        mkdir(ReconDir)
    end

    ReconDir = dir(ReconDir).folder;
end

for NetType = 2
    %% Set your own data paths and parameters
    deepMRI_root = '~/deepMRI'; % where deepMRI git repo is downloaded/cloned to
    CheckPoints_folder = '~/deepMRI/iQSM_Plus/PythonCodes/Evaluation/checkpoints';
    PyFolder = '~/deepMRI/iQSM_Plus/PythonCodes/Evaluation/iQSM_series';

    switch NetType
        case 0 %% original iqsm
            KeyWord = 'iQSM_original';
            checkpoints  = sprintf('%s/%s/', CheckPoints_folder ,KeyWord);
            InferencePath = sprintf('%s/%s/Inference_iQSMSeries.py', PyFolder, KeyWord);

        case 1 %% original iqfm
            KeyWord = 'iQFM_original';
            checkpoints  = sprintf('%s/%s/', CheckPoints_folder ,KeyWord);
            % checkpoints  = sprintf('%s/%s_old/', CheckPoints_folder ,KeyWord);
            InferencePath = sprintf('%s/%s/Inference_iQFM.py', PyFolder, KeyWord);

        case 2 %% iQSM+_v1
            KeyWord = 'iQSM_plus_v1';
            checkpoints  = sprintf('%s/%s/', CheckPoints_folder ,KeyWord);
            % checkpoints  = sprintf('%s/%s_old/', CheckPoints_folder ,KeyWord);
            InferencePath = sprintf('%s/%s/Inference_iQSMSeries.py', PyFolder, KeyWord);

        case 3 %% xQSM-original
            KeyWord = 'xQSM_original';
            checkpoints  = sprintf('%s/%s/', CheckPoints_folder , KeyWord);
            % checkpoints  = sprintf('%s/%s_old/', CheckPoints_folder ,KeyWord);
            InferencePath = sprintf('%s/%s/Inference_xQSMSeries.py', PyFolder, KeyWord);

        case 4 %% xQSM-original
            KeyWord = 'xQSM_plus_v1';
            checkpoints  = sprintf('%s/%s/', CheckPoints_folder , KeyWord);
            % checkpoints  = sprintf('%s/%s_old/', CheckPoints_folder ,KeyWord);
            InferencePath = sprintf('%s/%s/Inference_xQSMSeries.py', PyFolder, KeyWord);
    end

    if ~exist('paramspath','var') || isempty(paramspath)
        error('Please input the params file!')
    end

    load(paramspath);
    %B0 = 3;
    ori_vox = vox;
    z_prjs = z_prjs / norm(z_prjs); 
    Eroded_voxel = 3; %% brain erosion for 3 voxels, or 0 for whole head recon;

    %% add MATLAB paths
    addpath(genpath([deepMRI_root,'/iQSM/iQSM_fcns/']));  % add necessary utility function for saving data and echo-fitting;
    addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;

    %% read data

    %% 1. read in data
    sf = -1;   %% for cooridinates mismatch; set as -1 for ChinaJapan data;
    nii = load_nii(PhasePath);
    phase = single(nii.img);

    if NetType > 2
        sf = 1;
    end 

    phase = sf * phase;

    %     phase = phase(:, end:-1:1, :, :);

    % interpolate the phase to isotropic
    imsize = size(phase);
    if length(imsize) == 3
        imsize(4) = 1;
    end
    imsize2 = [round(imsize(1:3).*vox/min(vox)), imsize(4)];

    vox2 = imsize(1:3).*vox ./ imsize2(1:3);

    vox2 = round(vox2 * 100) / 100; %% only keep 2 floating points precesion;

    vox = vox2;

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
        mag = single(nii.img);
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
        mask = single(nii.img);
        % interpolate the mask to isotropic
        if interp_flag
            mask = imresize3(mask,imsize2(1:3));
        end
    end

    %% mkdir for output folders
    if ~exist(ReconDir, 'dir')
        mkdir(ReconDir)
    end

    %% recon starts;

    permute_flag = 0; 
    if (abs(z_prjs(3)) < abs(z_prjs(2)))

        % z_prjs = permute(z_prjs, [1, 3, 2])
        z_prjs2 = z_prjs; 
        z_prjs(2) = z_prjs2(3); 
        z_prjs(3) = z_prjs2(2)
        phase = permute(phase, [1, 3, 2, 4]);
        mask = permute(mask, [1, 3, 2, 4]);
        mag = permute(mag, [1, 3, 2, 4]);


        permute_flag = 1; 

    end

    tmp_phase = ZeroPadding(phase, 16);
    [mask, pos] = ZeroPadding(mask, 16);

    mask_eroded = Save_Input_iQSMplus(tmp_phase, mask, TE', B0, Eroded_voxel, z_prjs, vox, ReconDir);

    % Call Python script to conduct the reconstruction; use python API to run iQSM on the demo data
    PythonRecon(InferencePath, [ReconDir,filesep,'Network_Input.mat'], ReconDir, checkpoints);

    %% load reconstruction data and save as NIFTI
    load([ReconDir,'/iQSM.mat']);
    % pred_chi = pred_lfs;

    pred_chi = squeeze(pred_chi);
    if length(size(pred_chi)) ~= 3
        pred_chi = permute(pred_chi, [2, 3, 4, 1]);
    end

    chi = ZeroRemoving(pred_chi, pos);
    clear tmp_phase;

    %% save results of all echoes before echo fitting
    nii = make_nii(chi, vox2);
    save_nii(nii, [ReconDir, filesep, 'iQSM_all_echoes.nii']);

    %% magnitude weighted echo-fitting and save as NIFTI

    if imsize(4) > 1
        for echo_num = 1 : imsize(4)
            chi(:,:,:,echo_num) = TE(echo_num) .* chi(:,:,:,echo_num);
        end

        chi_fitted = echofit(chi, mag, TE);
    else
        chi_fitted = chi;
    end

    if interp_flag

        nii = make_nii(chi_fitted, vox2);
        save_nii(nii, [ReconDir, 'iQSM_interp_echo_fitted.nii']);

        % back to original resolution if anisotropic
        chi_fitted = imresize3(chi_fitted,imsize(1:3));
        % chi_fitted = imresize3(chi_fitted,ori_imsize(1:3));
    end

    if permute_flag

        chi_fitted = permute(chi_fitted, [1, 3, 2]); 

    end


    nii = make_nii(chi_fitted, ori_vox);
    save_nii(nii, [ReconDir,'/iQSM_plus.nii']);

    delete([ReconDir,'/Network_Input.mat']);
    delete([ReconDir,'/iQSM.mat']);

end
end
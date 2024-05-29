function QSM = iQSM_plus(phase, TE, varargin)
%------------------- iQSM+ reconstruction manual ------------------------------%
% this is a latest version of the matlab API for our iQSM+ method;
%
% Example Usage:
% *********************************************************************************
% QSM = iQSM_plus(phase, TE, ...
% 'mag', mag, 'mask', mask, ...
% 'voxel_size', [1, 1, 1], ...
% 'B0', 3, 'B0_dir', [0, 0, 1], ...
% 'eroded_rad', 3);
%
% where Compulsory Inputs are:
% 1. phase: GRE (gradient echo) MRI phase data;
% organized as a 3D (single-echo, e.g., a 256x256x128 numerical volume)
% or 4D volume (multi-echo data,  e.g., a data volume of size 256x256x128x8);
% 2. TE: Echo Time; Here are two example inputs for
%   i. a single-echo data: TE = 20 * 1e-3; (unit: seconds);
%   ii. a n-echo data (1xn vector): TE = [4, 8, 12, 16, 20, 24, 28, ...] * 1e-3; (unit: seconds);
%
% Optional Inputs are:
% 3. mag: magnitude data, which is a numerical volume of the same size as the
%    phase input; default: ones;
% 4. mask: Brain Mask ROI, whose size is the same as the phase input (1-st
%    echo); default: ones;
% 5. voxel_size: image resolution; default: [1 1 1] mm isotropic;
% 6. B0_dir: B0 field direction; the same as B0_dir in MEDI toolbox;
%    default: [0 0 1] for pure axial head orientation;
% 7. B0: B0 field strength; detault: 3 (unit: Tesla);
% 8. eroded_rad: a radius for brain mask erosion control;
%    default: 3 (3-voxel erosion);
% *********************************************************************************
%
% Please cite:
% [1] Gao Y, et al. Plug-and-Play latent feature editing for orientation-adaptive
% quantitative susceptibility mapping neural networks. Medical Image
% Analysis, 2024. doi: https://doi.org/10.1016/j.media.2024.103160
% [2] Gao Y, et al. Instant tissue field and magnetic susceptibility mapping
% from MRI raw phase using Laplacian enhanced deep neural networks. Neuroimage. 2022
% doi: https://doi.org/10.1016/j.neuroimage.2022.119410
% [3] Gao, Y, Zhu, X, Moffat, BA, et al. xQSM: quantitative susceptibility mapping
% with octave convolutional and noise-regularized neural networks.
% NMR in Biomedicine. 2021; 34:e4461. https://doi.org/10.1002/nbm.4461.
%
% Author(s): Yang Gao [1,2], Hongfu Sun[2,3]
% yang.gao@csu.edu.cn / yang.gao@uq.edu.au / hongfu.sun@uq.edu.au
% [1]: Central South University, China
% [2]: the University of Queensland, Australia
% [3]: the University of NewCastle, Australia
%
%------------------- Phase Evolution Type Notice ------------------------------%
%
% 1. phase evolution type:
% The relationship between the phase data and filed pertubation (delta_B)
% is assumed to satisfy the following equation:
% "phase = -delta_B * gamma * TE"
% Therefore, if your phase data is in the format of "phase = delta_B * gamma * TE;"
% it will have to be preprocessed by multiplication by -1;
%
%------------------- Phase Evolution Type Notice ends --------------------------%
%
% For more deep learning based algorithms for background removal and dipole
% inversion, plese
% (1) download or clone github repo for deepMRI: https://github.com/sunhongfu/deepMR
%
% For more conventional algorithms, e.g., phase combination, phase unwrapping, please
% download or clone github repo for Hongfu's QSM toolbox: https://github.com/sunhongfu/QSM


% created 08.11, 2022
% last modified 25.08, 2023
% latest version: 17.05, 2024

ReconDir = './'; % path for temporary files;

    
%% Set your own data paths and parameters
deepMRI_root = '~/deepMRI'; % where deepMRI git repo is downloaded/cloned to
CheckPoints_folder = '~/deepMRI/iQSM_Plus/PythonCodes/Evaluation/checkpoints';
PyFolder = '~/deepMRI/iQSM_Plus/PythonCodes/Evaluation/iQSM_series';
% KeyWord = 'iQSM_plus_v1';
KeyWord = 'iQSM_plus_v2';

checkpoints  = sprintf('%s/%s/', CheckPoints_folder ,KeyWord);
% checkpoints  = sprintf('%s/%s_old/', CheckPoints_folder ,KeyWord);
InferencePath = sprintf('%s/%s/Inference_iQSMSeries.py', PyFolder, KeyWord);

%% add MATLAB paths
addpath(genpath([deepMRI_root,'/iQSM_Plus/iQSM_fcns/']));  % add necessary utility function for saving data and echo-fitting;
addpath(genpath([deepMRI_root,'/utils']));  %  add NIFTI saving and loading functions;

if ~exist('phase','var') || isempty(phase)

    cprintf('*red', 'Phase data input is missing! \n');
    cprintf('-[0, 128, 19]',['The phase input data should be a 3-D (single-echo data, e.g., a 256x256x128 volume) \n ' ...
        'or 4-D (multi-echo data: 3D image * echo_num, e.g., 256x256x128x8) volume \n'])
    cprintf('*red', 'Key Parameter Missing, Ends with Error! \n');
    error(' ');
end

if ~exist('TE','var') || isempty(TE)

    cprintf('*red', ['Missing Echo Time input! \n' ...
        'Echo Time vector is necessary for reconstruction! \n']);
    cprintf('-[0, 128, 19]',['Here are two example inputs for \n' ...
        '1. a single-echo data: TE = 20 * 1e-3 (unit: seconds) \n'])
    cprintf('-[0, 128, 19]','2. a n-echo data (1xn vector): TE = [4, 8, 12, 16, 20, 24, 28, ...] * 1e-3 (unit: seconds) \n')

    cprintf('*red', 'Key Parameter Missing, Ends with Error! \n');
    error(' ');
end

disp(' ')
cprintf('*[0, 0, 0]', '------------- Extracting optional parameters for reconstruction --------------------\n');
disp(' ')

[mag, mask, vox, z_prjs, B0, eroded_rad] = parse_iQSM_inputs(size(phase), varargin{:});


ori_vox = vox;
z_prjs = z_prjs / norm(z_prjs);

if size(phase, 4) > 1
    disp(sprintf('Phase is a 4D multi-echo data of size %dx%dx%dx%d',size(phase,1),size(phase,2),size(phase,3), size(phase,4)));
else 
    disp(sprintf('Phase is a 3D single-echo data of size %dx%dx%d',size(phase,1),size(phase,2),size(phase,3)));
end
disp(sprintf('Mask is a numerical volume of size %dx%dx%d', size(mask, 1),size(mask, 2),size(mask, 3)));
disp(sprintf('voxel_size = [%s, %s, %s] mm', num2str(vox(1)), num2str(vox(2)), num2str(vox(3))));
disp(sprintf('B0_dir = [%s, %s, %s]', num2str(z_prjs(1)), num2str(z_prjs(2)),num2str(z_prjs(3))));
disp(['B0 field strength = ', num2str(B0), '(T)']);
disp(['eroded_rad = ', num2str(eroded_rad)]);

te_str = [];

for ii = 1 : size(phase,4)
    te_str=[te_str, num2str(TE(ii)), ' '];
end

disp(['TE = [', te_str, '] (s)'])

disp(' ')
cprintf('*[0, 0, 0]', '------------- Optional Parameters Extracted Successfully! --------------------------\n');
disp(' ')

cprintf('*[0, 0, 0]', 'Saving all data as NetworkInput.mat for Pytorch Recon! \n ... \n');

%% 1. save all the data into a NetworkInput.mat file.
sf = 1;   %% for cooridinates mismatch;
phase = single(phase);
phase = sf * phase;

% interpolate the phase to isotropic
imsize = size(phase);
if length(imsize) == 3
    imsize(4) = 1;
end
imsize2 = [round(imsize(1:3).*vox/min(vox)), imsize(4)];

vox2 = imsize(1:3).*vox ./ imsize2(1:3);

vox2 = round(vox2 * 100) / 100; %% only keep 2 floating points precesion;

vox = vox2;

% interpolate the ph to isotropic if necessary
interp_flag = ~isequal(imsize,imsize2);

if interp_flag
    for echo_num = 1:imsize(4)
        phase2(:,:,:,echo_num) = angle(imresize3(exp(1j*phase(:,:,:,echo_num)),imsize2(1:3)));
    end
    phase = phase2;
    clear phase2
end

% interpolate the mag to isotropic
if interp_flag
    for echo_num = 1:imsize(4)
        mag2(:,:,:,echo_num) = imresize3(mag(:,:,:,echo_num),imsize2(1:3));
    end
    mag = mag2;
    clear mag2
end

% interpolate the mask to isotropic
if interp_flag
    mask = imresize3(mask,imsize2(1:3));
end

%% data permutation if necessary;

permute_flag = 0;
if (abs(z_prjs(3)) < abs(z_prjs(2)))
    % z_prjs = permute(z_prjs, [1, 3, 2])
    z_prjs2 = z_prjs;
    z_prjs(2) = z_prjs2(3);
    z_prjs(3) = z_prjs2(2);
    phase = permute(phase, [1, 3, 2, 4]);
    mask = permute(mask, [1, 3, 2, 4]);
    mag = permute(mag, [1, 3, 2, 4]);
    permute_flag = 1;
end

tmp_phase = ZeroPadding(phase, 16);
[mask, pos] = ZeroPadding(mask, 16);

[~] = Save_Input_iQSMplus(tmp_phase, mask, TE', B0, eroded_rad, z_prjs, vox, ReconDir);

cprintf('*[0, 0, 0]', 'Network Input File generated successfully! \n');

disp(' ')

cprintf('*[0, 0, 0]', 'Pytorch Reconstruction Starts! \n  \n');

% Call Python script to conduct the reconstruction; use python API to run iQSM on the demo data
PythonRecon(InferencePath, [ReconDir,filesep,'Network_Input.mat'], ReconDir, checkpoints);

cprintf('*[0, 0, 0]', 'Reconstruction Ends! \n');
disp(' ')

cprintf('*[0, 0, 0]', 'Postprocessing (e.g., echo fitting) Starts! \n ... \n');

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
% comment these codes if you dont want to save temporary results;
%     nii = make_nii(chi, vox2);
%     save_nii(nii, [ReconDir, filesep, 'iQSM_all_echoes.nii']);

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
    % comment these codes if you dont want to save temporary results;
    %         nii = make_nii(chi_fitted, vox2);
    %         save_nii(nii, [ReconDir, filesep, 'iQSM_interp_echo_fitted.nii']);

    % back to original resolution if anisotropic
    chi_fitted = imresize3(chi_fitted,imsize(1:3));
    % chi_fitted = imresize3(chi_fitted,ori_imsize(1:3));
end

if permute_flag

    chi_fitted = permute(chi_fitted, [1, 3, 2]);

end

QSM = chi_fitted;

delete([ReconDir,'/Network_Input.mat']);
delete([ReconDir,'/iQSM.mat']);

cprintf('*[0, 0, 0]', 'iQSM+ results successfully returned! \n');


    function [mag, mask, vox, z_prjs, B0, eroded_rad] = parse_iQSM_inputs(imsize, varargin)

        %% get optional inputs;
        if size(varargin,2)>0
            for k=1:size(varargin,2)
                if strcmpi(varargin{k},'mag')
                    mag = varargin{k+1};  %% magnitude data;
                end
                if strcmpi(varargin{k},'mask')
                    mask = varargin{k+1}; %% brain mask;
                end
                if strcmpi(varargin{k},'voxel_size')
                    vox = varargin{k+1}; %% resolution; voxel size in mm;
                end
                if strcmpi(varargin{k},'B0_dir')
                    z_prjs = varargin{k+1};  %% zprjs, B0_dir in MEDI;
                end
                if strcmpi(varargin{k},'B0')
                    B0 = varargin{k+1};  %% B0 field strength;
                end
                if strcmpi(varargin{k},'eroded_rad')
                    eroded_rad = varargin{k+1};  %% brain erosion radius;
                end
            end

        end
        %% setup default values;
        if ~exist('B0','var') || isempty(B0)
            cprintf('*[0, 0, 0]', 'Missing B0 input, using default ones: \n')
            cprintf('-[0, 128, 19]', 'B0 = 3T \n')
            B0 = 3;  % unit: T
        end

        if ~exist('vox','var') || isempty(vox)
            cprintf('*[0, 0, 0]', 'Missing voxel size input, using default ones: \n')
            cprintf('-[0, 128, 19] ', 'voxel_size = [1 1 1] (unit: mm) \n')
            vox = [1 1 1]; % units: mm;
        end

        if ~exist('z_prjs','var') || isempty(z_prjs)

            cprintf('*[0, 0, 0]', 'Missing B0 direction input, using default ones: \n')
            cprintf('-[0, 128, 19]', 'B0_dir = [0 0 1]  \n')
            z_prjs = [0, 0, 1];
        end

        if ~exist('eroded_rad','var') || isempty(eroded_rad)
            cprintf('*[0, 0, 0]', 'Missing eroded_radius input for brain erosion caculation, using default ones: \n')
            cprintf('-[0, 128, 19]', 'eroded_rad = 3  \n')
            eroded_rad = 3;
        end

        if ~exist('mag','var') || isempty(mag)
            cprintf('*[0, 0, 0]', 'Missing Magnitude data input, using default ones: \n')
            cprintf('-[0, 128, 19]', 'mag = 1;  \n')
            mag = ones(imsize);
        end

        if ~exist('mask','var') || isempty(mask)
            cprintf('*[0, 0, 0]', 'Missing Brain Mask input, using default ones: \n')
            cprintf('-[0, 128, 19]', 'mask = 1;  \n')
            mask = ones(imsize(1:3));
        end


    end


end












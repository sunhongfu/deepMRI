function QSM = xQSM_plus(lfs, varargin)
%------------------- xQSM reconstruction manual ------------------------------%
% this is a latest version of the matlab API for our xQSM method;
%
% Example Usage:
% *********************************************************************************
% QSM = xQSM(lfs, 'B0_dir', [0, 0, 1], ...
% 'mask', mask, ...
% 'voxel_size', [1, 1, 1], ...
% 'output_dir', pwd);
%
% where Compulsory Inputs are:
% 1. lfs: local tissue field data after background removal; 
%
% Optional Inputs are:
% 2. B0_dir: B0 field direction; the same as B0_dir in MEDI toolbox;
%    default: [0 0 1] for pure axial head orientation;
% 3. mask: Brain Mask ROI, whose size is the same as the lfs input (1-st
%    echo); default: ones;
% 4. voxel_size: image resolution; default: [1 1 1] mm isotropic;
% 5. output_dir: directory/folder for output of temporary and final results
%    default: pwd (current working directory)
% 6. save_flag (bool): flag for saving the temporary output files.
%    default: 0 (delete all output files); 
% *********************************************************************************
%
% Please cite:
% [1] Gao, Y, Zhu, X, Moffat, BA, et al. xQSM: quantitative susceptibility mapping
% with octave convolutional and noise-regularized neural networks.
% NMR in Biomedicine. 2021; 34:e4461. https://doi.org/10.1002/nbm.4461.
%
% Author(s): Yang Gao [1,2], Hongfu Sun[2,3]
% yang.gao@csu.edu.cn / yang.gao@uq.edu.au / hongfu.sun@uq.edu.au
% yang.gao@csu.edu.cn / yang.gao@uq.edu.au / hongfu.sun@uq.edu.au
% [1]: Central South University, China
% [2]: the University of Queensland, Australia
% [3]: the University of NewCastle, Australia
%
% For more deep learning based algorithms for background removal and dipole
% inversion, plese
% (1) download or clone github repo for deepMRI: https://github.com/sunhongfu/deepMR
%
% For more conventional algorithms, e.g., phase combination, phase unwrapping, please
% download or clone github repo for Hongfu's QSM toolbox: https://github.com/sunhongfu/QSM

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


% try to automatically locate where the 'xQSM' folder is downloaded and assign to 'iQSM_dir'
[xQSM_dir, ~, ~] = fileparts(which('xQSM_plus.m'));
% try to automatically locate where the 'deepMRI' repository is downloaded and assign to 'deepMRI_dir'
deepMRI_dir = fileparts(xQSM_dir);

% add MATLAB paths of deepMRI repository
% add necessary utility function for saving data and echo-fitting;
% add NIFTI saving and loading functions;
addpath(genpath(deepMRI_dir));

%% Set checkpoint versions and location
CheckPoints_folder = [xQSM_dir, '/PythonCodes/Evaluation/checkpoints'];
PyFolder = [xQSM_dir, '/PythonCodes/Evaluation/iQSM_series'];
KeyWord = 'xQSM_plus_v1';

checkpoints  = fullfile(CheckPoints_folder, KeyWord);
InferencePath = fullfile(PyFolder, KeyWord, 'Inference_xQSMSeries.py');


if ~exist('lfs','var') || isempty(lfs)

    cprintf('*red', 'lfs data input is missing! \n');
    cprintf('-[0, 128, 19]', 'The lfs input data should be a 3-D data (e.g., a 256x256x128 volume) \n')
    cprintf('*red', 'Key Parameter Missing, Ends with Error! \n');
    error(' ');
end


disp(' ')
cprintf('*[0, 0, 0]', '------------- Extracting optional parameters for reconstruction --------------------\n');

[mask, vox, B0_dir, output_dir] = parse_iQSM_inputs(size(lfs), varargin{:});

disp(' ')

fprintf('lfs is a 3D single-echo data of size %d x %d x %d\n',size(lfs,1),size(lfs,2),size(lfs,3));
fprintf('B0_dir = [%s, %s, %s]\n', num2str(B0_dir(1)), num2str(B0_dir(2)),num2str(B0_dir(3)));
fprintf('Mask is a numerical volume of size %d x %d x %d\n', size(mask, 1),size(mask, 2),size(mask, 3));
fprintf('voxel_size = [%s, %s, %s] mm\n', num2str(vox(1)), num2str(vox(2)), num2str(vox(3)));

cprintf('*[0, 0, 0]', '------------- Optional Parameters Extracted Successfully! --------------------------\n');
disp(' ')

cprintf('*[0, 0, 0]', 'Saving all data as NetworkInput.mat for Pytorch Recon! \n');

%% 1. save all the data into a NetworkInput.mat file.
lfs = single(lfs);

% interpolate the lfs to isotropic
imsize = size(lfs);
if length(imsize) == 3
    imsize(4) = 1;
end
imsize2 = [round(imsize(1:3).*vox/min(vox)), imsize(4)];

vox2 = imsize(1:3).*vox ./ imsize2(1:3);

vox2 = round(vox2 * 100) / 100; %% only keep 2 floating points precesion;

vox = vox2;

% interpolate the ph to isotropic if necessary
interp_flag = ~isequal(imsize,imsize2);

% interpolate the mag to isotropic
if interp_flag
    lfs = imresize3(lfs,imsize2(1:3));
end

% interpolate the mask to isotropic
if interp_flag
    mask = imresize3(mask,imsize2(1:3));
end

%% data permutation if necessary;

tmp_phase = ZeroPadding(lfs, 16);
[mask, pos] = ZeroPadding(mask, 16);

TE = 0; 
eroded_rad = 0; 
B0 = 3; 

[~] = Save_Input_iQSMplus(tmp_phase, mask, TE', B0, eroded_rad, B0_dir, vox, output_dir);

cprintf('*[0, 0, 0]', 'Network Input File generated successfully! \n');

cprintf('*[0, 0, 0]', 'Pytorch Reconstruction Starts! \n');

% Call Python script to conduct the reconstruction; use python API to run xQSM on the demo data
PythonRecon(InferencePath, [output_dir,filesep,'Network_Input.mat'], output_dir, checkpoints);

cprintf('*[0, 0, 0]', 'Reconstruction Ends! \n');

cprintf('*[0, 0, 0]', 'Postprocessing (e.g., echo fitting) Starts! \n');

%% load reconstruction data and save as NIFTI
load([output_dir,'/xQSM.mat']);
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
%     save_nii(nii, [output_dir, filesep, 'xQSM_all_echoes.nii']);

if interp_flag
    % comment these codes if you dont want to save temporary results;
    %         nii = make_nii(chi_fitted, vox2);
    %         save_nii(nii, [output_dir, filesep, 'xQSM_interp_echo_fitted.nii']);

    % back to original resolution if anisotropic
    chi= imresize3(chi,imsize(1:3));
    % chi_fitted = imresize3(chi_fitted,ori_imsize(1:3));
end

QSM = chi;

delete([output_dir,'/Network_Input.mat']);
delete([output_dir,'/xQSM.mat']);

nii = make_nii(QSM, vox);
save_nii(nii, [output_dir,'/xQSM_plus.nii']);

cprintf('*[0, 0, 0]', 'xQSM results successfully returned! \n');


    function [mask, vox, B0_dir, output_dir] = parse_iQSM_inputs(imsize, varargin)

        %% get optional inputs;
        if size(varargin,2)>0
            for k=1:size(varargin,2)
                if strcmpi(varargin{k},'mask')
                    mask = varargin{k+1}; %% brain mask;
                end
                if strcmpi(varargin{k},'voxel_size')
                    vox = varargin{k+1}; %% resolution; voxel size in mm;
                end
                if strcmpi(varargin{k},'output_dir')
                    output_dir = varargin{k+1};  %% brain erosion radius;
                end
                if strcmpi(varargin{k},'B0_dir')
                    B0_dir = varargin{k+1};  %% zprjs, B0_dir in MEDI;
                end
            end

        end
        %% setup default values;

        if ~exist('vox','var') || isempty(vox)
            cprintf('*[0, 0, 0]', 'Missing voxel size input, using default ones: \n')

            cprintf('-[0, 128, 19] ', 'vox = [1 1 1] \n')
            vox = [1 1 1]; % units: mm;
        end

        if ~exist('mask','var') || isempty(mask)
            cprintf('*[0, 0, 0]', 'Missing Brain Mask input, using default ones: \n')

            cprintf('-[0, 128, 19]', 'mask = 1 \n')
            mask = ones(imsize);
        end

        if ~exist('output_dir','var') || isempty(output_dir)
            cprintf('*[0, 0, 0]', 'Missing output_dir for reconstruction outputs, using default ones: \n')
            cprintf('-[0, 128, 19]', 'output_dir = pwd \n')
            output_dir = pwd; % path for reconstruction files;
        end

        if ~exist('B0_dir','var') || isempty(B0_dir)

            cprintf('*[0, 0, 0]', 'Missing B0 direction input, using default ones: \n')
            cprintf('-[0, 128, 19]', 'B0_dir = [0 0 1]  \n')

            B0_dir = [0, 0, 1];
        end

    end


end












%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Herre only showed the processing of one full-brain figure.
% For moulti-processing, plase do the loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
clc 

%% default parameters;
vox = [1 1 1]; % 1 mm isotropic
z_prjs = [0, 0, 1]; % Neutral
padding_flag = 1;

%% load Full-brain QSM
nii = load_untouch_nii('F:/BFRnet_Paper/Codes and Model/cosmos.nii');
qsm = nii.img; 
mask = qsm ~= 0;

%% Simulate background susceptibility sources and field map
R= 20; % Set the radius od geometric sources
Number = 200; % Set the number of geomotric sources
[bkg_sus,~] = PhanGene(matrix,R,Number);

bkg_sus = bkg_sus .* (~mask);
bkg_field = forward_field_calc_HS(bkg_sus, vox, z_prjs, padding_flag);
bkg_field = bkg_field .* mask;

nii = make_nii(bkg_field);
save_nii(nii, ['F:/BFRnet_Paper/Codes and Model/bkg_field_fullbrain.nii']);

%% Cropping (300 patches from one full-brain figure)

% Load full-brain data
path_qsm = ['F:/BFRnet_Paper/Codes and Model/cosmos'];
path_bkg = ['F:/BFRnet_Paper/Codes and Model/bkg_field_fullbrain'];

mkdir '../../QSM_patch/' % Optional, set your own folder
mkdir '../../BKG_patch/' % Optional

path_qsm_patch = ['F:/BFRnet_Paper/Codes and Model/QSM_patch/qsm_']; 
path_bkg_patch = ['F:/BFRnet_Paper/Codes and Model/BKG_patch/bkg_']; 

% Default Parameters
batch_size = 64;
crop_step = [16, 16, 16];
randomN = 30;
Iteration = 1; % Only test one full-brain image here.

[TotalNumber,inputStep_nii,labelStep_nii] = BFRtrainingDataCrop(path_qsm,path_bkg,...
    path_qsm_patch, path_bkg_patch, batch_size,crop_step,randomN,Iteration);

%% Implant high-susceptibility sources and simulate total fields.
Number = 300; % Calculate your own number, including the random and step-by-step cropping.

mkdir '../../QSM_HC_patch/' % Optional
mkdir '../../tfs/'          % Optional
mkdir '../../mask/'         % Optional

path_QSM_HC_patch = ['F:/BFRnet_Paper/Codes and Model/QSM_HC_patch/qsm_HC_'];
path_tfs_patch = ['F:/BFRnet_Paper/Codes and Model/tfs/tfs_'];
path_mask_patch = ['F:/BFRnet_Paper/Codes and Model/mask/mask_'];

for FileNo = 1 : Number
    % 'Number' refers to the total number of cropped patches from your loaded
    % full-brain figure. Default number is 300 patches per full-brain figure.
    
    qsm_patch_name = ([path_qsm_patch,num2str(FileNo),'.nii']);  % Load your QSM patches
    load_untouch_nii(qsm_patch_name);
    chi = ans.img; 
    clear ans;
    
    hemo_size = randi([12, 24]);
    pos_x = randi([2, 62 - hemo_size]); 
    pos_y = randi([2, 62 - hemo_size]);
    pos_z = randi([2, 62 - hemo_size]);

    hemo_value = 0.4 + 0.8 * rand; 
    
    if rand > 0.5
        hemo_value = -1 * (0.2 + 0.1 * rand);
    end

    one_data = generate_one_source([16, 16, 16]); % Set a region for leision simulation
    seg_data = abs(one_data) ~= 0; 
    SE = strel('sphere',8);
    new_seg = imclose(seg_data, SE);
    new_seg(:,:,1) = 0;
    new_seg(:,:,16) = 0;
    new_seg(:,1,:) = 0;
    new_seg(:,16,:) = 0;
    new_seg(1,:,:) = 0;
    new_seg(16,:,:) = 0;
    
    hemo_data = imresize3(single(new_seg),[hemo_size, hemo_size, hemo_size]);
    hemo_data = hemo_data > 0.5; 
    hemo_data = imclose(hemo_data, SE);
    hemo_data = hemo_data * hemo_value; 
    
    hemo_add = zeros(64, 64, 64); 
    hemo_add(pos_x:pos_x + hemo_size - 1, pos_y:pos_y + hemo_size - 1 ,...
        pos_z : pos_z + hemo_size - 1) = hemo_data;
    
    BET_mask = chi ~= 0;
    chi = chi + hemo_add;
    chi = chi .* BET_mask; 
    
    chi_patch = chi; 
    nii = make_nii(chi_patch);
    save_nii(nii, [path_QSM_HC_patch,num2str(FileNo),'.nii']); % Save the high-sus implanted QSM patches

    bkg_field_name = ([path_bkg_patch,num2str(FileNo),'.nii']); % Load the corresponding bkg field patches
    load_untouch_nii(bkg_field_name);
    bkg_field = ans.img;
    clear ans;
    
    %% LFS + BKG
    
    lfs_patch = forward_field_calc_HS(chi, vox, z_prjs, 1);
    lfs_patch = lfs_patch .* BET_mask;
    
    tfs_patch = lfs_patch + bkg_field;
    nii = make_nii(tfs_patch);
    save_nii(nii, [path_tfs_patch,num2str(FileNo),'.nii']); % Save the resulting total field map
    
    Mask_patch = MaskConvert2(BET_mask, 1);
    nii = make_nii(Mask_patch);
    save_nii(nii, [path_mask_patch,num2str(FileNo),'.nii']); % The eroded mask of patches, optional    
end
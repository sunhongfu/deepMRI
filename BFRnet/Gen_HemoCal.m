clear 
clc 

%% addpath for utilities
addpath(genpath('../../iQSM_fcns/')); 

%%
lfs_Folder = 'lfs_hemoCal_training';
wph_Folder = 'wph_hemoCal_training';
mask_Folder = 'mask_hemoCal_training';
TE_Folder = 'TE_hemoCal_training';
qsm_Folder = 'qsm_hemoCal_training'; 

mkdir(qsm_Folder)

mkdir(lfs_Folder)
mkdir(wph_Folder )
mkdir(mask_Folder)
mkdir(TE_Folder)

%% default parameters;
vox = [1 1 1]; % 1 mm isotropic
B0 = 3; % tesla
z_prjs = [0, 0, 1]; % tesla
gamma = 267.52; % gyro ratio: rad/s/T


for FileNo = 1 : 13824
    
    chi_name = sprintf('./qsm_training/chi_patch_%d.mat', FileNo);
    load(chi_name);
    chi = chi_patch;
    clear chi_patch;
    
    hemo_size = randi([12, 24]);
    pos_x = randi([2, 62 - hemo_size]); 
    pos_y = randi([2, 62 - hemo_size]);
    pos_z = randi([2, 62 - hemo_size]);

    hemo_value = 0.4 + 0.8 * rand; 
    
    if rand > 0.5
        hemo_value = -1 * (0.2 + 0.1 * rand);
    end

    one_data = generate_one_source([16, 16, 16]); %What is this?
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
    save(sprintf('./%s/QSM_patch_%d.mat', qsm_Folder, FileNo), 'chi_patch');

    bkg_name = sprintf('./bkg_training/bkg_patch_%d.mat', FileNo);
    load(bkg_name);
    bkg = bkg_patch;
    clear bkg_patch;
    
    %% different TEs.
    
    TE = 20e-3 + 5e-3 * randn(1);
    
    if TE < 5e-3
        TE = 5e-3;
    elseif TE > 35e-3
        TE = 35e-3;
    end
    
    TE_patch = repmat(TE, size(chi));
    save(sprintf('./%s/TE_patch_%d.mat', TE_Folder, FileNo), 'TE_patch');
    
    lfs_patch = forward_field_calc(chi, vox, z_prjs, 1);
    lfs_patch = lfs_patch .* BET_mask;
    save(sprintf('./%s/lfs_patch_%d.mat', lfs_Folder, FileNo), 'lfs_patch');
    
    tfs_patch = lfs_patch + bkg;
    %% save(sprintf('./%s/tfs_patch_%d.mat', tfs_Folder, FileNo), 'tfs_patch');
    unwph = -tfs_patch * gamma * B0 * TE; % assume the phase shift equals 0;
    
    cph = exp(1j * unwph); % converted into complex domain;
    
    wph = angle(cph);
    
    wph_patch = wph .* BET_mask;
    save(sprintf('./%s/wph_patch_%d.mat', wph_Folder, FileNo), 'wph_patch');
    
    Mask_patch = MaskErode(BET_mask, 1);
    save(sprintf('./%s/Mask_patch_%d.mat', mask_Folder, FileNo), 'Mask_patch');
    
    
end
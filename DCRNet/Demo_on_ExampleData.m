
clear
clc
close all

%% set parameters for saving the inputs data and subsampling mask;
cd ./MatlabCodes/

MaskDir = '../TestData/';
FileNo = 1;
addpath ./utils;

%% data loading
inp = load('../TestData/kspace_example.mat');
f = fields(inp);
k = inp.(f{1}); 

%% generate subsampling mask (AF = 4)
k = permute(k, [1, 3, 2, 4]); %conduct subsampling in the ky-kz (coronal) plane

[ny, nz, nx, ne] = size(k); % image size;

disp('Generating Subsampling Mask')
[mask] = Gen_Sampling_Mask([ny, nz], 4, 12, 1.8); %

%% Subsampling the fully-sampled kspace data and save it in appropriate for DCRNet;
disp('k-Space Undersampling')
Amp_Nor_factors = Save_Input_Data_For_DCRNet(k, mask, FileNo, MaskDir);

OpenFolder(MaskDir);

% % save(['Amp_Nor_factors_', num2str(FileNo),'.mat'],'Amp_Nor_factors');

%% Call Python script to conduct the reconstruction; 
PythonRecon('../PythonCodes/Inference.py')

%% After Python Reconstruction:
% PostProcessing: save MRI magnitude and phase images;
% addpath ./utils
SourceDir = '../TestData/'; %% make it the same as the MaskDir;
PhaseDir = '../MRI_QSM_recon/ExampleData/';  % you can modify it to be your own directory;
vox = [1 1 1]; % voxel size;

%% load reconstruction data;
recon_r_path = [SourceDir,'rec_Input_',num2str(FileNo), '_real.mat'];
recon_i_path = [SourceDir,'rec_Input_',num2str(FileNo), '_imag.mat'];

load(recon_r_path);
load(recon_i_path);

ini_recon_r_path = [SourceDir,'ini_rec_Input_',num2str(FileNo), '_real.mat'];
ini_recon_i_path = [SourceDir,'ini_rec_Input_',num2str(FileNo), '_imag.mat'];

load(ini_recon_r_path);
load(ini_recon_i_path);

%% load amplitude normlaization factors;
% load(['Amp_Nor_factors_', num2str(FileNo),'.mat']);

%% postprocessing starts;
recs = recons_r + 1j * recons_i;

recs_nodc = ini_recons_r + 1j * ini_recons_i; 

recs_new = zeros(size(recs));
recs_nodc_new = zeros(size(recs));

[ny, nz, nx, ne] = size(recs); % image size;

disp('PostProcessing Starts')
for m = 1 : ne % from echo 1 to echo ne;
    rec_tmp = recs(:,:,:,m);  % reconstruction by DCRNet;
    recs_new(:,:,:,m) = Amp_Nor_factors(m) * rec_tmp * 30; % inverse the amplitude normlization;
    
    rec_tmp = recs_nodc(:,:,:,m);  % reconstruction by DCRNet;
    recs_nodc_new(:,:,:,m) = Amp_Nor_factors(m) * rec_tmp * 30; % inverse the amplitude normlization;
end

%% save magnitude and phase images;
% nii = make_nii(abs(recs_new), vox);
% save_nii(nii, [PhaseDir, 'rec_Input_',num2str(FileNo),'_mag.nii']);
niftiwrite(abs(recs_new), [PhaseDir, 'rec_Input_',num2str(FileNo),'_DCRNet_mag.nii']);
% nii = make_nii(angle(recs_new), vox);
% save_nii(nii, [PhaseDir, 'rec_Input_',num2str(FileNo),'_ph.nii']);
niftiwrite(angle(recs_new), [PhaseDir, 'rec_Input_',num2str(FileNo),'_DCRNet_ph.nii']);

niftiwrite(abs(recs_nodc_new), [PhaseDir, 'rec_Input_',num2str(FileNo),'_DCRNet_withoutDC_mag.nii']);
niftiwrite(angle(recs_nodc_new), [PhaseDir, 'rec_Input_',num2str(FileNo),'_DCRNet_withoutDC_ph.nii']);

%% save fully-sampled gruond truth and zero-filling reconstruction for comparison; 
FS = zeros(ny, nz, nx, ne);
ZF = zeros(ny, nz, nx, ne);

for m = 1 : ne
    tmp_full = k(:,:,:,m); 
    tmp_zf = tmp_full .* mask; 
    
    FS(:,:,:,m) = fftn(fftshift(tmp_full));
    ZF(:,:,:,m) = fftn(fftshift(tmp_zf)); 
end 

niftiwrite(abs(FS), [PhaseDir, 'rec_Input_',num2str(FileNo),'_FullySampled_mag.nii']);
niftiwrite(angle(FS), [PhaseDir, 'rec_Input_',num2str(FileNo),'_FullySampled_ph.nii']);

niftiwrite(abs(ZF), [PhaseDir, 'rec_Input_',num2str(FileNo),'_ZeroFilling_mag.nii']);
niftiwrite(angle(ZF), [PhaseDir, 'rec_Input_',num2str(FileNo),'_ZeroFilling_ph.nii']);

OpenFolder(PhaseDir);

disp('PostProcessing ends, now ready for QSM reconstruction')

QSM_Recon_From_Phase; 





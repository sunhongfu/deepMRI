function kSpaceSubsampling(datapath, FileNo)

% function descriptions: 
% inputs: 
%   1) datapath: path to the k-space data; 
%   2) File No: File identifier; 
% 
% outputs: 
% 1) Amp_Nor_factors: amplitude normalization factors, which will be used
%    for postprocessing after the network reconstruciton, to recover 
%    the magnitude amplitudes of different TEs; save as 
%    Amp_Nor_factors_{FileNo}_.mat
%
% 2) two '.mat' files will be saved as network inputs:
%    Input_{FileNo}_img.mat (image-domain subsampled data) and
%    Input_{FileNo}_k.mat (undersampled kspace data); 
%    in default folder '../TestData/'
% 
%

%% set parameters for saving the inputs data and subsampling mask;
MaskDir = '../TestData/';
% % FileNo = 1;
addpath ./utils;

%% data loading
inp = load(datapath);
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

save(['Amp_Nor_factors_', num2str(FileNo),'.mat'],'Amp_Nor_factors');
end


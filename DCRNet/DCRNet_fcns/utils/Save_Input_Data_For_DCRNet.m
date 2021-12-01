function  Amp_Nor_factors = Save_Input_Data_For_DCRNet(k, mask, Dir, SaveKdata)
%
% preprocessing codes for the 3D k-space data; this programme will save the
% input data to appropriate format (.mat) for DCRNet reconstruction;
%
% inputs variables descriptions:
% 1) k: the k-space data with an image size: (Nx * Ny * Nz) * NE,
%    where NE is the number of echos, and Nx, Ny, Nz are the Matrix sizes
%    (FOV size ./ resolution);
%
% 2) mask: k-space subsampling mask; please refer to "Gen_Sampling_Mask.m"
%    about how it is generated;
%
%
% 3) Dir: directory for the output files. Default: '../TestData/'
%
% 4) save kspace data or not?
%
% output descriptions:
% 1) Amp_Nor_factors: amplitude normalization factors, which will be used
%    for postprocessing after the network reconstruciton, to recover
%    the magnitude amplitudes of different TEs;
%
% 2) two '.mat' files will be saved as network inputs for image domain
% and kspace domain data, respectively.
%
% example usage:
% Save_Input_Data_For_DCRNet(k_full, SamplingMask, '../TestData/')
%

if ~ exist('Dir','var') || isempty(Dir)
    Dir = '../TestData/';
end

if ~ exist('SaveKdata','var') || isempty(SaveKdata)
    SaveKdata = true;
end

ll = size(k);

if length(ll) == 3
    ll = [ll, 1];
end

Amp_Nor_factors = zeros(ll(4));
inputs_img = zeros(ll);% image-domain subsampled data (zero-filling reconstructions)
inputs_k = zeros(ll); % undersampled kspace data

for i = 1 : ll(4)
    k_full = k(:,:,:,i);
    
    Amp_Nor_factors(i) =  max(abs(k_full(:)));  %%
    
    k_full = k_full / max(abs(k_full(:))); %% amplitude normalization
    
    % subsample the fully-sampeld k-space data
    k_sub = k_full .* mask;
    
    % zero-filling reconstructions (image-domain inputs)
    img_tmp = fftn(fftshift(k_sub));
    
    % one dimensional fft on the kx (read out direction);
    % k-space data consistency inputs
    k_sub = fft(fftshift(k_sub, 3), [], 3);
    
    % slicing into 2D slices along kx direction;
    for j = 1 : ll(3)
        tmp = img_tmp(:,:, j);
        tmp = tmp / 30;  %% simple scaling normalization
        inputs_img(:, :, j, i) = tmp;
        
        tmp = k_sub(:,:, j);
        tmp = tmp / 30;  %% simple scaling normalization
        inputs_k(:, :, j, i) = tmp;
    end
end

%% saving data 

if SaveKdata
    save([Dir, '/NetworkInput.mat'],'inputs_img','inputs_k', 'mask', '-v7.3');
else 
    save([Dir, '/NetworkInput.mat'],'inputs_img', '-v7.3');
end

end


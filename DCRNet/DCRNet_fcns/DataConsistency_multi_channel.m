% function [rec_dc_combined, rec_dc_mc]= DataConsistency_multi_channel(recs, k_sub, mask, coil_sens, factor)
function rec_dc_mc = DataConsistency_multi_channel(recs, k_sub, mask, coil_sens, factor)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% data consitentcy for multi-channel data; 

% inputs:

% 1. recs: network reconstructions; size: ky * kz * kx,  where kx is the readout direction; 
% 2. k_sub: subsampled k-space data; size: ky * kz * kx * num_channels; 
% 3. mask: subsampling mask; size: ky * kz; 
% 4. coil_sens: coil sensitivity from POEM; size: ky * kz * kx * number_channels
% 5. factor: data consistency weights subject to [0, 1]. 0 means no data consistency;
% rec_dc(k) = (1 - dc_weights) * rec(k) * mask + dc_weights * k_sub + (1 - mask) * rec(k);

% outputs: 

% 1. rec_dc_combined: final reconstruction combined using ESPIRiT; size: ky * kz * kx; 
% 2. rec_dc_mc: uncombined final reconstruction data of mutli-channels; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imsize = size(k_sub); 

% rec_dc_combined = zeros(size(recs));  %% size: 256 * 128 * 256 * 1;

rec_dc_mc = zeros(imsize); %% size: 256 * 128 * 256 * 32;

k_sub = fftshift(fft(fftshift(k_sub, 3), [], 3), 3);  % size: 256 * 128 * 512 * 32; 1D FFT along readout direction before slicing; 

% k_sub = k_sub(:,:, imsize(3) / 4 + 1: 3 * imsize(3) / 4, :);  %% only recon the middle slices; % 256 * 128 * 256;

for ns = 1 : imsize(3)  % number of slices; 
    tmp_img = recs(:,:,ns);
    
    % S = ESPIRiT(squeeze(coil_sens(:,:,ns, :)), ones(imsize(1), imsize(2)));   %% for espirit combination; 
    
    tmp_img_MC = tmp_img .* squeeze(coil_sens(:,:,ns, :));  % 256 * 128 * 32;
    
    tmp_k_sub = k_sub(:,:,ns, :);
    
    tmp_rec_MC_dc = zeros(size(tmp_img_MC)); 

    % mask(imsize(1) / 2 - 10 : imsize(1) / 2 + 10, imsize(2) / 2 - 8 : imsize(2) / 2 + 8) = 0; % keep the low frequency of the DCRNet recon; 
    
    for i = 1 : imsize(4)  % number of receivers; 
        temp4 = tmp_img_MC(:,:,i); 
        temp = tmp_k_sub(:,:,i); 
        k_rec = ifftshift(ifft2(ifftshift(temp4)));
        
        % calib1 = abs(k_rec(imsize(1) / 2 - 10 : imsize(1) / 2 + 10, imsize(2) / 2 - 8 : imsize(2) / 2 + 8));
        % calib2 = abs(temp(imsize(1) / 2 - 10 : imsize(1) / 2 + 10, imsize(2) / 2 - 8 : imsize(2) / 2 + 8));
        
        % % p = polyfit(abs(calib1),abs(calib2),1);  % ksp calibration for data consistency; 
        
        % % k_rec = p(1) * k_rec;

        % s = sum(calib1(:).*calib2(:))/sum(calib1(:).^2);
        % k_rec = s * k_rec;


        % k_rec = k_rec / max(abs(k_rec(:)));
        % k_rec = k_rec * max(abs(temp(:)));
        k_dc = factor * mask .* temp + (1 - mask) .* k_rec + (1 - factor) * k_rec .* mask; 
        %% store the results after data consistency
        data = fftshift(fft2(fftshift(k_dc)));
        %%data = data ./ max(abs(data(:))); normalization not necessary;
        tmp_rec_MC_dc(:,:,i) = data; 
    end
    
    % tmp_combined = S' * tmp_rec_MC_dc;  %% 256 * 128 * 1; 
    
    % rec_dc_combined(:,:,ns) = tmp_combined;
    
    rec_dc_mc(:,:,ns,:) = tmp_rec_MC_dc;
    
    disp(['Recon for No.', num2str(ns), ' slice ends'])
end

end


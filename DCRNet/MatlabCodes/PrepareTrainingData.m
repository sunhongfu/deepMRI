%% Preprocessing for training datasets;
% 1. place your trianing datasets (fully-sampled kspace ground truths) 
% (in .mat format) in folder "../TrainingData" (maybe you need to create this
% directory by your self (mkdir ../TrainingData/)); 
% ps: it is recommended that you firstly reconstruct the fully-sampled
% complex images (img) and then use the following codes
% for i = 1 : size(img, 4) (we were using ME-GRE data with matrix size of 256*128*256*8)
%    tmp = ifftshift(ifftn(img(:,:,:,i))); 
%    k(:,:,:,i) = tmp; 
% end
% to generate the fully-sampled k-space data; 
imds = imageDatastore('../TrainingData/*.mat', 'FileExtensions', '.mat');

files = imds.Files;  % file list to do QSM;

mkdir('../k_full_2d_data_for_Training/');

count = 0; % file counts
for i = 1 : length(files)
    inp = load(files{i});
    f = fields(inp);
    k = inp.(f{1});
    k = permute(k, [1, 3, 2, 4]); % make sure subsampling in coronal (ky-kz) plane. 
    
    for j = 1 : size(k, 4)
        
        k_full = k(:,:,:,j); 
        
        k_full = k_full ./ max(abs(k_full(:))); % k_full: normalisaed kspace data; by max(abs(k_full))
        
        if max(abs(k_full(:)))  > 1.1
            error('Something Wrong withe the Normalizatioon!')
        end
        
        k_temp = fft(fftshift(k_full, 3), [], 3);  %% 1d FFT along readout direction.
        %%
        for n  = 26  : 235  %% get the middle slices
            k_full_2d = k_temp(:,:,n);
            file_name2 = strcat('../k_full_2d_data_for_Training/k_full_2d_', num2str(count), '.mat');
            save(file_name2, 'k_full_2d');
            count = count + 1;
        end
    end
end
disp(count)

%% genearte file list for training data load (see Dataload.py and Train.py for details); 
path = '../PythonCodes/test_IDs.txt'; 
fid = fopen(path, 'w'); 
for m = 1 : count 
    fprintf(fid, '%s \n', num2str(m)); 
end
fclose(fid); 




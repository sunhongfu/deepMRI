function data = DataRead(filename,ths)
% data = matRead(filename) reads the image data in the MAT-file filename
%% ths for adding noise; default? 0.8 means 80% percentage the data were corrupted. 
tmp = rand; 

if ths == 2
    data = niftiread(filename);
elseif tmp >= ths
    data = niftiread(filename);  % original data.
    %% add noise 
    idx = randi([1, 4]); 
    SNR = [40, 20, 10, 5]; 
    tmp_SNR = SNR(idx);
    data = AddNoise(data,tmp_SNR);
else
    data = niftiread(filename);
end
end
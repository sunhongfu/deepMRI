function [im2] = AddNoise(im1,SNR)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
sig = im1; 
sigPower = sum(abs(sig(: )).^2)/length(sig(: ));
noisePower=sigPower/SNR;
noise =  sqrt(noisePower)*randn(size(im1));

im2 = im1 + noise; 

%% new added. 
mask = im1 ~= 0; 
mask = imfill(mask, 'hole');
im2 = im2 .* mask; 

end


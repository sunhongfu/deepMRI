function [mask_ero] = MaskErode(BET_mask, ker_rad)
%MASKERODE Summary of this function goes here
%   Detailed explanation goes here

if ~ exist('ker_rad','var') || isempty(ker_rad)
    ker_rad = 3;
end

vox = [1 1 1];
% make spherical/ellipsoidal convolution kernel (ker)
rx = round(ker_rad/vox(1));
ry = round(ker_rad/vox(2));
rz = round(ker_rad/vox(3));
% % rx = max(rx,2);
% % ry = max(ry,2);
% % rz = max(rz,2);
% rz = ceil(ker_rad/vox(3));
[X,Y,Z] = ndgrid(-rx:rx,-ry:ry,-rz:rz);
h = (X.^2/rx^2 + Y.^2/ry^2 + Z.^2/rz^2 <= 1);
ker = h/sum(h(:));

% circularshift, linear conv to Fourier multiplication
csh = [rx,ry,rz]; % circularshift

% pad zeros around to avoid errors in trans between linear conv and FT multiplication
mask = padarray(BET_mask,csh);

imsize = size(mask);

mask_ero = zeros(imsize);
mask_tmp = convn(mask,ker,'same');
%mask_ero(mask_tmp > 1-1/sum(h(:))) = 1; % no error points tolerence
mask_ero(mask_tmp > 0.999999) = 1; % no error points tolerence

mask_ero = mask_ero(rx+1:end-rx,ry+1:end-ry,rz+1:end-rz);

end


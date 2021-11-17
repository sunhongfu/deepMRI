function [mask] = Gen_Sampling_Mask(MatrixSize, AF, Pa, Pb, Dir)
%% generate CS sampling mask with varied density
%  input parameters: 
% 1. MatrixSize: matrix size: default: 256 by 256; 
% 2. AF: accelerating factors (1 / undersampling rate)
% 3. Pa and Pb are parameters of the probability density function: 
% the detailed settings of Pa and Pb for different sampling rates are as
% follows:
% pa = 7, pb = 1.8 AF = 2;(0.5 sampling),
% pa = 12, pb = 1.8 , AF = 4; (0.25 sampling), 
% pa = 17, pb = 1.8 AF = 6; 
% pa = 22, pb = 1.8, AF = 8;

%% default paramters (AF = 2).
if ~ exist('MatrixSize','var') || isempty(MatrixSize)
    MatrixSize = [256, 128];
end
if ~ exist('AF','var') || isempty(AF)
    AF = 4;
end

if ~ exist('Pa','var') || isempty(Pa)
    Pa = 12; 
end

if ~ exist('Pb','var') || isempty(Pb)
    Pb = 1.8;
end

if ~ exist('Dir','var') || isempty(Dir)
    Dir = '../TestData/'; 
end

%% code implementation 
nx = MatrixSize(1); 
ny = MatrixSize(2);

x = -nx/2:nx/2-1;
y = -ny/2:ny/2-1;
[X,Y] = ndgrid(x,y);

SP = exp(-Pa*((sqrt(X.^2/nx^2 + Y.^2/ny^2)).^Pb));

SP_normalized = SP/(sum(SP(:))/(nx*ny/AF));

% generate for each pixel
mask = zeros(nx,ny);
for x = 1: nx
    for y = 1:ny
        if(rand < SP_normalized(x,y))
            mask(x,y) = 1;
        end
    end
end

%% show sampling mask;
figure; imagesc(mask); colormap gray; title('Subsampling Mask'); axis off; drawnow

%% save mask in the designated folder
save(sprintf('%sReal_Mask_Acc%d_%d_by_%d', Dir, AF, MatrixSize(1), MatrixSize(2)), 'mask'); 
end


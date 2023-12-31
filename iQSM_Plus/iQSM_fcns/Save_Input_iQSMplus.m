function mask = Save_Input_iQSMplus(phase, mask, TE, B0, erode_voxels, z_prjs, vox, Folder)

% phs_tissue: raw warpped phase images (single echo);
% mask: brain mask from FSL-BET tool; 
% TE: echo time in seconds; for example 20e-3 for 20 milliseconds; 
% B0: main field strength (T)
% erode_voxels: number of voxels for brain edge erosion;  0 for no erosion;
% Folder: saving folder; 

% example usasge: 
% SaveInput(phase, mask, TE, B0, 3, './PhaseInputs/', 1);

if ~ exist('erode_voxels','var') || isempty(erode_voxels)
    erode_voxels = 3; 
end

if ~ exist('mask','var') || isempty(mask)
    mask = ones(size(phase)); 
end


if ~ exist('z_prjs','var') || isempty(mask)
    z_prjs = [0 0 1];  
end

phase = phase .* mask; 

if erode_voxels
    mask = MaskErode(mask, erode_voxels);  % mask erosion
end

save(sprintf('%s/Network_Input.mat', Folder), 'phase', 'mask', 'TE', 'B0', 'z_prjs', 'vox'); 

end


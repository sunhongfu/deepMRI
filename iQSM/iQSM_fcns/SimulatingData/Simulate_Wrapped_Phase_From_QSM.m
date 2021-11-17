function wph = Simulate_Wrapped_Phase_From_QSM(chi, tfs_invivo, BET_mask, FileNo, TE, vox, z_prjs, B0, Dir)
% *** function descriptions ***
% this program will generate training datasets based on the QSM ground
% truths and the corresponding total field maps; 
% ** inputs: 
%  * chi: QSM ground truth; (local source in brain ROI)
%  * tfs_invivo: invivo total field maps; 
%  * BET_mask: brain mask; 
%  * FileNo: file identifier; 
%  * TE: echo time; 
%  * vox: voxel size, default 1 mm isotropic; 
%  * z_prjs: normal vector of the imaging plane, e.g. [0, 0, 1] for pure axial
%  * B0: B0 field, default 3 T; 
%  * dir: folder for saving data; 
% 
% ** outputs: 
%  * 1. wph: simulated wrapped pahse; 
%  * 2. simulated_wrapped_phase_{FileNo}.mat will be saved as the network
%       training input; 

%% code details 
%% default parameters 
gamma = 267.52; % gyro ratio: rad/s/T

if ~ exist('TE','var') || isempty(TE)
    TE = 10e-3; % second;
end
if ~ exist('vox','var') || isempty(vox)
    vox = [1 1 1]; % 1 mm isotropic
end

if ~ exist('B0','var') || isempty(B0)
    B0 = 3; % tesla 
end

if ~ exist('z_prjs','var') || isempty(z_prjs)
    z_prjs = [0, 0, 1]; % tesla 
end

if ~ exist('Dir','var') || isempty(Dir)
    Dir = './'; 
end

BET_mask = single(BET_mask);

% simulate the bkg_susceptibility source from the invivo total field maps; 
[~,bkg_sus,~] = projectionontodipolefields(tfs_invivo, BET_mask,vox,...
                        ones(size(BET_mask)),z_prjs,10);


chi = BET_mask .* chi + (1 - BET_mask) .* bkg_sus; % add the bkg_sus; 

[chi, pos] = ZeroPadding(chi, 16);
tfs_sim = forward_field_calc(chi); % ppm; 
tfs_sim = ZeroRemoving(tfs_sim, pos);

% simulate the unwrapped phase according to theoretical formula

disp(TE)

unwph = -tfs_sim * gamma * B0 * TE; % assume the phase shift equals 0; 

cph = exp(1j * unwph); % converted into complex domain;

% simulate the wrapped phase
% note this phase image is indeed similar to the shift-corrected phase instead of
% the raw invivo phase; 
wph = angle(cph); 

wph = wph .* BET_mask; 

% save(sprintf('%ssimulated_wrapped_phase_%d', Dir, FileNo), 'wph'); 
end


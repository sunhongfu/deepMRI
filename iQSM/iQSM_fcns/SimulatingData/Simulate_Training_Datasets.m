function wph = Simulate_Training_Datasets(chi, tfs_invivo, BET_mask, FileNo,  TE, vox, z_prjs, B0, Dir)
% ** inputs: 
%  * sourDir: Source Folder containing the QSM labels, invivo total field
%             maps, BET_masks. 
%  * FileNo: file identifier; 
%  * TE: echo time; 
%  * vox: voxel size, default 1 mm isotropic; 
%  * z_prjs: normal vector of the imaging plane, e.g. [0, 0, 1] for pure axial
%  * B0: B0 field, default 3 T; 
%  * dir: folder for saving data; 

% ** outputs:
%  * wph: simulated wrapped pahse; 

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

%% simulate the training inputs;  

wph = Simulate_Wrapped_Phase_From_QSM(chi, tfs_invivo, BET_mask, FileNo,...
                                TE, vox, z_prjs, B0, Dir);
                           
%%
% filepath = sprintf('%ssimulated_wrapped_phase_%d', Dir, FileNo); 
% niftiwrite(wph, filepath)            
end


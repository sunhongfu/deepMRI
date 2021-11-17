clear

clc


labels = imageDatastore('./QSM_full/chi*.nii', ...
    'FileExtensions','.nii','ReadFcn',@(x) niftiread(x))

bkgs = imageDatastore('./BKG_full/bkg*.nii', ...
    'FileExtensions','.nii','ReadFcn',@(x) niftiread(x))

qsm_folder = './qsm_training/';
bkg_folder = './bkg_training/';

mkdir(qsm_folder)
mkdir(bkg_folder)

count = 1; 

for i = 1 : length(labels.Files)
    chi_name = labels.Files{i}; 
    
    chi = niftiread(chi_name);
    
    [Nx, Ny, Nz] = size(chi);
    
    tmp_idx = randperm(Nx - 64);
    cx = [1, tmp_idx(1:4), Nx - 63]
    tmp_idx = randperm(Ny - 64);
    cy = [1, tmp_idx(1:4), Ny - 63]
    tmp_idx = randperm(Nz - 64);
    cz = [1, tmp_idx(1:2), Nz - 63]
    

    chi_name = labels.Files{i}; 
    chi = niftiread(chi_name);
    
    bkg_name = bkgs.Files{i}; 
    bkg = niftiread(bkg_name);

    for mx = 1 : 6
        for my = 1 : 6
            for mz = 1 : 4 
                idx = cx(mx);
                idy = cy(my);
                idz = cz(mz);
%                 %disp([idx, idy, idz])
                chi_patch = chi(idx: idx+63, idy:idy + 63, idz : idz+63);
                bkg_patch = bkg(idx: idx+63, idy:idy + 63, idz : idz+63);

                %disp([max(chi_patch(:)), min(chi_patch(:))])
                
                %disp([max(bkg_patch(:)), min(bkg_patch(:))])
                
                save(sprintf('%schi_patch_%d', qsm_folder, count), 'chi_patch');
                save(sprintf('%sbkg_patch_%d', bkg_folder, count), 'bkg_patch');

                count = count + 1; 
            end
        end
    end 
    
end

count 

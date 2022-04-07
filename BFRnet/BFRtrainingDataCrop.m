%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The training-set includes total field maps (tfs) and background field maps (bkg).
% Some default settings: 
% The image size of full-brain is (144, 192, 128), and the patch size is 64^3 . The crop step is 16 voxels in x/y/z axis.
% Random cropping number (randomN) is 25, and the Iterarion number equals to the number of full-brain images.
% One full-brain figure would generate 30 patches in random and 270 patches step-by-step.
% The patch size and cropping step could be set according to your own needs.
% We suggest to set a new folder for the cropped patches, e.g., ./path_tfs_patch and ./path_bkg_patch.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TotalNumber,inputStep_nii,labelStep_nii]=BFRtrainingDataCrop(path_bkg,path_tfs,path_bkg_patch, path_tfs_patch, batch_size,crop_step,randomN,Iteration)
%% Random Cropping
for itrC =1:1:Iteration
bkg_name = [path_bkg,'/bkg',num2str(itrC),'.nii']; 
bkg_nii = load_untouch_nii(bkg_name); bkg = bkg_nii.img;
tfs_name = [path_tfs,'/tfs',num2str(itrC),'.nii']; 
tfs_nii = load_untouch_nii(tfs_name); tfs = tfs_nii.img;

size_bkg = size(bkg);
size_x = size_bkg(1); size_y = size_bkg(2); size_z = size_bkg(3);
crop_stepX = crop_step(1);crop_stepY = crop_step(2);crop_stepZ = crop_step(3);

    for i0 = 1:1:randomN

        coorX = round(rand(1)*(size_x-batch_size)+1);
        coorY = round(rand(1)*(size_y-batch_size)+1);
        coorZ = round(rand(1)*(size_z-batch_size)+1);

        inputRandom = tfs(coorX:coorX+batch_size-1,coorY:coorY+batch_size-1,coorZ:coorZ+batch_size-1);
        labelRandom = bkg(coorX:coorX+batch_size-1,coorY:coorY+batch_size-1,coorZ:coorZ+batch_size-1);

        inputStep_N = [path_tfs_patch,'/Input/tfs_',num2str(randomN*(itrC-1)+i0),'.nii']; 
        inputStep_nii = make_nii(double(inputRandom)); save_nii(inputStep_nii,inputStep_N);
        labelStep_N = [path_bkg_patch,'/Label/bkg_',num2str(randomN*(itrC-1)+i0),'.nii']; 
        labelStep_nii = make_nii(double(labelRandom)); save_nii(labelStep_nii,labelStep_N);

    end
end

for iterS = 1:1:Iteration
%% Step Cropping
for k2 = 1:1:(size_z - batch_size)/crop_stepZ+1
    for j2 = 1:1:(size_y - batch_size)/crop_stepY+1
        for i2 = 1:1:(size_x - batch_size)/crop_stepX+1
            
            labelStep = bkg((i2-1)*crop_stepX+1:(i2-1)*crop_stepX+batch_size,(j2-1)*crop_stepY+1:(j2-1)*crop_stepY+batch_size,(k2-1)*crop_stepZ+1:(k2-1)*crop_stepZ+batch_size);
            inputStep = tfs((i2-1)*crop_stepX+1:(i2-1)*crop_stepX+batch_size,(j2-1)*crop_stepY+1:(j2-1)*crop_stepY+batch_size,(k2-1)*crop_stepZ+1:(k2-1)*crop_stepZ+batch_size);           
            
            Step_number = i2+(j2-1)*((size_y - batch_size)/crop_stepY+1)+(k2-1)*((size_y - batch_size)/crop_stepY+1)*((size_x - batch_size)/crop_stepX+1);
            SinIteNum = ((size_z-batch_size)/crop_stepZ+1)*((size_y-batch_size)/crop_stepY+1)*((size_x-batch_size)/crop_stepX+1);
            TotalNumber = Step_number+SinIteNum*(iterS-1)+randomN*Iteration;
            
            inputStep_N = [path_tfs_patch,'/Input/tfs_',num2str(TotalNumber),'.nii'];
            inputStep_nii = make_nii(double(inputStep)); save_nii(inputStep_nii,inputStep_N);
            labelStep_N = [path_bkg_patch,'/Label/bkg_',num2str(TotalNumber),'.nii'];
            labelStep_nii = make_nii(double(labelStep)); save_nii(labelStep_nii,labelStep_N);
        end
    end
end
end

%% 

volReader = @(x) matRead(x);
inputs = imageDatastore('./trainPatch48/*_field.mat', ...
'FileExtensions','.mat','ReadFcn',volReader);
labels = imageDatastore('./trainPatch48/?????.mat', ...
'FileExtensions','.mat','ReadFcn',volReader);

%% re-sort. 
indx = randperm(length(inputs.Files));

disp(length(inputs.Files))

for  i = 1 :  length(inputs.Files)
    temp = indx(i);
    inputs.Files{i} = strcat('./trainPatch48/', num2str(temp),'_field.mat');
    labels.Files{i} = strcat('./trainPatch48/', num2str(temp),'.mat');
end 

%% 
patchSize = [48, 48, 48];
patchPerImage = 1;
miniBatchSize = 32;
patchds = randomPatchExtractionDatastore(inputs,labels,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;
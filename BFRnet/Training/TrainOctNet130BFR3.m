

%% intialize my Unet
volReader = @(x) matRead(x);
inputs = imageDatastore('../../tfs_*.nii', ...
'FileExtensions','.nii','ReadFcn',volReader);
labels = imageDatastore('../../bkg_*.nii', ...
'FileExtensions','.nii','ReadFcn',volReader);

%% re-sort. check the data length. 
disp('Data Length: ')
disp(length(labels.Files))
disp(length(inputs.Files))

disp('Input Files');
inputs
disp('Label Files')
labels

%% 
patchSize = [64, 64, 64];
patchPerImage = 1;


miniBatchSize = 24;
patchds = randomPatchExtractionDatastore(inputs,labels,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;

%% 

[myUnet , info_net] = create3DOctNet130BN([64, 64, 64]);
disp(myUnet.Layers)
% %% training set data;

%% training optins 
initialLearningRate = 0.001;
maxEpochs = 45;
minibatchSize = miniBatchSize
l2reg = 0.00000;

options = trainingOptions('adam',...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minibatchSize,...
    'VerboseFrequency',20,...  
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','multi-gpu');
%% training function;
% parpool(2);
[net, info] = trainNetwork(patchds, myUnet, options);
%% 
disp('save trainning results')
save BFRnet_L2_64PS_24BS_45Epo_NewHCmix_2GPU.mat net; 
disp('saving complete!');




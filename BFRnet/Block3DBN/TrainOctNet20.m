%% intialize my Unet
volReader = @(x) matRead(x);
inputs = imageDatastore('../inputPatch48/*.mat', ...
'FileExtensions','.mat','ReadFcn',volReader);
labels = imageDatastore('../labelPatch48/*.mat', ...
'FileExtensions','.mat','ReadFcn',volReader);

%% re-sort. 
disp('Data Length: ')
disp(length(labels.Files))
disp(length(inputs.Files))

disp('Input Files');
inputs
disp('Label Files')
labels

%% 
patchSize = [48, 48, 48];
patchPerImage = 1;
miniBatchSize = 30;
patchds = randomPatchExtractionDatastore(inputs,labels,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;

%% 
disp('3D Octave  27 AUG, - L1 loss - 20 EPO');
[myUnet , info_net] = create3DOctNet130BN([48, 48,48]);
disp(myUnet.Layers)
% %% training set data;

%% training optins 
initialLearningRate = 0.001;
maxEpochs = 20;
minibatchSize = miniBatchSize;
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
save OCTNET_20EPO_MAE.mat net; 
disp('saving complete!');




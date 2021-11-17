
%% database for training; 
inputs = imageDatastore('../Field_VIVO/*.nii', ...
'FileExtensions','.nii','ReadFcn',@(x) DataRead(x, 0.8));

labels = imageDatastore('../QSM_VIVO/*.nii', ...
'FileExtensions','.nii','ReadFcn',@(x) DataRead(x, 2));

%% Check Data Length; 
disp('Data Length: ')
disp(length(labels.Files))
disp(length(inputs.Files))
%% combine the labels and inputs into one database. 
patchSize = [48, 48, 48];
patchPerImage = 1;
miniBatchSize = 30;
imdsTrain = randomPatchExtractionDatastore(inputs,labels,patchSize, ...
    'PatchesPerImage',patchPerImage);
imdsTrain.MiniBatchSize = miniBatchSize;

%% Create Network
disp('creating network')
[xQSM, info] = CreateXQSM([48,48,48,1]);
disp(xQSM.Layers) % check layers information. 

%% training optins 
initialLearningRate = 0.001;
maxEpochs = 100; % 
minibatchSize = miniBatchSize; % mini-batch size; 
l2reg = 0;  % weight decay factor;

options = trainingOptions('adam',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',40,...
    'LearnRateDropFactor',0.1,...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minibatchSize,...
    'VerboseFrequency',20,...  
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','multi-gpu');

%% training
[net, info] = trainNetwork(imdsTrain, xQSM, options);
%% 
disp('save trainning results')
save xQSM_noiseStudy_4level.mat net; 
disp('saving complete!');




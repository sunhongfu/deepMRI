
%% database for training; 
inputs = imageDatastore('../Field_VIVO/*.nii', ...
'FileExtensions','.nii','ReadFcn',@(x) niftiread(x));

labels = imageDatastore('../QSM_VIVO/*.nii', ...
'FileExtensions','.nii','ReadFcn',@(x) niftiread(x));

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

%% validation dataset; 
val_inputs = imageDatastore('../Field_val_VIVO/*.nii', ...
'FileExtensions','.nii','ReadFcn',@(x) niftiread(x));

val_labels = imageDatastore('../QSM_val_VIVO/*.nii', ...
'FileExtensions','.nii','ReadFcn',@(x) niftiread(x));

%% combine the labels and inputs into val database. 
imdsVal = randomPatchExtractionDatastore(val_inputs,val_labels,patchSize, ...
    'PatchesPerImage',miniBatchSize);
imdsVal.MiniBatchSize = miniBatchSize;

%% Create Network
disp('creating network')
[xQSM, info] = CreateXQSM([48,48,48,1]);
disp(xQSM.Layers) % check layers information. 

%% training optins 
initialLearningRate = 0.001;
maxEpochs = 100; % 
minibatchSize = miniBatchSize; % mini-batch size; 
l2reg = 1e-7;  % weight decay factor;

options = trainingOptions('adam',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',40,...
    'LearnRateDropFactor',0.1,...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minibatchSize,...
    'VerboseFrequency',20,...  
    'ValidationData', imdsVal,...
    'ValidationFrequency', 500,... % validate once every epoch
    'ValidationPatience',7,...    % tolerance 7 epoch for early-stopping;
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','multi-gpu');

%% training
[net, info] = trainNetwork(imdsTrain, xQSM, options);
%% 
disp('save trainning results')
save xQSM.mat net; 
disp('saving complete!');




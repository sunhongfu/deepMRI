function [lgraph, info] = OctBlock3D(info)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
lgraph = layerGraph();

Alphax = info.Alphax; 
Alphay = info.Alphay; 
convFilterSize = info.convFilterSize;
UpconvFilterSize = info.UpconvFilterSize; 
numInputChannels = info.numInputChannels; 
numOutputChannels = info.numOutputChannels; 
SectionName = info.SectionName; 

f = 0.01;
XH_channels = round((1 - Alphax) * numInputChannels);
XL_channels = numInputChannels - XH_channels;
info.XH_channels = XH_channels;
info.XL_channels = XL_channels;
YH_channels = round((1 - Alphay) * numOutputChannels); 
YL_channels = numOutputChannels - YH_channels;
info.YH_channels = YH_channels;
info.YL_channels = YL_channels;
%% traditional convolution blocks. 
%% number of parameterrs: numInputChannels * convFilterSize^2 * numOutputChannes.
% conv_name = strcat(SectionName, '-Conv-1');
% relu_name = strcat(SectionName, '-RELU-1');
% 
% conv1 = convolution3dLayer(convFilterSize,numOutputChannels,...
%     'Padding',[1 1],...
%     'BiasL2Factor',0,...
%     'Name',conv_name);
% 
% conv1.Weights = f  * randn(convFilterSize,convFilterSize,numInputChannels,numOutputChannels);
% conv1.Bias = zeros(1,1,2*numOutputChannels);
% 
% relu1 = reluLayer('Name',relu_name);
%% Octave Convolutional blocks, have the same paramter size with the traditional ones.
%% number of parameters (Alpha = 0.5): numInputChannels/2 * convFilterSize^2 * numOutputChannes/2 * 4;
conv_HH = strcat(SectionName, '-Conv-HH');

convHH = convolution3dLayer(convFilterSize,YH_channels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',conv_HH);

convHH.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XH_channels,YH_channels);
convHH.Bias = zeros(1,1,1,YH_channels);

lgraph = addLayers(lgraph,convHH);
%%
BN_HH = strcat(SectionName, '-BN_HH');
BNHH = batchNormalizationLayer('Name',BN_HH);

lgraph = addLayers(lgraph,BNHH);
%%
avgName1 = strcat(SectionName, '-AvgPooling-HL');
avg1 = averagePooling3dLayer(2, 'Stride', 2,'Name', avgName1,  'Padding',[0 0 0; 0 0 0]);

lgraph = addLayers(lgraph,avg1);

conv_HL = strcat(SectionName, '-Conv-HL');
convHL = convolution3dLayer(convFilterSize,YL_channels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...   
    'Name',conv_HL);

convHL.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XH_channels,YL_channels);
convHL.Bias = zeros(1,1,1,YL_channels);

lgraph = addLayers(lgraph,convHL);
%% 
BN_HL = strcat(SectionName, '-BN_HL');
BNHL = batchNormalizationLayer('Name',BN_HL);

lgraph = addLayers(lgraph,BNHL);
%% 
conv_LL = strcat(SectionName, '-Conv-LL');
convLL = convolution3dLayer(convFilterSize,YL_channels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...   
    'Name',conv_LL);

convLL.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XL_channels,YL_channels);
convLL.Bias = zeros(1,1,1,YL_channels);

lgraph = addLayers(lgraph,convLL);
%% 
BN_LL = strcat(SectionName, '-BN_LL');
BNLL = batchNormalizationLayer('Name',BN_LL);

lgraph = addLayers(lgraph,BNLL);
%% 
conv_LH = strcat(SectionName, '-Conv-LH');
convLH = convolution3dLayer(convFilterSize,YH_channels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...   
    'Name',conv_LH);

convLH.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XL_channels,YH_channels);
convLH.Bias = zeros(1,1,1,YH_channels);

upConv_LH = strcat(SectionName, '-UpConv-LH');
upConv = transposedConv3dLayer(UpconvFilterSize, YH_channels,...
    'Stride',2,...
    'Cropping', [0 0 0; 0 0 0],...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...      
    'Name',upConv_LH);

upConv.Weights = f  * randn(UpconvFilterSize,UpconvFilterSize, UpconvFilterSize,YH_channels,YH_channels);
upConv.Bias = zeros(1,1,1,YH_channels);
    
lgraph = addLayers(lgraph,convLH);
lgraph = addLayers(lgraph,upConv);
%% 
BN_LH = strcat(SectionName, '-BN_LH');
BNLH = batchNormalizationLayer('Name',BN_LH);

lgraph = addLayers(lgraph,BNLH);
%% addition layer
addH_name = strcat(SectionName, '-AddH');
addH = additionLayer(2,'Name',addH_name);
lgraph = addLayers(lgraph,addH);

addL_name = strcat(SectionName, '-AddL');
addL = additionLayer(2,'Name',addL_name);
lgraph = addLayers(lgraph,addL);
%% activation layers
reluH_name = strcat(SectionName, '-ReLU_H');
ReluH = reluLayer('Name',reluH_name);
lgraph = addLayers(lgraph,ReluH);

reluL_name = strcat(SectionName, '-ReLU_L');
ReluL = reluLayer('Name',reluL_name);
lgraph = addLayers(lgraph,ReluL);
%% connect the layers. 
addH1 = strcat(addH_name, '/in1');
addH2 = strcat(addH_name, '/in2');

addL1 = strcat(addL_name, '/in1');
addL2 = strcat(addL_name, '/in2');

lgraph = connectLayers(lgraph,conv_HH,BN_HH);
lgraph = connectLayers(lgraph,BN_HH,addH1);
lgraph = connectLayers(lgraph,conv_LL,BN_LL);
lgraph = connectLayers(lgraph,BN_LL,addL1);
lgraph = connectLayers(lgraph,avgName1,conv_HL);
lgraph = connectLayers(lgraph,conv_HL,BN_HL);
lgraph = connectLayers(lgraph,BN_HL,addL2);
lgraph = connectLayers(lgraph,conv_LH,upConv_LH);
lgraph = connectLayers(lgraph,upConv_LH,BN_LH);
lgraph = connectLayers(lgraph,BN_LH,addH2);

lgraph = connectLayers(lgraph,addH_name,reluH_name);
lgraph = connectLayers(lgraph,addL_name,reluL_name);
%% updata portal information;
info.portalH = reluH_name;
info.portalL = reluL_name;
end


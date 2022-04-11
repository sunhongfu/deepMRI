function [lgraph, info] = OctInut3D(info,inputTileSize)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
lgraph = layerGraph();
layers = image3dInputLayer(inputTileSize,...
    'Name','ImageInputLayer', 'Normalization', 'none');
lgraph = addLayers(lgraph, layers);

Alphay = info.Alphay; 
numInputChannels = info.numInputChannels; 
convFilterSize = info.convFilterSize;
numOutputChannels = info.numOutputChannels; 
SectionName = info.SectionName; 

f = 0.01;

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

convHH.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,numInputChannels,YH_channels);
convHH.Bias = zeros(1,1,1,YH_channels);

lgraph = addLayers(lgraph,convHH);

%%
BN_HH = strcat(SectionName, '-BN_HH');
BNHH = batchNormalizationLayer('Name',BN_HH);

lgraph = addLayers(lgraph,BNHH);
%%
avgName1 = strcat(SectionName, '-AvgPooling-HL');
avg1 = averagePooling3dLayer(2, 'Stride', 2,'Name', avgName1, 'Padding',[0 0 0; 0 0 0]);

lgraph = addLayers(lgraph,avg1);

conv_HL = strcat(SectionName, '-Conv-HL');
convHL = convolution3dLayer(convFilterSize,YL_channels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...
    'Name',conv_HL);

convHL.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,numInputChannels,YH_channels);
convHL.Bias = zeros(1,1,1,YH_channels);

lgraph = addLayers(lgraph,convHL);
%% 
BN_HL = strcat(SectionName, '-BN_HL');
BNHL = batchNormalizationLayer('Name',BN_HL);

lgraph = addLayers(lgraph,BNHL);
%% activation layers
reluH_name = strcat(SectionName, '-ReLU_H');
ReluH = reluLayer('Name',reluH_name);
lgraph = addLayers(lgraph,ReluH);

reluL_name = strcat(SectionName, '-ReLU_L');
ReluL = reluLayer('Name',reluL_name);
lgraph = addLayers(lgraph,ReluL);
%% connect the layers. 
lgraph = connectLayers(lgraph,'ImageInputLayer',conv_HH);
lgraph = connectLayers(lgraph,'ImageInputLayer',avgName1);

lgraph = connectLayers(lgraph,conv_HH,BN_HH);
lgraph = connectLayers(lgraph,avgName1,conv_HL);
lgraph = connectLayers(lgraph,conv_HL,BN_HL);

lgraph = connectLayers(lgraph,BN_HH,reluH_name);
lgraph = connectLayers(lgraph,BN_HL,reluL_name);
%% upadata connection portal; 
info.portalH = reluH_name;
info.portalL = reluL_name;
end


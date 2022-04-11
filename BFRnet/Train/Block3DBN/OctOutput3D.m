function [lgraph, info] = OctOutput3D(info)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
lgraph = layerGraph();

Alphax = info.Alphax; 
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

convHH = convolution3dLayer(convFilterSize,numOutputChannels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...   
    'Name',conv_HH);

convHH.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XH_channels,numOutputChannels);
convHH.Bias = zeros(1,1,1,numOutputChannels);

lgraph = addLayers(lgraph,convHH);
%% 
BN_HH = strcat(SectionName, '-BN_HH');
BNHH = batchNormalizationLayer('Name',BN_HH);

lgraph = addLayers(lgraph,BNHH);
%% 
conv_LH = strcat(SectionName, '-Conv-LH');
convLH = convolution3dLayer(convFilterSize,numOutputChannels,...
    'Padding', 'same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...   
    'Name',conv_LH);

convLH.Weights = f  * randn(convFilterSize,convFilterSize, convFilterSize,XL_channels,numOutputChannels);
convLH.Bias = zeros(1,1,1,numOutputChannels);

upConv_LH = strcat(SectionName, '-UpConv-LH');
upConv = transposedConv3dLayer(UpconvFilterSize,numOutputChannels,...
    'Stride',2,...
    'Cropping', [0 0 0; 0 0 0],...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',upConv_LH);

upConv.Weights = f * randn(UpconvFilterSize,UpconvFilterSize, UpconvFilterSize ,numOutputChannels,numOutputChannels);
upConv.Bias = zeros(1,1,1, numOutputChannels);
    
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
%% activation layers
reluH_name = strcat(SectionName, '-ReLU_H');
ReluH = reluLayer('Name',reluH_name);
lgraph = addLayers(lgraph,ReluH);

%%  final 1*1 conv layer; 
finalConv = convolution3dLayer(1, 1,...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...
    'Name','Final-ConvolutionLayer');

finalConv .Weights = f  * randn(1,1,1,info.numOutputChannels,1);
finalConv .Bias = zeros(1,1,1,1);

lgraph = addLayers(lgraph, finalConv);

%% connect the layers. 
addH1 = strcat(addH_name, '/in1');
addH2 = strcat(addH_name, '/in2');

lgraph = connectLayers(lgraph,conv_HH,BN_HH);
lgraph = connectLayers(lgraph,BN_HH,addH1);
lgraph = connectLayers(lgraph,conv_LH,upConv_LH);
lgraph = connectLayers(lgraph,upConv_LH,BN_LH);
lgraph = connectLayers(lgraph,BN_LH,addH2);

lgraph = connectLayers(lgraph,addH_name,reluH_name);

lgraph = connectLayers(lgraph,reluH_name,'Final-ConvolutionLayer');
end


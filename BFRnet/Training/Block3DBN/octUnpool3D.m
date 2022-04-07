function [octnet, info] = octUnpool3D(octnet, info, sections, encoding_depth)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
f = 0.01;
UpconvFilterSize = info.UpconvFilterSize;
convFilterSize = info.convFilterSize; 
%% unpooling 
upconvHH = ['Decoder-Section-' num2str(sections) '-UpConvHH'];
upConv_H = transposedConv3dLayer(UpconvFilterSize, info.YH_channels,...
    'Stride',2,...
    'Cropping', [0 0 0; 0 0 0],...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',upconvHH);

upConv_H .Weights = f * randn(UpconvFilterSize,UpconvFilterSize, UpconvFilterSize ,info.YH_channels,info.YH_channels);
upConv_H .Bias = zeros(1,1,1, info.YH_channels);

octnet = addLayers(octnet,upConv_H);


upconvLH = ['Decoder-Section-' num2str(sections) '-UpConvLH'];
upConv_LH = transposedConv3dLayer(2 * UpconvFilterSize, info.YH_channels,...
    'Stride',4,...
    'BiasL2Factor',0,...
    'Name',upconvLH);

upConv_LH.Weights = f * randn(2 * UpconvFilterSize, 2 * UpconvFilterSize,  2 * UpconvFilterSize ,info.YL_channels,info.YH_channels);
upConv_LH.Bias = zeros(1,1,1,info.YH_channels);

octnet = addLayers(octnet,upConv_LH);
%% 
addH_name = ['Decoder-Section-' num2str(sections) '-addH'];
addH = additionLayer(2,'Name',addH_name);
octnet = addLayers(octnet,addH);
%% 
BN_H = ['Decoder-Section-' num2str(sections) '-BN_H'];
BNH = batchNormalizationLayer('Name',BN_H);

octnet = addLayers(octnet,BNH);
%% 

upreluH = ['Decoder-Section-' num2str(sections) '-UpReLUH'];
upReLU_H = reluLayer('Name',upreluH);

octnet = addLayers(octnet,upReLU_H);

upconvLL = ['Decoder-Section-' num2str(sections) '-UpConvLL'];
upConv_L = transposedConv3dLayer(UpconvFilterSize, info.YL_channels,...
    'Stride',2,...
    'Cropping', [0 0 0; 0 0 0],...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',upconvLL);

upConv_L .Weights = f * randn(UpconvFilterSize,UpconvFilterSize, UpconvFilterSize ,info.YL_channels,info.YL_channels);
upConv_L .Bias = zeros(1,1,1, info.YL_channels);

octnet = addLayers(octnet,upConv_L);

conv_HL = ['Decoder-Section-' num2str(sections) '-convHL'];
convHL = convolution3dLayer(convFilterSize,info.YL_channels,...
    'Padding','same',...
    'BiasL2Factor',0,...
    'WeightsInitializer', 'glorot',...
    'BiasInitializer', 'zeros',...    
    'Name',conv_HL);
convHL.Weights = f  * randn(convFilterSize, convFilterSize,convFilterSize,info.YH_channels,info.YL_channels);
convHL.Bias = zeros(1,1,1,info.YL_channels);

octnet = addLayers(octnet,convHL);
%% 
addL_name = ['Decoder-Section-' num2str(sections) '-addL'];
addL = additionLayer(2,'Name',addL_name);
octnet = addLayers(octnet,addL);
%%
BN_L = ['Decoder-Section-' num2str(sections) '-BN_L'];
BNL = batchNormalizationLayer('Name',BN_L);

octnet = addLayers(octnet,BNL);
%% 
upreluL = ['Decoder-Section-' num2str(sections) '-UpReLUL'];
upReLU_L = reluLayer('Name',upreluL);
octnet = addLayers(octnet,upReLU_L);

%% connect layers.
octnet = connectLayers(octnet,info.portalH,upconvHH);
octnet = connectLayers(octnet,info.portalH,conv_HL);
octnet = connectLayers(octnet,info.portalL,upconvLL);
octnet = connectLayers(octnet,info.portalL,upconvLH);

addH1 = strcat(addH_name, '/in1');
addH2 = strcat(addH_name, '/in2');

addL1 = strcat(addL_name, '/in1');
addL2 = strcat(addL_name, '/in2');

octnet = connectLayers(octnet,upconvHH,addH1);
octnet = connectLayers(octnet,conv_HL ,addL1);
octnet = connectLayers(octnet,upconvLL,addL2);
octnet = connectLayers(octnet,upconvLH,addH2);

octnet = connectLayers(octnet,addH_name,BN_H);
octnet = connectLayers(octnet,BN_H,upreluH);
octnet = connectLayers(octnet,addL_name,BN_L);
octnet = connectLayers(octnet,BN_L,upreluL);

info.portalH = upreluH;
info.portalL = upreluL;
%% concatenation
concatH = ['Decoder-Section-' num2str(sections) '-DepthConcatenationH'];
depthConcatLayerH = concatenationLayer(4, 2,'Name',concatH);
octnet = addLayers(octnet,depthConcatLayerH);

concatL = ['Decoder-Section-' num2str(sections) '-DepthConcatenationL'];
depthConcatLayerL = concatenationLayer(4, 2,'Name',concatL);
octnet = addLayers(octnet,depthConcatLayerL);

%% connect layers:
temp = encoding_depth + 1 - sections; 
temp = num2str(temp);
temp_portalH = ['EncoderSection-',temp,'-Oct2-ReLU_H'];
temp_portalL = ['EncoderSection-',temp,'-Oct2-ReLU_L'];

temp_strH1 = strcat(concatH, '/in1');
temp_strH2 = strcat(concatH, '/in2');

temp_strL1 = strcat(concatL, '/in1');
temp_strL2 = strcat(concatL, '/in2');

octnet = connectLayers(octnet, upreluH,temp_strH1);
octnet = connectLayers(octnet, temp_portalH,temp_strH2);

octnet = connectLayers(octnet, upreluL,temp_strL1);
octnet = connectLayers(octnet, temp_portalL,temp_strL2);
%%  updata portal informaiton;
info.numOutputChannels = 4 * info.YH_channels;

info.portalH = concatH;
info.portalL = concatL;
end


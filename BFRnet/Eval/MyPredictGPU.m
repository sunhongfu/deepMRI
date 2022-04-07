% load total field map
function [Q_pre] = MyPredictGPU(tfs)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
imSize= size(tfs); 

load BFRnet_L2_64PS_24BS_45Epo_NewHCmix.mat
newInput = image3dInputLayer(imSize,...
    'Name','ImageInputLayer', 'Normalization', 'none');

L1Net = net; 

L1Net = replaceLayer(layerGraph(L1Net),'ImageInputLayer',newInput);
L1Net = assembleNetwork(L1Net);

clear net; 
%% prediction

predict(L1Net,  zeros(imSize), 'ExecutionEnvironment', 'auto'); % to pre-load the parameters into the memory;
tic
Q_pre = predict(L1Net,  V, 'ExecutionEnvironment', 'auto'); % Or switch 'auto' to 'gpu'/'multi-gpu' 
toc

end

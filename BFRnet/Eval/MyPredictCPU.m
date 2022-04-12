% Load total field map
function [Q_pre] = MyPredictCPU(tfs, NetPath)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
imSize= size(tfs); 

%% load Net; 

load(NetPath)% 09/01/2022

newInput = image3dInputLayer(imSize,...
    'Name','ImageInputLayer', 'Normalization', 'none');

L1Net = net; 

L1Net = replaceLayer(layerGraph(L1Net),'ImageInputLayer',newInput);
L1Net = assembleNetwork(L1Net);

clear net; 
%% prediction

predict(L1Net,  zeros(imSize), 'ExecutionEnvironment', 'cpu'); % to pre-load the parameters into the memory;
tic
Q_pre = predict(L1Net, tfs, 'ExecutionEnvironment', 'cpu'); 
toc

end

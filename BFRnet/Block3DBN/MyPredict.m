function [Q_pre] = MyPredict(V)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
imSize= size(V); 

%% load Net; 
load BFR_L2_64_24BS_45Epo_BKG_geoCorrected.mat
% load xQSM_noiseStudy_4level_HybridDataset.mat
% load BFR_L2_64_32BS_45Epo_BKG.mat % 06072021 Test2
% load BFR_L2_64_32BS_45Epo_BKG.mat; % Background-trained OctNet
% load Oct_L2_64_24BS_45Epo.mat % Network for Dipole Inversion
% load BFR_Unet_L2_64_24BS_45Epo.mat
% load 3dresnet_L1loss_vivo.mat
% load BFR_L2_64_24BS_45Epo_BKGoriginal.mat  % 06072021 Test1

newInput = image3dInputLayer(imSize,...
    'Name','ImageInputLayer', 'Normalization', 'none');

L1Net = net; 

L1Net = replaceLayer(layerGraph(L1Net),'ImageInputLayer',newInput);
L1Net = assembleNetwork(L1Net);

clear net; 
%% prediction

Q_pre = predict(L1Net, V, 'ExecutionEnvironment', 'cpu'); 

end
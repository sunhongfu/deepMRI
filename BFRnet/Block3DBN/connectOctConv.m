function [lgraph, info] = connectOctConv(OctConv1,OctConv2,info1, info2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
lgraph = OctConv1;
info = info1; 
tempLayers = OctConv2.Layers;
for i = 1 : 1 : length(tempLayers)
templ = tempLayers(i);
lgraph = addLayers(lgraph, templ);
end
%% section1 RELU ---- convs of Section2. 
% SectionName1 = info1.SectionName; 
% 
% Section1_portalH = strcat(SectionName1, '-ReLU_H');
% Section1_portalL = strcat(SectionName1, '-ReLU_L');
Section1_portalH = info1.portalH; 
Section1_portalL = info1.portalL;

SectionName2 = info2.SectionName; 
Section2_portalHH = strcat(SectionName2, '-Conv-HH');
Section2_portalLL = strcat(SectionName2, '-Conv-LL');
Section2_portalHL = strcat(SectionName2, '-AvgPooling-HL');
Section2_portalLH = strcat(SectionName2, '-Conv-LH');

lgraph = connectLayers(lgraph,Section1_portalH,Section2_portalHH);
lgraph = connectLayers(lgraph,Section1_portalH,Section2_portalHL);
lgraph = connectLayers(lgraph,Section1_portalL,Section2_portalLL);
lgraph = connectLayers(lgraph,Section1_portalL,Section2_portalLH);

%% 
tempConns = OctConv2.Connections;
tempConns = table2array(tempConns);
for i = 1 : 1 : length(tempConns)
    lgraph = connectLayers(lgraph,tempConns{i,1},tempConns{i,2});
end

%% updata portal information;
info = info2;
info.portalH = info2.portalH;
info.portalL = info2.portalL;
end


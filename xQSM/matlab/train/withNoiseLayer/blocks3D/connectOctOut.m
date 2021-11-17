function [lgraph, info] = connectOctOut(OctConv1,OctConv2,info1, info2)
%% connect the final layer into xQSM network; 
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
Section2_portalLH = strcat(SectionName2, '-Conv-LH');

lgraph = connectLayers(lgraph,Section1_portalH,Section2_portalHH);
lgraph = connectLayers(lgraph,Section1_portalL,Section2_portalLH);

%% 
tempConns = OctConv2.Connections;
tempConns = table2array(tempConns);
for i = 1 : 1 : length(tempConns)
    lgraph = connectLayers(lgraph,tempConns{i,1},tempConns{i,2});
end

%% create Residual block:
add_final = additionLayer(2,'Name','add_final');
lgraph = addLayers(lgraph, add_final);
lgraph = connectLayers(lgraph, 'ImageInputLayer','add_final/in1');
lgraph = connectLayers(lgraph, 'Final-ConvolutionLayer','add_final/in2');
%% regression layer

regLayer = regressionLayer('Name', 'Output Layer');
lgraph = addLayers(lgraph,regLayer);
lgraph = connectLayers(lgraph,'add_final','Output Layer');
end


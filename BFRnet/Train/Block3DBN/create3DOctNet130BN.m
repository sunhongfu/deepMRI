function [octnet, info_net] = create3DOctNet130BN(inputTileSize)
%UNTITLED9 Summary of this function goes here

disp('correct')
%   Detailed explanation goes here
addpath ./blocks3DBN_mae
%% test codes for how to create Octave-Res-Unet, can be extended to pytorch, tf. 
info.Alphax = 0.5; 
info.Alphay = 0.5; 
info.convFilterSize = 3;
info.UpconvFilterSize = 2; 
info.numInputChannels = 64; 
info.numOutputChannels = 64; 
info.SectionName = 'TestSection'; 
info.portalH = '';
info.portalL = '';
%%
% [1, 64, 64, 64, 128, 128, 128, 256, 256, 256, 128, 128, 128];
%% input section; 
info_input = info; 
info_input.Alphax = 0; 
info_input.numInputChannels = 1; % Change the number of Input Channel ZXY
info_input.SectionName = 'Input';
[oct_input, info_input] = OctInut3D(info_input, inputTileSize);

infos{1} = info_input; 
octaves{1} = oct_input;
octnet = oct_input;
info_net = info_input;
ind = 1;
%% encoding path: encoding depth: 2
encoding_depth = 4;  %%%% Xuanyu 18/12/2019
for i = 1 : encoding_depth
    %% octave conv block 1; 
    SectionName = ['EncoderSection', '-', num2str(i), '-Oct1']; 
    info.SectionName = SectionName; 
    info_temp = info; 
    info_temp.numInputChannels = info_net.numOutputChannels;
    info_temp.numOutputChannels = 2 * info_net.numOutputChannels;  
    [oct_temp, info_temp] = OctBlock3D(info_temp);
    %% updata layers information;
    ind = ind + 1; 
    infos{ind} = info_temp;
    octaves{ind} = oct_temp;
    [octnet, info_net] = connectOctConv(octnet,oct_temp,info_net, info_temp);
    %% octave conv block 2
    SectionName = ['EncoderSection', '-', num2str(i), '-Oct2']; 
    info.SectionName = SectionName; 
    info_temp = info; 
    info_temp.numInputChannels = info_net.numOutputChannels;
    info_temp.numOutputChannels = info_net.numOutputChannels;  
    [oct_temp, info_temp] = OctBlock3D(info_temp);
    %% updata layers information;
    ind = ind + 1; 
    infos{ind} = info_temp;
    octaves{ind} = oct_temp;
    [octnet, info_net] = connectOctConv(octnet,oct_temp,info_net, info_temp);    
    %% max pooling layers. 
    maxpool1 = ['EncoderSection', '-', num2str(i), '-MaxPooling1'];
    maxPoolLayer1 = maxPooling3dLayer(2, 'Stride', 2, 'Name',maxpool1);
    octnet = addLayers(octnet,maxPoolLayer1);
    
    maxpool2 = ['EncoderSection', '-', num2str(i), '-MaxPooling2'];
    maxPoolLayer2 = maxPooling3dLayer(2, 'Stride', 2, 'Name',maxpool2);
    octnet = addLayers(octnet,maxPoolLayer2);
    %% connect pooling layers.   
    octnet = connectLayers(octnet,info_net.portalH,maxpool1);
    octnet = connectLayers(octnet,info_net.portalL,maxpool2);
    %% update portals with maxpooling output; 
    info_net.portalH = maxpool1; 
    info_net.portalL = maxpool2;
end
%% mid layers
factors = [2, 1/2];
for i = 1 : 2
    %% construct octave convs.
    SectionName = ['MidSection', '-', num2str(i), '-Oct1'];
    info.SectionName = SectionName;
    info_temp = info;
    info_temp.numInputChannels = info_net.numOutputChannels;
    info_temp.numOutputChannels = factors(i) * info_net.numOutputChannels;
    [oct_temp, info_temp] = OctBlock3D(info_temp);
    %% updata layers information;
    ind = ind + 1; 
    infos{ind} = info_temp;
    octaves{ind} = oct_temp;
    [octnet, info_net] = connectOctConv(octnet,oct_temp,info_net, info_temp);    
end

%% expanding paths: encoding depth: 2
for i = 1 : encoding_depth
    %% first unpooling and contactation
    [octnet, info_net] = octUnpool3D(octnet, info_net, i, encoding_depth);
    %% octave conv block 1;
    SectionName = ['DecoderSection', '-', num2str(i), '-Oct1']; 
    info.SectionName = SectionName; 
    info_temp = info; 
    info_temp.numInputChannels = info_net.numOutputChannels;
    info_temp.numOutputChannels = info_net.numOutputChannels / 2;  
    [oct_temp, info_temp] = OctBlock3D(info_temp);
    %% updata layers information;
    ind = ind + 1; 
    infos{ind} = info_temp;
    octaves{ind} = oct_temp;
    [octnet, info_net] = connectOctConv(octnet,oct_temp,info_net, info_temp);
    %% octave conv block 2;
    if i == encoding_depth
        SectionName = ['DecoderSection', '-', num2str(i), '-Oct2'];
        info.SectionName = SectionName;
        info_temp = info;
        info_temp.numInputChannels = info_net.numOutputChannels;
        info_temp.numOutputChannels = info_net.numOutputChannels;
        [oct_temp, info_temp] = OctBlock3D(info_temp);
        %% updata layers information;
        ind = ind + 1;
        infos{ind} = info_temp;
        octaves{ind} = oct_temp;
        [octnet, info_net] = connectOctConv(octnet,oct_temp,info_net, info_temp);
    else
        SectionName = ['DecoderSection', '-', num2str(i), '-Oct2'];
        info.SectionName = SectionName;
        info_temp = info;
        info_temp.numInputChannels = info_net.numOutputChannels;
        info_temp.numOutputChannels = info_net.numOutputChannels / 2;
        [oct_temp, info_temp] = OctBlock3D(info_temp);
        %% updata layers information;
        ind = ind + 1;
        infos{ind} = info_temp;
        octaves{ind} = oct_temp;
        [octnet, info_net] = connectOctConv(octnet,oct_temp,info_net, info_temp);
    end
end

%% output section: 
info_out = info; 
info_out.Alphay=0;
info_out.numInputChannels = info_net.numOutputChannels; 
info_out.numOutputChannels = 64; 
info_out.SectionName = 'Output';
[oct_out, info_out] = OctOutput3D(info_out);

infos{end + 1} = info_out; 
octaves{end + 1} = oct_out;

[octnet, info_net] = connectOctOut(octnet,oct_out,info_net, info_out);
rmpath ./blocks3DBN_mae
end


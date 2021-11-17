% createXQSM creates a deep learning network with the  xQSM architecture
% for the details, refer to ...;
% for example, [xQSM, info] = CreateXQSM([48,48,48,1]); returns an xQSM network which accepts
% images of size inputTileSize of 48^3 with only 1 input channel.

function [octnet, info_net] = CreateXQSM(inputTileSize)
%% addpath path for xQSM blocks. 
addpath ./blocks3D
%% default parameters for the OctConv layers to facotrize the input images into 
%% High and Low resolution groups. 
info.Alphax = 0.5;  % factorization factor of input;
info.Alphay = 0.5;  % factorization factor of output;
info.convFilterSize = 3; % kernal size of convolution layers; 
info.UpconvFilterSize = 2; % kernal size of transposed convolutional layers;
info.SectionName = 'MidSection'; % temporary name, could be arbitrary;
info.portalH = '';  % connections for adjecent OctConv layers of high resolutions; 
info.portalL = '';  % connections for adjecent OctConv layers of low resolutions;
info.numInputChannels = 64; % the default number of input channels
info.numOutputChannels = 64; % the default number of output channels
%% default encoding path: 2
encoding_depth = 2;  % adjusted for the training data (patch size 48^3);
for i = 1 : encoding_depth
    if i == 1
        %% first layer;
        info_input = info; % get the default parameters;
        info_input.Alphax = 0;  % for the first layer alpha_x = 0;
        info_input.numInputChannels = 1; % single-channel input;
        info_input.SectionName = 'Input';
        [oct_input, info_input] = OctInut3D(info_input, inputTileSize); % create first layer
        octnet = oct_input; % add the first layer;
        info_net = info_input; % get the connection information;
    else
        %% octave conv block 1;
        SectionName = ['EncoderSection', '-', num2str(i), '-Oct1'];
        info.SectionName = SectionName;
        info_temp = info; % get the default parameters;
        info_temp.numInputChannels = info_net.numOutputChannels; % calculate the number of the kernels;
        info_temp.numOutputChannels = 2 * info_net.numOutputChannels;  % calculate the number of output kernels
        [oct_temp, info_temp] = OctBlock3D(info_temp); % create mid-layers;
        %% updata layers information;
        [octnet, info_net] = connectOctConv(octnet,oct_temp,info_net, info_temp);
    end
    %% octave conv block 2
    SectionName = ['EncoderSection', '-', num2str(i), '-Oct2']; 
    info.SectionName = SectionName; 
    info_temp = info; 
    info_temp.numInputChannels = info_net.numOutputChannels;
    info_temp.numOutputChannels = info_net.numOutputChannels;  
    [oct_temp, info_temp] = OctBlock3D(info_temp);
    %% updata layers information;
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
    [octnet, info_net] = connectOctConv(octnet,oct_temp,info_net, info_temp);
    %% octave conv block 2;
    if i == encoding_depth
        %% output section:
        info_out = info;
        info_out.Alphay=0;
        info_out.numInputChannels = info_net.numOutputChannels;
        info_out.SectionName = 'Output';
        [oct_out, info_out] = OctOutput3D(info_out);
        [octnet, info_net] = connectOctOut(octnet,oct_out,info_net, info_out);
    else
        SectionName = ['DecoderSection', '-', num2str(i), '-Oct2'];
        info.SectionName = SectionName;
        info_temp = info;
        info_temp.numInputChannels = info_net.numOutputChannels;
        info_temp.numOutputChannels = info_net.numOutputChannels / 2;
        [oct_temp, info_temp] = OctBlock3D(info_temp);
        %% updata layers information;
        [octnet, info_net] = connectOctConv(octnet,oct_temp,info_net, info_temp);
    end
end
rmpath ./blocks3D
end


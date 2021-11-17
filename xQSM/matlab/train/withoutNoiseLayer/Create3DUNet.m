% create3DUnet creates a deep learning network with the 3D U-net
% architecture which is the backbone netowrk for xQSM. 
% for the details, refer to ...;
% for example, oriUnet = Create3DUnet([48,48,48,1]); returns an 3D U-net which accepts
% images of size inputTileSize of 48^3 with only 1 input channel.
% modified from the original functions provided by MATLAB inc. 

function lgraph = Create3DUNet(inputTileSize)

%% default parameters which are the same as those of xQSM network; 
encoderDepth = 2;  
initialEncoderNumChannels = 64;
inputNumchannels = inputTileSize(4);
convFilterSize = 3;
UpconvFilterSize = 2;
f = 0.01;
%%
layers = image3dInputLayer(inputTileSize,...
    'Name','ImageInputLayer', 'Normalization', 'none');
layerIndex = 1;

% Create encoder layers
for sections = 1:encoderDepth
    
    encoderNumChannels = initialEncoderNumChannels * 2^(sections-1);
        
    if sections == 1
        conv1 = convolution3dLayer(convFilterSize,encoderNumChannels,...
            'Padding','same',...
            'NumChannels',inputNumchannels,...
            'BiasL2Factor',0,...
            'Name',['Encoder-Section-' num2str(sections) '-Conv-1']);
        
        conv1.Weights = f * randn(convFilterSize,convFilterSize,convFilterSize,inputNumchannels,encoderNumChannels);
    else
        conv1 = convolution3dLayer(convFilterSize,encoderNumChannels,...
            'Padding','same',...
            'BiasL2Factor',0,...
            'Name',['Encoder-Section-' num2str(sections) '-Conv-1']);
        
        conv1.Weights = f * randn(convFilterSize,convFilterSize,convFilterSize,encoderNumChannels/2,encoderNumChannels);
    end
    
    conv1.Bias = zeros(1,1,1,encoderNumChannels);
    
    BN_1 = ['Encoder-Section-' num2str(sections) '-BN-1'];
    BN1 = batchNormalizationLayer('Name',BN_1);
    
    relu1 = reluLayer('Name',['Encoder-Section-' num2str(sections) '-ReLU-1']);
    
    conv2 = convolution3dLayer(convFilterSize,encoderNumChannels,...
        'Padding','same',...
        'BiasL2Factor',0,...
        'Name',['Encoder-Section-' num2str(sections) '-Conv-2']);
    
    conv2.Weights = f * randn(convFilterSize, convFilterSize,convFilterSize,encoderNumChannels,encoderNumChannels);
    conv2.Bias = zeros(1,1,1,encoderNumChannels);
    
    BN_2 = ['Encoder-Section-' num2str(sections) '-BN-2'];
    BN2 = batchNormalizationLayer('Name',BN_2);
    
    relu2 = reluLayer('Name',['Encoder-Section-' num2str(sections) '-ReLU-2']);
       
    layers = [layers; conv1; BN1; relu1; conv2; BN2 ;relu2];     %#ok<*AGROW>
    layerIndex = layerIndex + 6;
    
    maxPoolLayer = maxPooling3dLayer(2, 'Stride', 2, 'Name',['Encoder-Section-' num2str(sections) '-MaxPool']);
    
    layers = [layers; maxPoolLayer];
    layerIndex = layerIndex +1;
    
end
%Create mid layers
conv1 = convolution3dLayer(convFilterSize,2*encoderNumChannels,...
    'Padding','same',...
    'BiasL2Factor',0,...
    'Name','Mid-Conv-1');

conv1.Weights = f  * randn(convFilterSize, convFilterSize,convFilterSize,encoderNumChannels,2*encoderNumChannels);
conv1.Bias = zeros(1,1,1, 2*encoderNumChannels);

BN_1 = 'mid-Section-BN-1';
BN1 = batchNormalizationLayer('Name',BN_1);

relu1 = reluLayer('Name','Mid-ReLU-1');

conv2 = convolution3dLayer(convFilterSize,encoderNumChannels,...
    'Padding','same',...
    'BiasL2Factor',0,...
    'Name','Mid-Conv-2');

conv2.Weights = f * randn(convFilterSize, convFilterSize,convFilterSize,2*encoderNumChannels,encoderNumChannels);
conv2.Bias = zeros(1,1,1,encoderNumChannels);

BN_2 = 'mid-Section-BN-2';
BN2 = batchNormalizationLayer('Name',BN_2);

relu2 = reluLayer('Name','Mid-ReLU-2');
layers = [layers; conv1; BN1;relu1; conv2; BN2;relu2];
layerIndex = layerIndex + 6;

initialDecoderNumChannels = encoderNumChannels;

%Create decoder layers
for sections = 1:encoderDepth
    
    decoderNumChannels = initialDecoderNumChannels / 2^(sections-1);
    
    upConv = transposedConv3dLayer(UpconvFilterSize, decoderNumChannels,...
        'Stride',2,...
        'BiasL2Factor',0,...
        'Name',['Decoder-Section-' num2str(sections) '-UpConv']);

    upConv.Weights = f * randn(UpconvFilterSize, UpconvFilterSize,UpconvFilterSize,decoderNumChannels,decoderNumChannels);
    upConv.Bias = zeros(1,1, 1,decoderNumChannels);    
    
        BN_1 = ['Decoder-Section-' num2str(sections) '-BN_1'];
    BN1 = batchNormalizationLayer('Name',BN_1);
    
    upReLU = reluLayer('Name',['Decoder-Section-' num2str(sections) '-UpReLU']);
    
    depthConcatLayer = concatenationLayer(4, 2,'Name',...
        ['Decoder-Section-' num2str(sections) '-DepthConcatenation']);
    
    conv1 = convolution3dLayer(convFilterSize,decoderNumChannels,...
        'Padding','same',...
        'BiasL2Factor',0,...
        'Name',['Decoder-Section-' num2str(sections) '-Conv-1']);
    
    conv1.Weights = f * randn(convFilterSize,convFilterSize,convFilterSize,2*decoderNumChannels,decoderNumChannels);
    conv1.Bias = zeros(1,1,1, decoderNumChannels);
    
    
    BN_2 = ['Decoder-Section-' num2str(sections) '-BN_2'];
    BN2 = batchNormalizationLayer('Name',BN_2);
    
    relu1 = reluLayer('Name',['Decoder-Section-' num2str(sections) '-ReLU-1']);
   if sections == encoderDepth
       conv2 = convolution3dLayer(convFilterSize,decoderNumChannels,...
        'Padding','same',...
        'BiasL2Factor',0,...
        'Name',['Decoder-Section-' num2str(sections) '-Conv-2']);
    
    conv2.Weights = f * randn(convFilterSize, convFilterSize,convFilterSize,decoderNumChannels,decoderNumChannels);
    conv2.Bias = zeros(1,1,1, decoderNumChannels);
    
        BN_3 = ['Decoder-Section-' num2str(sections) '-BN_3'];
    BN3 = batchNormalizationLayer('Name',BN_3);
    
    relu2 = reluLayer('Name',['Decoder-Section-' num2str(sections) '-ReLU-2']);
   else
    conv2 = convolution3dLayer(convFilterSize,decoderNumChannels/2,...
        'Padding','same',...
        'BiasL2Factor',0,...
        'Name',['Decoder-Section-' num2str(sections) '-Conv-2']);
    
    conv2.Weights = f * randn(convFilterSize, convFilterSize,convFilterSize,decoderNumChannels,decoderNumChannels/2);
    conv2.Bias = zeros(1,1,1, decoderNumChannels/2);
    
        BN_3 = ['Decoder-Section-' num2str(sections) '-BN_3'];
    BN3 = batchNormalizationLayer('Name',BN_3);
    
    relu2 = reluLayer('Name',['Decoder-Section-' num2str(sections) '-ReLU-2']);
   end
    layers = [layers; upConv; BN1 ;upReLU; depthConcatLayer; conv1; BN2 ;relu1; conv2; BN3;relu2];
    
    layerIndex = layerIndex + 10;
end

finalConv = convolution3dLayer(1, 1,...
    'BiasL2Factor',0,...
    'Name','Final-ConvolutionLayer');

finalConv.Weights = f * randn(1,1,1,decoderNumChannels,1);
finalConv.Bias = zeros(1,1,1,1);

%% use MSE loss function. 
regLayer = regressionLayer('Name', 'Output Layer');
layers = [layers; finalConv; regLayer];

% Create the layer graph and create connections in the graph
lgraph = layerGraph(layers);

% Connect concatenation layers
lgraph = connectLayers(lgraph, 'Encoder-Section-1-ReLU-2','Decoder-Section-2-DepthConcatenation/in2');
lgraph = connectLayers(lgraph, 'Encoder-Section-2-ReLU-2','Decoder-Section-1-DepthConcatenation/in2');

end


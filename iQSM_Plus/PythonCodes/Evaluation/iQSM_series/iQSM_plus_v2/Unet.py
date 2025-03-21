###### Yang's U-net for comparision study ################
## This file is for network construction. 

#################### Section 1 ###########################
## Import necessary library and packages and further pre-
## -defined packages here.
## import self-defined packages
import torch.nn.functional as F
from Unet_blocks import * 
################# End Section 1 ##########################

#################### Section 2 ###########################
## Parameters： Encoding depth: Times of Poolings 
class Unet(nn.Module):
    def __init__(self, EncodingDepth, In_channels, Out_channels):
        super(Unet, self).__init__()
        self.EncodeConvs = []
        self.DecodeConvs = []

        self.EncodingDepth = EncodingDepth
        initial_num_layers = 16

        self.Z_embed = nn.Sequential(nn.Linear(3, 3), nn.GELU(), nn.Linear(3, 5))

        self.ini_embd_layer = Embeding_Layer(initial_num_layers)

        temp = list(range(1, EncodingDepth + 1))
################### Encoding Layers #######################
        for encodingLayer in temp:
            if encodingLayer == 1:
                num_outputs= initial_num_layers * 2 ** (encodingLayer - 1)
                self.EncodeConvs.append(EncodingBlocks(In_channels, num_outputs))

            else:             
                num_outputs = initial_num_layers * 2 ** (encodingLayer - 1)
                self.EncodeConvs.append(EncodingBlocks(num_outputs // 2, num_outputs))

        self.EncodeConvs = nn.ModuleList(self.EncodeConvs)
################### Mid Layers ############################
        self.MidConv1 = MidBlocks(num_outputs)
        initial_decode_num_ch = num_outputs
################### Decoding Layers #######################
        for decodingLayer in temp:
            if decodingLayer == EncodingDepth:
                num_inputs = initial_decode_num_ch // 2 ** (decodingLayer - 1)
                self.DecodeConvs.append(DecodingBlocks(num_inputs, num_inputs))

            else:
                num_inputs = initial_decode_num_ch // 2 ** (decodingLayer - 1)
                self.DecodeConvs.append(DecodingBlocks(num_inputs, num_inputs // 2))
                
        self.DecodeConvs = nn.ModuleList(self.DecodeConvs)

        self.FinalConv = nn.Conv3d(num_inputs, Out_channels, 1, stride = 1, padding = 0)
################## End Section 2 ##########################

    def forward(self, x_b, x_d, z_prjs):
        ## input： x, wrapped phase images; 
        Input = x_b

        x = torch.cat([x_b, x_d], dim = 1)

        z_prjs = self.Z_embed(z_prjs).squeeze(1)        ## z embedding

        x = self.ini_embd_layer(x, z_prjs)
        
        names = self.__dict__
        temp = list(range(1, self.EncodingDepth + 1))
        for encodingLayer in temp:
            temp_conv = self.EncodeConvs[encodingLayer - 1]
            x = temp_conv(x, z_prjs)
            #print('EncodeConv' + str(encodingLayer) + str(x.size()))
            names['EncodeX' + str(encodingLayer)] = x
            x = F.max_pool3d(x, 2)
           #print('Pooling' + str(encodingLayer) + str(x.size()))

        x = self.MidConv1(x, z_prjs)
        #print('Mid' + str(encodingLayer) + str(x.size()))

        for decodingLayer in temp:
            temp_conv = self.DecodeConvs[decodingLayer - 1]
            x2 = names['EncodeX' + str(self.EncodingDepth - decodingLayer + 1)]
            x = temp_conv(x, x2, z_prjs)
            #print('DecodeConv' + str(decodingLayer) + str(x.size()))

        x = self.FinalConv(x)
        x = x + Input
        return x

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(m.bias)   
    if isinstance(m, nn.ConvTranspose3d):
        nn.init.normal_(m.weight, mean=0, std=1e-2)
        nn.init.zeros_(m.bias)   
    if isinstance(m, nn.BatchNorm3d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)   

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

#################### For Code Test ##################################
## before running the training codes, verify the network architecture. 
if __name__ == '__main__':
    unet1 = Unet(4, 16, 1)
    unet2 = Unet(4, 16, 1)
    LPLayer = LapLayer(torch.randn(1, 1, 3, 3, 3))
    unet1.apply(weights_init)
    print(get_parameter_number(unet1))
    x = torch.randn(1,1,48,48,48, dtype=torch.float)
    M = torch.randn(1,1,48,48,48, dtype=torch.float)
    print('input' + str(x.size()))
    print(x.dtype)
    b_i, d_i = LPLayer(x, M)
    y1 = unet1(b_i, d_i)
    y2 = unet2(b_i, d_i)
    print('output'+str(y1.size()))
    print('output'+str(y2.size()))
    print('output'+str(y2.size()))

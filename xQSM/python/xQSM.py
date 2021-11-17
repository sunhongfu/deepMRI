##############  octave convolution #######################
## build a octave-u-net with the same backbone of 
## traditional U-net. this programme is in the same
## structure withe Unet.py
#################### Section 1 ###########################
from xQSM_blocks import * 
################# End Section 1 ##########################

#################### Section 2 ###########################
## Parameters： Encoding depth: Times of Poolings 
class xQSM(nn.Module):
    def __init__(self, EncodingDepth):
        super(xQSM, self).__init__()
        self.EncodeConvs = []
        self.DecodeConvs = []
        #self.downs = []
        self.EncodingDepth = EncodingDepth
        initial_num_layers = 64
        temp = list(range(1, EncodingDepth + 1))
        self.InputOct = OctEncodingBlocks(1, initial_num_layers, alphax = 1, alphay = 0.5)
################### Encoding Layers #######################
        for encodingLayer in temp:
            if encodingLayer == 1:
                num_outputs= initial_num_layers * 2 ** (encodingLayer - 1)
                self.EncodeConvs.append(OctEncodingBlocks(initial_num_layers, num_outputs))
            else:             
                num_outputs = initial_num_layers * 2 ** (encodingLayer - 1)
                self.EncodeConvs.append(OctEncodingBlocks(num_outputs // 2, num_outputs))
        self.EncodeConvs = nn.ModuleList(self.EncodeConvs)       
        #self.downs = nn.ModuleList(self.downs)
################### Mid Layers ############################
        self.MidConv1 = OctMidBlocks(num_outputs)
        initial_decode_num_ch = num_outputs
################### Decoding Layers #######################
        for decodingLayer in temp:
            if decodingLayer == EncodingDepth:
                num_inputs = initial_decode_num_ch // 2 ** (decodingLayer - 1)
                self.DecodeConvs.append(OctDecodingBlocks(num_inputs, num_inputs))
            else:
                num_inputs = initial_decode_num_ch // 2 ** (decodingLayer - 1)
                self.DecodeConvs.append(OctDecodingBlocks(num_inputs, num_inputs // 2))
        self.DecodeConvs = nn.ModuleList(self.DecodeConvs)
        self.FinalOct =  OctConv(num_inputs, num_inputs, alphax = 0.5, alphay = 1)
################## End Section 2 ##########################

    def forward(self, x):
        INPUT = x
        x_h = x
        x_l = 0
        names = self.__dict__
        x_h, x_l, = self.InputOct(x_h, x_l)
        temp = list(range(1, self.EncodingDepth + 1))
        for encodingLayer in temp:
            temp_conv = self.EncodeConvs[encodingLayer - 1]
            x_h, x_l = temp_conv(x_h, x_l)
            #print('EncodeConv' + str(encodingLayer) + str(x.size()))
            names['EncodeX' + str(encodingLayer)] = x_h, x_l
            ## corresponding to matlab, no inter-section in maxpooling. 
            x_h = F.max_pool3d(x_h, 2)
            x_l = F.max_pool3d(x_l, 2)

        x_h, x_l = self.MidConv1(x_h, x_l)
        #print('Mid' + str(encodingLayer) + str(x.size()))

        for decodingLayer in temp:
            temp_conv = self.DecodeConvs[decodingLayer - 1]
            x_h2, x_l2 = names['EncodeX' + str(self.EncodingDepth - decodingLayer + 1)]
            x_h, x_l = temp_conv(x_h, x_l, x_h2, x_l2)
            #print('DecodeConv' + str(decodingLayer) + str(x.size()))

        x = self.FinalOct(x_h, x_l)
        x = x + INPUT
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
    xqsm = xQSM(2)
    xqsm.apply(weights_init)
    print(xqsm.state_dict()['FinalOct.FinalConv.bias'])
    print(get_parameter_number(xQSM))
    x = torch.randn(1,1,48,48,48, dtype=torch.float)
    print('input' + str(x.size()))
    print(x.dtype)
    y = xqsm(x)
    print('output'+str(y.size()))

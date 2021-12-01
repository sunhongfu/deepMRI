import torch
import torch.nn as nn 
import torch.nn.functional as F
from DC_layer import * 

class DCRNet(nn.Module):
    def __init__(self, EncodingDepth = 5):
        super(DCRNet, self).__init__()
        initial_num_layers = 64
        No_channels = 1
        self.EncodingDepth = EncodingDepth
        self.init1 = CConv2d_BN_RELU(No_channels, initial_num_layers)

        self.midLayers = []
        temp = list(range(1, EncodingDepth + 1))
        for encodingLayer in temp:
            self.midLayers.append(Basic_block(initial_num_layers, initial_num_layers))
        self.midLayers = nn.ModuleList(self.midLayers)

        self.FinalConv =  CConv2d(initial_num_layers, No_channels)

        self.dc = DataConsistencyInKspace()

    def forward(self, x_r, x_i, k0_r = 0, k0_i = 0, mask = 0):
        INPUT_r = x_r
        INPUT_i = x_i
        x_r, x_i = self.init1(x_r, x_i)

        temp = list(range(1, self.EncodingDepth + 1))
        for encodingLayer in temp:
            temp_conv = self.midLayers[encodingLayer - 1]
            x_r, x_i = temp_conv(x_r, x_i)
        
        x_r, x_i = self.FinalConv(x_r, x_i)
        x_r = x_r + INPUT_r  ## 
        x_i = x_i + INPUT_i ## 

        """ alternative data consistency; 
        ini_r = x_r
        ini_i = x_i
        
        x_r, x_i = self.dc(x_r, x_i, k0_r, k0_i, mask)
        """
        return x_r, x_i



class Basic_block(nn.Module):
    def __init__(self, num_in, num_out):
        super(Basic_block, self).__init__()
        self.cconv1 = CConv2d_BN_RELU(num_in, num_out)
        self.cconv2 = CConv2d_BN_RELU(num_out, num_out)

    def forward(self, x_r, x_i):
        INPUT_r = x_r
        INPUT_i = x_i
        x_r, x_i = self.cconv1(x_r, x_i)
        x_r = x_r + INPUT_r
        x_i = x_i + INPUT_i
        x_r, x_i = self.cconv2(x_r, x_i)
        return x_r, x_i

## complex convolution; 
class CConv2d_BN_RELU(nn.Module):
    def __init__(self, num_in, num_out, ks = 3, pad = 1):
        super(CConv2d_BN_RELU, self).__init__()
        self.conv_r = nn.Conv2d(num_in, num_out, ks, padding= pad)
        self.conv_i = nn.Conv2d(num_in, num_out, ks, padding= pad)
        self.bn_r = nn.BatchNorm2d(num_out)
        self.bn_i = nn.BatchNorm2d(num_out)
        self.relu_r = nn.ReLU(inplace = True)
        self.relu_i = nn.ReLU(inplace = True)

    def forward(self, x_r, x_i):
        x_rr = self.conv_r(x_r)
        x_ri = self.conv_i(x_r)
        x_ir = self.conv_r(x_i)
        x_ii = self.conv_i(x_i)
        x_r = x_rr - x_ii 
        x_i = x_ri + x_ir
        x_r = self.bn_r(x_r)
        x_i = self.bn_i(x_i)
        x_r = self.relu_r(x_r)
        x_i = self.relu_i(x_i)
        return x_r, x_i


## complex convolution; 
class CConv2d(nn.Module):
    def __init__(self, num_in, num_out, ks = 1, pad = 0, bs = True):
        super(CConv2d, self).__init__()
        self.conv_r = nn.Conv2d(num_in, num_out, ks, bias = bs, padding= pad)
        self.conv_i = nn.Conv2d(num_in, num_out, ks, bias = bs, padding= pad)
        
    def forward(self, x_r, x_i):
        x_rr = self.conv_r(x_r)
        x_ri = self.conv_i(x_r)
        x_ir = self.conv_r(x_i)
        x_ii = self.conv_i(x_i)
        x_r = x_rr - x_ii 
        x_i = x_ri + x_ir

        return x_r, x_i



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(m.bias)   
    if isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)   

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

#################### For Code Test ##################################
## before running the training codes, verify the network architecture. 
if __name__ == '__main__':
    dcrnet = DCRNet(5)
    dcrnet.apply(weights_init)
    print(dcrnet.state_dict)
    print(get_parameter_number(dcrnet))
    x_r = torch.randn(2,1,48,48, dtype=torch.float)
    x_i = torch.randn(2,1,48,48, dtype=torch.float)
    print('input' + str(x_r.size()))
    print(x_r.dtype)
    ini_r, ini_i, y_r, y_i = dcrnet(x_r, x_i, x_r, x_i, x_r)
    print(torch.max(y_r - y_i))
    print('output'+str(y_r.size()))

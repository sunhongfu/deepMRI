#################### ResNet Blocks ########################
## import defaults packages 
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np 


class LoTLayer(nn.Module):
    def __init__(self, conv_x):
        super(LoTLayer, self).__init__()
        self.conv_x = nn.Parameter(conv_x.clone(),requires_grad=True)
        self.conv_y = nn.Parameter(conv_x.repeat([15, 1, 1, 1, 1]).clone(), requires_grad=True)

    def forward(self, phi, mask, TE, B0):
        
        ## mask: chi mask
        expPhi_r = torch.cos(phi)
        expPhi_i = torch.sin(phi)

        a_r = self.LG(expPhi_r, self.conv_x)  ## first term. (delta(1j * phi)
        a_i = self.LG(expPhi_i, self.conv_x)  

        ## b_r = a_r * expPhi_r + a_i * expPhi_i    ## first term  multiply the second term (exp(-1j * phi) = cos(phi) - j * sin(phi)))
        b_i = a_i * expPhi_r - a_r * expPhi_i

        b_i = b_i * mask

        ## normalization 
        b_i = b_i / (B0 * TE)
        b_i = b_i * (3 * 20e-3)

        a_r = self.LG(expPhi_r, self.conv_y)  ## first term. (delta(1j * phi)
        a_i = self.LG(expPhi_i, self.conv_y)  

        ## b_r = a_r * expPhi_r + a_i * expPhi_i    ## first term  multiply the second term (exp(-1j * phi) = cos(phi) - j * sin(phi)))
        d_i = a_i * expPhi_r - a_r * expPhi_i

        d_i = d_i * mask

        ## normalization 
        d_i = d_i / (B0 * TE)
        d_i = d_i * (3 * 20e-3)

        return b_i, d_i

    def LG(self, tensor_image, weight):
        out = F.conv3d(tensor_image, weight, bias=None,stride=1,padding=1)  ## 3 * 3 kernel, padding 1 zeros. 

        h, w, d = out.shape[2], out.shape[3], out.shape[4]
        out[:, :, [0, h-1], :,:] = 0
        out[:, :, :, [0, w-1],:] = 0
        out[:, :, :, :, [0, d-1]] = 0
        return out

## HWD embedding 
class FinalZPrjs_Conv_Layer(nn.Module):
    def __init__(self):
        super(FinalZPrjs_Conv_Layer, self).__init__()
        self.embed_hwd = FinalHWD_Embedding_Layer()
        self.embed_c = FinalC_Embeding_Layer(1)
        self.FinalConv = nn.Conv3d(16, 1, 1, stride = 1, padding = 0)

    def forward(self, x, z_prjs):
        x_conv = self.FinalConv(x)
        x = self.embed_hwd(x, z_prjs)
        x = self.embed_c(x, z_prjs)
        x_all = x_conv + x
        return x_all

## C embedding 
class FinalC_Embeding_Layer(nn.Module):
    def __init__(self, num_outputs = 1):
        super(FinalC_Embeding_Layer, self).__init__()
        self.num_out = num_outputs
        self.embd = nn.Sequential(nn.SiLU(), nn.Linear(5, 2 * 1))
        
    def forward(self, x, z_prjs):
        x = x.repeat(1, self.num_out, 1, 1, 1)
        scale_shift = self.embd(z_prjs)
        c_embd = self.embd(z_prjs)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        scale, shift = c_embd.chunk(2, dim = 1)
        x = x * scale + shift
        return x

## HWD embedding 
class FinalHWD_Embedding_Layer(nn.Module):
    def __init__(self):
        super(FinalHWD_Embedding_Layer, self).__init__()
        self.embd = nn.Sequential(nn.SiLU(), nn.Linear(5, 1))

    def forward(self, x, z_prjs):

        weight = self.embd(z_prjs).view(x.shape[0], 1, 1, 1, 1, 1)
        
        out = torch.zeros((x.shape[0], 1, x.shape[2],  x.shape[3],  x.shape[4]), device = x.device)

        for i in range(0, x.shape[0]):
            tmp_out = F.conv3d(x[i, :, :, :, :].unsqueeze(0), weight[i, :, :, :, :].repeat(1, x.shape[1], 1, 1, 1), bias=None,stride=1,padding=0)  ## 3 * 3 kernel, padding 1 zeros. 
            out[i, :, :, :, :] = tmp_out.squeeze(0)
        return out



## HWD embedding 
class ZPrjs_Conv_Layer(nn.Module):
    def __init__(self, num_in, num_out):
        super(ZPrjs_Conv_Layer, self).__init__()
        self.embed_hwd = HWD_Embedding_Layer()
        self.embed_c = C_Embeding_Layer(num_out)

        self.EncodeConv = nn.Sequential(
            nn.Conv3d(num_in, num_out, 3, padding=1)
        )
        self.BN_RELU = nn.Sequential(
            nn.BatchNorm3d(num_out),
            nn.ReLU(inplace = True)
        )

    def forward(self, x, z_prjs):
        x_conv = self.EncodeConv(x)
        x = self.embed_hwd(x, z_prjs)
        x = self.embed_c(x, z_prjs)
        x_all = x_conv + x
        x = self.BN_RELU(x_all)
        return x


## HWD embedding 
class HWD_Embedding_Layer(nn.Module):
    def __init__(self):
        super(HWD_Embedding_Layer, self).__init__()
        self.embd = nn.Sequential(nn.SiLU(), nn.Linear(5, 10), nn.SiLU(), nn.Linear(10, 27))

    def forward(self, x, z_prjs):

        weight = self.embd(z_prjs).view(x.shape[0], 1, 1, 3, 3, 3)
        
        out = torch.zeros((x.shape[0], 1, x.shape[2],  x.shape[3],  x.shape[4]), device = x.device)

        for i in range(0, x.shape[0]):
            tmp_out = F.conv3d(x[i, :, :, :, :].unsqueeze(0), weight[i, :, :, :, :].repeat(1, x.shape[1], 1, 1, 1), bias=None,stride=1,padding=1)  ## 3 * 3 kernel, padding 1 zeros. 
            out[i, :, :, :, :] = tmp_out.squeeze(0)
        return out

## C embedding 
class C_Embeding_Layer(nn.Module):
    def __init__(self, num_outputs):
        super(C_Embeding_Layer, self).__init__()
        self.num_out = num_outputs
        self.embd = nn.Sequential(nn.SiLU(), nn.Linear(5, 10), nn.SiLU(), nn.Linear(10, 2 * num_outputs))
        
    def forward(self, x, z_prjs):
        x = x.repeat(1, self.num_out, 1, 1, 1)
        scale_shift = self.embd(z_prjs)
        c_embd = self.embd(z_prjs)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        scale, shift = c_embd.chunk(2, dim = 1)
        x = x * scale + shift
        return x

## basic blocks
class EncodingBlocks(nn.Module):
    def __init__(self, num_in, num_out):
        super(EncodingBlocks, self).__init__()
        self.embd_layer1 = ZPrjs_Conv_Layer(num_in, num_out)
        self.embd_layer2 = ZPrjs_Conv_Layer(num_out, num_out)

    def forward(self, x, z_prjs):
        x = self.embd_layer1(x, z_prjs)
        x = self.embd_layer2(x, z_prjs)
        return x


class MidBlocks(nn.Module):
    def __init__(self, num_ch):
        super(MidBlocks, self).__init__()
        self.embd_layer1 = ZPrjs_Conv_Layer(num_ch, 2 * num_ch)
        self.embd_layer2 = ZPrjs_Conv_Layer(2 * num_ch, num_ch)

    def forward(self, x, z_prjs):
        x = self.embd_layer1(x, z_prjs)
        x = self.embd_layer2(x, z_prjs)
        return x


class DecodingBlocks(nn.Module):
    def __init__(self, num_in, num_out, bilinear=False):
        super(DecodingBlocks, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.BatchNorm3d(num_in),
                nn.ReLU(inplace = True)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose3d(num_in, num_in, 2, stride = 2),
                nn.BatchNorm3d(num_in),
                nn.ReLU(inplace = True)
            )
        self.embd_layer1 = ZPrjs_Conv_Layer(2 * num_in, num_in)
        self.embd_layer2 = ZPrjs_Conv_Layer(num_in, num_out)

    def forward(self, x1, x2, z_prjs):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim = 1)
        x = self.embd_layer1(x, z_prjs)
        x = self.embd_layer2(x, z_prjs)
        return x


if __name__ == '__main__':
    LGOP =  scio.loadmat("3D_Laplacian_Operator.mat", verify_compressed_data_integrity=False)
    conv_op = LGOP['LM']
    conv_op = np.array(conv_op)
    conv_op = torch.from_numpy(conv_op)
    conv_op = conv_op.float()
    conv_op = torch.unsqueeze(conv_op, 0)
    conv_op = torch.unsqueeze(conv_op, 0)

    matMask =  scio.loadmat("test_mask.mat", verify_compressed_data_integrity=False)
    mask = matMask['BET_mask_ero']
    mask = np.array(mask)
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask = torch.unsqueeze(mask, 0)
    mask = torch.unsqueeze(mask, 0)

    matTest = scio.loadmat("test_phase_py.mat", verify_compressed_data_integrity=False)
    image = matTest['wph_sim']
    image = np.array(image)
    image = torch.from_numpy(image)
    image = image.float()
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)

    image = image

    LPLayer = LapLayer(conv_op)
    LPLayer.eval()
    
    recons, _ = LPLayer(image,mask)

    recons = recons.to('cpu')
    recons = recons.detach().numpy()

    print('Saving results')
    path = "recon" + "test_py.mat"
    scio.savemat(path, {'recons':recons})

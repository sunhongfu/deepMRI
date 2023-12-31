##################### octave convolutions  ########################
# import defaults packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# basic blocks
class LapLayer(nn.Module):
    def __init__(self, conv_x):
        super(LapLayer, self).__init__()
        self.conv_x = nn.Parameter(conv_x.clone(),requires_grad=True)
        self.conv_y = nn.Parameter(conv_x.repeat([15, 1, 1, 1, 1]).clone(), requires_grad=True)

    def forward(self, phi, mask, TE):
        
        ## mask: chi mask
        expPhi_r = torch.cos(phi)
        expPhi_i = torch.sin(phi)

        a_r = self.LG(expPhi_r, self.conv_x)  ## first term. (delta(1j * phi)
        a_i = self.LG(expPhi_i, self.conv_x)  

        ## b_r = a_r * expPhi_r + a_i * expPhi_i    ## first term  multiply the second term (exp(-1j * phi) = cos(phi) - j * sin(phi)))
        b_i = a_i * expPhi_r - a_r * expPhi_i

        b_i = b_i * mask

        ## normalization 
        b_i = b_i / (3 * TE)
        b_i = b_i * (3 * 20e-3)

        a_r = self.LG(expPhi_r, self.conv_y)  ## first term. (delta(1j * phi)
        a_i = self.LG(expPhi_i, self.conv_y)  

        ## b_r = a_r * expPhi_r + a_i * expPhi_i    ## first term  multiply the second term (exp(-1j * phi) = cos(phi) - j * sin(phi)))
        d_i = a_i * expPhi_r - a_r * expPhi_i

        d_i = d_i * mask

        ## normalization 
        d_i = d_i / (3 * TE)
        d_i = d_i * (3 * 20e-3)

        return b_i, d_i

    def LG(self, tensor_image, weight):
        out = F.conv3d(tensor_image, weight, bias=None,stride=1,padding=1)  ## 3 * 3 kernel, padding 1 zeros. 

        h, w, d = out.shape[2], out.shape[3], out.shape[4]
        out[:, :, [0, h-1], :,:] = 0
        out[:, :, :, [0, w-1],:] = 0
        out[:, :, :, :, [0, d-1]] = 0
        return out

## z_prjs embedding 
class Embeding_Layer(nn.Module):
    def __init__(self, num_outputs):
        super(Embeding_Layer, self).__init__()
        self.embd = nn.Sequential(nn.Linear(5, 10), nn.SiLU(), nn.Linear(10, 2 * num_outputs))
        
    def forward(self, x, z_prjs):
        scale_shift = self.embd(z_prjs)
        c_embd = self.embd(z_prjs)[:, :, None, None, None].repeat(1, 1, x.shape[-3], x.shape[-2], x.shape[-1])
        scale, shift = c_embd.chunk(2, dim = 1)
        x = x * (1 + scale) + shift
        return x

class OctEncodingBlocks(nn.Module):
    def __init__(self, num_in, num_out, alphax=0.5, alphay=0.5):
        super(OctEncodingBlocks, self).__init__()
        self.EncodeConv1 = OctConv(num_in, num_out, alphax, alphay)
        if alphax == 1:
            alphax = 0.5
        self.EncodeConv2 = OctConv(num_out, num_out, alphax, alphay)

    def forward(self, x_h, x_l, z_prjs):
        y_h, y_l = self.EncodeConv1(x_h, x_l, z_prjs)
        y_h, y_l = self.EncodeConv2(y_h, y_l, z_prjs)
        return y_h, y_l


class OctMidBlocks(nn.Module):
    def __init__(self,  num_ch, alphax=0.5, alphay=0.5):
        super(OctMidBlocks, self).__init__()
        self.MidConv1 = OctConv(num_ch, 2 * num_ch, alphax, alphay)
        self.MidConv2 = OctConv(2 * num_ch, num_ch, alphax, alphay)

    def forward(self, x_h, x_l, z_prjs):
        y_h, y_l = self.MidConv1(x_h, x_l, z_prjs)
        y_h, y_l = self.MidConv2(y_h, y_l, z_prjs)
        return y_h, y_l


class OctDecodingBlocks(nn.Module):
    def __init__(self,  num_in, num_out, alphax=0.5, alphay=0.5, bilinear=False):
        super(OctDecodingBlocks, self).__init__()
        if bilinear:
            self.up = OctUp(num_in, alphax=0.5,  bilinear=True)
        else:
            self.up = OctUp(num_in, alphax=0.5,  bilinear=False)
        self.DecodeConv1 = OctConv(2 * num_in, num_in, alphax, alphay)
        self.DecodeConv2 = OctConv(num_in, num_out, alphax, alphay)

    def forward(self, x_h1, x_l1, x_h2, x_l2, z_prjs):
        x_h1, x_l1 = self.up(x_h1, x_l1)
        x_h = torch.cat([x_h1, x_h2], dim=1)
        x_l = torch.cat([x_l1, x_l2], dim=1)
        y_h, y_l = self.DecodeConv1(x_h, x_l, z_prjs)
        y_h, y_l = self.DecodeConv2(y_h, y_l, z_prjs)
        return y_h, y_l


class OctUp(nn.Module):
    def __init__(self, num_in, alphax=0.5,  bilinear=False, hasbias=True):
        super(OctUp, self).__init__()
        self.In_H = int(num_in * alphax)
        self.In_L = (num_in - self.In_H)
        if bilinear:
            pass
        else:
            self.up_HH = nn.ConvTranspose3d(
                self.In_H, self.In_H, 2, stride=2, bias=hasbias)
            self.up_LH = nn.ConvTranspose3d(
                self.In_L, self.In_H, 4, stride=4, bias=hasbias)
            self.BN_H = nn.BatchNorm3d(self.In_H)
            self.ReLU_H = nn.ReLU(inplace=True)
            self.up_LL = nn.ConvTranspose3d(
                self.In_L, self.In_L, 2, stride=2, bias=hasbias)
            self.up_HL = nn.Conv3d(self.In_H, self.In_L,
                                   3, padding=1, bias=hasbias)
            self.BN_L = nn.BatchNorm3d(self.In_L)
            self.ReLU_L = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l):
        y_h1 = self.up_HH(x_h)
        y_h2 = self.up_LH(x_l)
        y_l1 = self.up_LL(x_l)
        y_l2 = self.up_HL(x_h)
        y_l = y_l1 + y_l2
        y_h = y_h1 + y_h2
        y_h = self.BN_H(y_h)
        y_h = self.ReLU_H(y_h)
        y_l = self.BN_L(y_l)
        y_l = self.ReLU_L(y_l)
        return y_h, y_l


class OctConv(nn.Module):
    def __init__(self, num_in, num_out, alphax, alphay, ks=3, pd=1, hasbias=True):
        super(OctConv, self).__init__()
        # This calss Defines the OctConv operation.
        # define feature decomposition
        self.In_H = int(num_in * alphax)
        self.In_L = (num_in - self.In_H)
        self.Out_H = int(num_out * alphay)
        self.Out_L = (num_out - self.Out_H)
        self.alphax = alphax
        self.alphay = alphay
        # define operations
        if alphax == 1:
            # for the input layer.
            self.convHH = nn.Conv3d(
                self.In_H, self.Out_H, ks, padding=pd, bias=hasbias)
            self.convHL = nn.Conv3d(
                self.In_H, self.Out_L, ks, padding=pd, bias=hasbias)
            self.BN_HH = nn.BatchNorm3d(self.Out_H)
            self.BN_HL = nn.BatchNorm3d(self.Out_L)
            self.ReLU_H = nn.ReLU(inplace=True)
            self.ReLU_L = nn.ReLU(inplace=True)
            self.embd_layer = Embeding_Layer(self.Out_H)
            self.embd_layer2 = Embeding_Layer(self.Out_L)
        elif alphay == 1:
            # only for the final output layer
            self.convHH = nn.Conv3d(
                self.In_H, self.Out_H, ks, padding=pd, bias=hasbias)
            self.convLH = nn.Conv3d(
                self.In_L, self.Out_H, ks, padding=pd, bias=hasbias)
            self.up = nn.ConvTranspose3d(
                self.Out_H, self.Out_H, 2, stride=2, bias=hasbias)
            self.BN_HH = nn.BatchNorm3d(self.Out_H)
            self.BN_LH = nn.BatchNorm3d(self.Out_H)
            self.ReLU_H = nn.ReLU(inplace=True)
            self.FinalConv = nn.Conv3d(
                self.Out_H, 1, 1, stride=1, padding=0, bias=hasbias)
            self.embd_layer = Embeding_Layer(self.Out_H)
            self.embd_layer2 = Embeding_Layer(self.Out_H)
        else:
            # mid layers
            self.convHH = nn.Conv3d(
                self.In_H, self.Out_H, ks, padding=pd, bias=hasbias)
            self.convLL = nn.Conv3d(
                self.In_L, self.Out_L, ks, padding=pd, bias=hasbias)
            self.convHL = nn.Conv3d(
                self.In_H, self.Out_L, ks, padding=pd, bias=hasbias)
            self.convLH = nn.Conv3d(
                self.In_L, self.Out_H, ks, padding=pd, bias=hasbias)
            self.up = nn.ConvTranspose3d(
                self.Out_H, self.Out_H, 2, stride=2, bias=hasbias)
            self.BN_HH = nn.BatchNorm3d(self.Out_H)
            self.BN_LH = nn.BatchNorm3d(self.Out_H)
            self.ReLU_H = nn.ReLU(inplace=True)
            self.BN_LL = nn.BatchNorm3d(self.Out_L)
            self.BN_HL = nn.BatchNorm3d(self.Out_L)

            self.embd_layer = Embeding_Layer(self.Out_H)
            self.embd_layer2 = Embeding_Layer(self.Out_L)
            self.embd_layer3 = Embeding_Layer(self.Out_H)
            self.embd_layer4 = Embeding_Layer(self.Out_L)
            self.ReLU_L = nn.ReLU(inplace=True)

    def forward(self, x_h, x_l, z_prjs):
        ## Y_H = conv(H) + upsample(conv(L))
        ## Y_L = conv(L) + conv(avgpool(H))
        if self.alphax == 1:
            y_h = self.convHH(x_h)
            y_h = self.BN_HH(y_h)

            y_h = self.embd_layer(y_h, z_prjs)

            y_l = F.avg_pool3d(x_h, 2)
            y_l = self.convHL(y_l)
            y_l = self.BN_HL(y_l)

            y_l  = self.embd_layer2(y_l , z_prjs)
            ## BN and ReLU()
            y_h = self.ReLU_H(y_h)
            y_l = self.ReLU_L(y_l)
            return y_h, y_l
        elif self.alphay == 1:
            y_h1 = self.convHH(x_h)
            y_h1 = self.BN_HH(y_h1)

            y_h1 = self.embd_layer(y_h1, z_prjs)

            y_h2 = self.convLH(x_l)
            y_h2 = self.up(y_h2)
            y_h2 = self.BN_LH(y_h2)

            y_h2  = self.embd_layer2(y_h2 , z_prjs)
            ## BN and ReLU()
            y_h = y_h1 + y_h2
            y_h = self.ReLU_H(y_h)
            # final Output, Convolution without relu,
            y_h = self.FinalConv(y_h)
            return y_h
        else:
            # high fre group
            # first convLH then Upsample
            y_h1 = self.convHH(x_h)
            y_h2 = self.convLH(x_l)
            y_h2 = self.up(y_h2)
            y_h1 = self.BN_HH(y_h1)

            y_h1  = self.embd_layer(y_h1 , z_prjs)

            y_h2 = self.BN_LH(y_h2)

            y_h2  = self.embd_layer3(y_h2 , z_prjs)
            # print(y_h2.size())
            y_l1 = self.convLL(x_l)
            y_l2 = F.avg_pool3d(x_h, 2)
            y_l2 = self.convHL(y_l2)
            y_l1 = self.BN_LL(y_l1)

            y_l1  = self.embd_layer2(y_l1 , z_prjs)

            y_l2 = self.BN_HL(y_l2)

            y_l2  = self.embd_layer4(y_l2 , z_prjs)
            # final addition
            y_h = y_h1 + y_h2
            y_l = y_l1 + y_l2
            ## BN and ReLU()
            y_h = self.ReLU_H(y_h)
            y_l = self.ReLU_L(y_l)
            return y_h, y_l

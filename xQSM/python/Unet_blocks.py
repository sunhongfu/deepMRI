#################### ResNet Blocks ########################
## import defaults packages 
import torch
import torch.nn as nn

## basic blocks
class EncodingBlocks(nn.Module):
    def __init__(self, num_in, num_out):
        super(EncodingBlocks, self).__init__()
        self.EncodeConv = nn.Sequential(
            nn.Conv3d(num_in, num_out, 3, padding=1),
            nn.BatchNorm3d(num_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(num_out, num_out, 3, padding=1),
            nn.BatchNorm3d(num_out),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.EncodeConv(x)
        return x


class MidBlocks(nn.Module):
    def __init__(self, num_ch):
        super(MidBlocks, self).__init__()
        self.MidConv = nn.Sequential(
            nn.Conv3d(num_ch, 2 * num_ch, 3, padding=1),
            nn.BatchNorm3d(2 * num_ch),
            nn.ReLU(inplace = True),
            nn.Conv3d(2 * num_ch, num_ch, 3, padding=1),
            nn.BatchNorm3d(num_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.MidConv(x)
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
        self.DecodeConv = nn.Sequential(
            nn.Conv3d(2 * num_in, num_in, 3, padding=1),
            nn.BatchNorm3d(num_in),
            nn.ReLU(inplace = True),
            nn.Conv3d(num_in, num_out, 3, padding=1),
            nn.BatchNorm3d(num_out),
            nn.ReLU(inplace = True)
        )
          
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim = 1)
        x = self.DecodeConv(x)
        return x



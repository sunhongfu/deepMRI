import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as FFT 
import scipy.io as scio
import numpy as np


## Fourier Transforms
def FFT2D(x):
    return FFT.fftshift(FFT.fft2(FFT.fftshift(x, dim=(-2, -1))), dim=(-2, -1))

def IFFT2D(x):
    return FFT.ifftshift(FFT.ifft2(FFT.ifftshift(x,dim=(-2, -1))), dim=(-2, -1))


## data consistency in kspace; 
def data_consistency(self, k, k0, mask, WF):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    WF   - learnable Weighting factor/noise level
    """
    out = (1 - mask) * k + mask * k0 * (1 - WF) + mask * k * WF
    return out    

class DataConsistencyInKspace(nn.Module):
    def __init__(self):
        super(DataConsistencyInKspace, self).__init__()
        self.WF = nn.Parameter(torch.zeros(1), requires_grad=True) ## learnable noise level; 

    def forward(self, x_r, x_i, k0_r, k0_i, mask):
        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        WF   - learnable Weighting factor/noise level
        """
        k0 = torch.cat([k0_r, k0_i], dim = 1)
        k0 = k0.permute(0, 2, 3, 1) ## Nb * Nx * Ny * 2
        ## k0 = torch.view_as_complex(k0) # Nb * Nx * Ny, complex

        mask = mask.permute(0, 2, 3, 1) # Nb * Nx * Ny * 1, real
        mask = mask.expand(-1, -1, -1, 2)

        x = torch.cat([x_r, x_i], dim = 1)
        x = x.permute(0, 2, 3, 1).contiguous() ## Nb * Nx * Ny * 2 
        x = torch.view_as_complex(x) # for FFT. 

        k = IFFT2D(x)
        k = torch.view_as_real(k).contiguous()  #Nb * Nx * Ny * 2 

        out = data_consistency(self, k, k0, mask, self.WF) ## out shape: Nb * Nx * Ny * 2 real; 

        out = torch.view_as_complex(out.contiguous()) # Nb * Nx * Ny, complex

        x_res = FFT2D(out) ## complex nb * Nx * Ny 

        x_res = torch.view_as_real(x_res).contiguous()  ## real nb * Nx * Ny * 2

        x_res = x_res.permute(0, 3, 1, 2)
        x_r = x_res[:, 0, :, :]
        x_i = x_res[:, 1, :, :]
        
        return torch.unsqueeze(x_r, 1), torch.unsqueeze(x_i, 1)


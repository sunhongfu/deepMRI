import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import torch.fft as FFT 

def TVLoss(x, reg = 1e-4):
    ## x：network output, Nb * nc * H * L * D
    TV_H = torch.sum(torch.abs(x[:,:,1:,:,:] - x[:,:,:-1,:,:]))
    TV_L = torch.sum(torch.abs(x[:,:,:,1:,:] - x[:,:,:,:-1,:]))
    TV_D = torch.sum(torch.abs(x[:,:,:,:,1:] - x[:,:,:,:,:-1]))

    return reg * (TV_H + TV_L + TV_D)

def DataFidelity(chi, Dipole):
    ## device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = chi.size(2)
    L = chi.size(3)
    D = chi.size(4)

    chi = F.pad(chi, (32, 32, 32, 32, 32, 32), 'constant', 0)

    ##print(chi.size())

    x_k = FFT.fftn(chi, dim = (-3, -2, -1))
    #x_k = x_k.to(device)
    #D = D.to(device)
    x_k = x_k * Dipole## forward calculation in k-space. 
    x_img = FFT.ifftn(x_k, dim = (-3, -2, -1))

    x_img = torch.real(x_img)

    x_img = x_img[:, :, 32:32 + H, 32: 32 + L, 32: 32 + D]
    ####print(torch.max(x_img))

    return x_img

"""
def iFFT3D(x_k):
    x_k = x_k.permute(0, 2, 3, 4, 1)  ## FFT reconstruciton block. 
    ## x_iR = x_iR * 1e3
    
    x_img = torch.ifft(x_k, 3)
    x_img = x_img[:,:,:,:,0]  ## get the real channel. 0： real channel, 1, imaginary channel.
    x_img = torch.unsqueeze(x_img, 1) # reshape as Nb * 1 * H * L * D; Img; 

    return x_img

def FFT3D(x_img):

    nb = x_img.size(0)
    H = x_img.size(2)
    L = x_img.size(3)
    D = x_img.size(4)

    x = torch.zeros(nb, H, L, D, 2)
    x[:,:,:,:,0] = torch.squeeze(x_img, 1)  # real channel: chi_img, imaginary: zeros; 
    
    x_k = torch.fft(x, 3) ## 3D FFT. Nb * H * L * D * 2 
    x_k = x_k.permute(0, 4, 1, 2, 3)  ## re-shape. Make the multiplication with diplle kernel (kspace) is broadcastable.
    ## input_k = input_k / 1e3   ## noisy k-space field data. 
    
    return x_k
"""

if __name__ == '__main__':
    matTest =  scio.loadmat("test_df.mat", verify_compressed_data_integrity=False)
    image = matTest['test']
    image = np.array(image)
    image = torch.from_numpy(image)
    print(image.size())
    image = image.float()
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)

    matD = scio.loadmat("Dipole_128.mat", verify_compressed_data_integrity=False)
    D = matD['D']

    D = np.array(D)
    D = torch.from_numpy(D)
    print(torch.max(D))
    D = D.float()

    lfs = DataFidelity(image, D)

    lfs = torch.squeeze(lfs, 0)
    lfs = torch.squeeze(lfs, 0)

    lfs = lfs.numpy()

    print('Saving lfs')
    path = 'lfs_tested_py.mat'
    scio.savemat(path, {'lfs':lfs})

    







import os
import numpy as np
import random
import torch
from torch.utils import data
import scipy.io as scio


class DataSet(data.Dataset):
    def __init__(self, root, list_path, mask, transform=None):
        super(DataSet,self).__init__()
        self.root = root
        self.list_path = list_path
        self.Mask = mask  ## subsampling mask; file 'Real_Mask_Acc4_forTraining.mat' in current folder
        self.img_ids = []

        self.Prob = torch.tensor(0.95)   ## Addding Noise to the training datasets by modifying this number; probability to add noise: (1 - self.Prob)
        self.SNRs = torch.tensor([60, 40, 20])  # Noise SNRs, can be extended to more levels, based on your own data and system. 
        ## get the number of files. 
        # self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        # print(self.img_ids)
        ## get all fil names, preparation for get_item. 
        ## for example, we have two files: 
        ## 102-field.nii for input, and 102-phantom for label; 
        ## then image id is 102, and then we can use string operation
        ## to get the full name of the input and label files. 
        self.files = []
        for name in self.img_ids:
            label_file = self.root + ("/k_full_2d_data_for_Training/k_full_2d_%s.mat" % name)
            self.files.append({
                #"img": img_file,
                "label": label_file,
                "name": name
            })
        ## sprint(self.files)

    def __len__(self):
        return len(self.files)
 
 
    def __getitem__(self, index):
        datafiles = self.files[index]
 
        '''load the datas'''
        name = datafiles["name"]
        ## nifti read codes. 
        matLabel = scio.loadmat(datafiles["label"], verify_compressed_data_integrity=False)
        label = matLabel['k_full_2d']

        label = np.array(label)

        label = np.fft.fftshift(label)
        label = np.fft.fft2(label)
        ###mag = np.absolute(label)
        label = label / 30  # normalization

        label_r = np.real(label)
        label_i = np.imag(label)

        label_r = torch.from_numpy(label_r)
        label_i = torch.from_numpy(label_i)

        label_noise_r = label_r
        label_noise_i = label_i

        ### add noise into the input images; 
        tmp = torch.rand(1)
        if tmp > self.Prob:
            #print('noise')
            tmp_idx = torch.randint(3, (1,1))
            tmp_SNR = self.SNRs[tmp_idx]
            label_noise_r = AddNoise(label_r, tmp_SNR)
            label_noise_i = AddNoise(label_i, tmp_SNR)

        label_noise = label_noise_r + 1j * label_noise_i
        label_noise = label_noise.numpy()

        label_noise = np.fft.ifft2(label_noise)
        label_noise = np.fft.ifftshift(label_noise)

        image = np.multiply(label_noise, self.Mask)

        k0 = image

        # convert to images
        image = np.fft.fftshift(image)
        image = np.fft.fft2(image)

        image_r = np.real(image)
        image_i = np.imag(image)

        k0_i = np.imag(k0)
        k0_r = np.real(k0)

        mask = np.real(self.Mask)
        ## convert the image data to torch.tesors and return. 
        image_r = torch.from_numpy(image_r) 
        image_i = torch.from_numpy(image_i) 

        k0_r = torch.from_numpy(k0_r)
        k0_i = torch.from_numpy(k0_i)
        mask = torch.from_numpy(mask)

        image_r = image_r.float()
        label_r = label_r.float()
        image_i = image_i.float()
        label_i = label_i.float()

        k0_r = k0_r.float()
        k0_i = k0_i.float()
        mask = mask.float()

        image_r = torch.unsqueeze(image_r, 0)
        label_r = torch.unsqueeze(label_r, 0)
        image_i = torch.unsqueeze(image_i, 0)
        label_i = torch.unsqueeze(label_i, 0)

        k0_r = torch.unsqueeze(k0_r, 0)
        k0_i = torch.unsqueeze(k0_i, 0)
        mask = torch.unsqueeze(mask, 0)
        
        return image_r, image_i, label_r, label_i, k0_r, k0_i, mask, name
 
def AddNoise(ins, SNR):
    sigPower = SigPower(ins)
    noisePower = sigPower / SNR
    noise = torch.sqrt(noisePower) * torch.randn(ins.size())
    return ins + noise

def SigPower(ins):
    ll = torch.numel(ins)
    tmp1 = torch.sum(ins ** 2)
    return torch.div(tmp1, ll)

## before formal usage, test the validation of data loader. 
if __name__ == '__main__':
    DATA_DIRECTORY = '..'
    DATA_LIST_PATH = './test_IDs.txt'
    Batch_size = 5
    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print(dst.__len__())
    # just for test,  so the mean is (0,0,0) to show the original images.
    # But when we are training a model, the mean should have another value
    # test code on personal computer: 
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=False)
    for i, Data in enumerate(trainloader):
        imgs, labels, names = Data
        if i%10 == 0:
            print(i)
            print(names)
            print(imgs.size())
            print(labels.size())

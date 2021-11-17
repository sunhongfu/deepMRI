import os
import numpy as np
import random
import torch
from torch.utils import data
import scipy.io as scio


class DataSet(data.Dataset):
    def __init__(self, root, list_path, transform=None):
        super(DataSet,self).__init__()
        self.root = root
        self.list_path = list_path
        self.img_ids = []

        self.Prob = torch.tensor(0.8)   ## 20% (1 - 0.8) probability to add noise; 
        self.SNRs = torch.tensor([50, 40, 20, 10, 5])  # Noise SNRs. 
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
            lfs_file = self.root + ("/lfs_training/lfs_patch_%s.mat" % name)
            chi_file = self.root + ("/qsm_training/chi_patch_%s.mat" % name)
            self.files.append({
                #"img": img_file,
                "lfs": lfs_file,
                "chi": chi_file,
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
        matchi = scio.loadmat(datafiles["chi"], verify_compressed_data_integrity=False)
        chi = matchi['chi_patch']

        chi = np.array(chi)

        matlfs = scio.loadmat(datafiles["lfs"], verify_compressed_data_integrity=False)
        lfs = matlfs['lfs_patch']

        lfs = np.array(lfs)

        lfs = np.array(lfs)
       
        chi = torch.from_numpy(chi) 
        lfs = torch.from_numpy(lfs) 

        chi = chi.float()
        lfs = lfs.float()

        ### add noise into the input images; 
        tmp = torch.rand(1)
        if tmp > self.Prob:
            #print('noise')
            tmp_mask = lfs != 0
            tmp_idx = torch.randint(5, (1,1))
            tmp_SNR = self.SNRs[tmp_idx]
            lfs = AddNoise(lfs, tmp_SNR)

        chi = torch.unsqueeze(chi, 0)
        lfs = torch.unsqueeze(lfs, 0)
        
        return lfs, chi, name

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

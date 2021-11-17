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
            wph_file = self.root + ("/wph_training/wph_patch_%s.mat" % name)
            lfs_file = self.root + ("/lfs_training/lfs_patch_%s.mat" % name)
            chi_file = self.root + ("/qsm_training/chi_patch_%s.mat" % name)
            mask_file = self.root + ("/mask_training/Mask_patch_%s.mat" % name)
            TE_file = self.root + ("/TE_training/TE_patch_%s.mat" % name)

            wph_hemo_file = self.root + ("/wph_hemoCal_training/wph_patch_%s.mat" % name)
            lfs_hemo_file = self.root + ("/lfs_hemoCal_training/lfs_patch_%s.mat" % name)
            chi_hemo_file = self.root + ("/qsm_hemoCal_training/QSM_patch_%s.mat" % name)
            mask_hemo_file = self.root + ("/mask_hemoCal_training/Mask_patch_%s.mat" % name)
            TE_hemo_file = self.root + ("/TE_hemoCal_training/TE_patch_%s.mat" % name)
            self.files.append({
                #"img": img_file,
                "wph": wph_file,
                "chi": chi_file,
                "lfs": lfs_file,
                "mask": mask_file,
                "TE": TE_file,
                "wph_hemo": wph_hemo_file,
                "chi_hemo": chi_hemo_file,
                "lfs_hemo": lfs_hemo_file,
                "mask_hemo": mask_hemo_file,
                "TE_hemo": TE_hemo_file,
                "name": name
            })
        ## sprint(self.files)

    def __len__(self):
        return len(self.files)
 
 
    def __getitem__(self, index):
        datafiles = self.files[index]
 
        tmp = torch.rand(1)
        if tmp < 0.6:
            '''load the datas'''
            name = datafiles["name"]
            ## nifti read codes. 
            matchi = scio.loadmat(datafiles["chi"], verify_compressed_data_integrity=False)
            chi = matchi['chi_patch']

            chi = np.array(chi)

            matlfs = scio.loadmat(datafiles["lfs"], verify_compressed_data_integrity=False)
            lfs = matlfs['lfs_patch']

            lfs = np.array(lfs)

            matMask = scio.loadmat(datafiles["mask"], verify_compressed_data_integrity=False)
            mask = matMask['Mask_patch']

            mask = np.array(mask)

            matTE = scio.loadmat(datafiles["TE"], verify_compressed_data_integrity=False)
            TE = matTE['TE_patch']

            TE = np.array(TE)

            matWph = scio.loadmat(datafiles["wph"], verify_compressed_data_integrity=False)
            wph = matWph['wph_patch']

            wph = np.array(wph)
        
            chi = torch.from_numpy(chi) 
            lfs = torch.from_numpy(lfs) 
            wph = torch.from_numpy(wph) 
            mask = torch.from_numpy(mask)
            TE = torch.from_numpy(TE)

            chi = chi.float()
            lfs = lfs.float()
            wph = wph.float()
            mask = mask.float()
            TE = TE.float()
            chi = torch.unsqueeze(chi, 0)
            lfs = torch.unsqueeze(lfs, 0)
            wph = torch.unsqueeze(wph, 0)
            mask = torch.unsqueeze(mask, 0)
            TE = torch.unsqueeze(TE, 0)
            return wph, chi, lfs, TE, mask, name
        
        if tmp >= 0.6:
            #### synthetic data #######
            name = datafiles["name"]
            matchi = scio.loadmat(datafiles["chi_hemo"], verify_compressed_data_integrity=False)
            chi_hemo = matchi['chi_patch']

            chi_hemo = np.array(chi_hemo)

            matlfs = scio.loadmat(datafiles["lfs_hemo"], verify_compressed_data_integrity=False)
            lfs_hemo = matlfs['lfs_patch']

            lfs_hemo = np.array(lfs_hemo)

            matMask = scio.loadmat(datafiles["mask_hemo"], verify_compressed_data_integrity=False)
            mask_hemo = matMask['Mask_patch']

            mask_hemo = np.array(mask_hemo)

            matTE = scio.loadmat(datafiles["TE_hemo"], verify_compressed_data_integrity=False)
            TE_hemo = matTE['TE_patch']

            TE_hemo = np.array(TE_hemo)

            matWph = scio.loadmat(datafiles["wph_hemo"], verify_compressed_data_integrity=False)
            wph_hemo = matWph['wph_patch']

            wph_hemo = np.array(wph_hemo)
        
            chi_hemo = torch.from_numpy(chi_hemo) 
            lfs_hemo = torch.from_numpy(lfs_hemo) 
            wph_hemo = torch.from_numpy(wph_hemo) 
            mask_hemo = torch.from_numpy(mask_hemo)
            TE_hemo = torch.from_numpy(TE_hemo)

            chi_hemo = chi_hemo.float()
            lfs_hemo = lfs_hemo.float()
            wph_hemo = wph_hemo.float()
            mask_hemo = mask_hemo.float()
            TE_hemo = TE_hemo.float()
            chi_hemo = torch.unsqueeze(chi_hemo, 0)
            lfs_hemo = torch.unsqueeze(lfs_hemo, 0)
            wph_hemo = torch.unsqueeze(wph_hemo, 0)
            mask_hemo = torch.unsqueeze(mask_hemo, 0)
            TE_hemo = torch.unsqueeze(TE_hemo, 0) 
            return wph_hemo, chi_hemo, lfs_hemo, TE_hemo, mask_hemo, name

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

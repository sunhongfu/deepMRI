################ network evaluation ############################
################ import packages ###############################
import torch 
import torch.nn as nn
import sys
sys.path.append('..')
from xQSM import *
import numpy as np
import nibabel as nib
from Unet import *
import scipy.io as scio
from collections import OrderedDict
# import argparse
import time
from utils import ssim, psnr
###############################################################

def ZeroPadding(Field, factor = 8):
    """ 
    ZeroPadding to make the size of the field divisible by the designated factor; 
    Field: local field map; 
    pos: the position information of padding; 
    """
    ImSize = np.shape(Field)
    UpSize = np.ceil(ImSize / factor) * factor  # calculate the padding size; 
    pos_init = np.ceil((UpSize - ImSize) / 2) 
    pos_end = pos_init + ImSize - 1
    tmp_Field = np.zeros(UpSize)
    tmp_Field[pos_init[1]:pos_end[1], pos_init[2]:pos_end[2], pos_init[3]:pos_end[3]] = Field
    Field = tmp_Field
    pos = np.zeros([3, 2])
    pos[:,0] = pos_init
    pos[:,1] = pos_end
    return Field, pos

def ZeroRemoving(Field, pos):
    """
    ZeroRemoving: inverse function of ZeroPadding;
    """
    Field = Field[pos_init[1]:pos_end[1], pos_init[2]:pos_end[2], pos_init[3]:pos_end[3]]
    return Field

def Eval(Field, NetName):
    """
    Eval(Field, Netype, Env) retunrs the QSM reconstruction of the local field map (Field)
    using a designated Network (NetName); 
    """
    with torch.no_grad(): 
        ## Network Load; 
        print('Load Pretrained Network')
        model_weights_path = NetName + '.pth'
        if 'xQSM' in NetName:
            Net = xQSM(2)
        elif 'Unet' in NetName:
            Net = Unet(2)
        else:
            sys.stderr.write('Network Type Invalid!\n')
        if torch.cuda.is_available():  ## if GPU is available; 
            """
            It is highly recommended that this demo should be run on a GPU (8G memory or higher)
            """
            Net = nn.DataParallel(Net) ## our network is trained with dataparallel wrapper;
            Net.load_state_dict(torch.load(model_weights_path))
            #Net = Net.module
            device = torch.device("cuda:0")
            Net.to(device)
            Net.eval()  ## set the model to evaluation mode
            Field = Field.to(device)
        else:
            """
            for now we dont recommend to use CPU for network inference because pytorch did not provide 
            optimized calculation for Convnets on CPU; 
            """
            weights = torch.load(model_weights_path, map_location='cpu')  
            new_state_dict = OrderedDict()
            print(new_state_dict)
            for k, v in weights.items():
                ## remove the first 7 charecters  "module." of the network weights 
                ## files to load the net into cpu, because our network is saved 
                ## as with dataparallel wrapper. 
                name = k[7:]  
                new_state_dict[name] = v
            Net.load_state_dict(new_state_dict)
            Net.eval()  ## set the model to evaluation mode
            print(Net.state_dict)
        ################ Evaluation ##################
        time_start = time.time()
        print('input image size: ',end ="")
        print(Field.size())            
        Recon = Net(Field)
        time_end = time.time()
        print('%f seconds elapsed!' % (time_end - time_start))
        Recon = torch.squeeze(Recon, 0)
        Recon = torch.squeeze(Recon, 0)
        Recon = Recon.to('cpu')  ## transfer to cpu for saving. 
        Recon = Recon.numpy()
    return Recon

def Read_nii(path):
    """
    read local field map from nifti file
    """
    nibField = nib.load(path)
    Field = nibField.get_fdata() 
    aff = nibField.affine
    Field = np.array(Field)
    return Field, aff

def Save_nii(Recon, aff, path):
    """
    save the results in nii format; 
    """
    nibRecon = nib.Nifti1Image(Recon,aff)
    nib.save(nibRecon, path) 

def Save_mat(Recon, path):
    """
    save the results in mat format; 
    """
    scio.savemat(path, {'Recon':Recon})   

if __name__ == '__main__':
    """
    Demonstration on a simulated COSMOS data; 
    """
    with torch.no_grad(): 
        ## Data Load;        
        print('Data Loading')   
        Field, aff = Read_nii('../../field_input.nii')
        print('Loading Completed')
        mask = Field != 0 
        ## note the size of the field map input needs to be divisibel by the factor
        ## otherwise 0 padding should be done first
        print('ZeroPadding')
        imSize = np.shape(Field)
        if np.mod(imSize,  8).any():
            Field, pos = ZeroPadding(Field, 8)  # ZeroPadding
        Field = torch.from_numpy(Field) 
        ## The networks in pytorch only supports inputs that are a mini-batch of samples,
        ## and not a single sample. Therefore we need  to squeeze the 3D tensor to be 
        ## a 5D tesor for model evaluation.  
        Field = torch.unsqueeze(Field, 0)
        Field = torch.unsqueeze(Field, 0)
        Field = Field.float()
        ## QSM Reconstruction 
        print('Reconstruction')
        Recon_Unet_invivo = Eval(Field, 'Unet_invivo')
        Recon_xQSM_invivo = Eval(Field, 'xQSM_invivo')
        #Recon_xQSM_syn = Eval(Field, 'xQSM_syn')
        #Recon_Unet_syn = Eval(Field, 'Unet_syn')
        if np.mod(imSize,  8).any():
            Recon_xQSM_invivo  = ZeroRemoving(Recon_xQSM_invivo , pos) # ZeroRemoving if zeropadding were performed; 
            Recon_Unet_invivo  = ZeroRemoving(Recon_Unet_invivo , pos) 
        Recon_xQSM_invivo = Recon_xQSM_invivo * mask
        Recon_Unet_invivo = Recon_Unet_invivo * mask
        ## calculate PSNR and SSIM
        label, aff = Read_nii('../../cosmos_label.nii')  # read label; 
        print('PSNR of xQSM_invivo is %f'% (psnr(Recon_xQSM_invivo, label)))
        print('PSNR of Unet_invivo is %f'% (psnr(Recon_Unet_invivo, label)))
        ## Saving Results (in .mat)
        print('saving reconstructions')
        path = './Chi_xQSM_invivo.mat' 
        Save_mat(Recon_xQSM_invivo, path)
        path = './Chi_Unet_invivo.mat' 
        Save_mat(Recon_Unet_invivo, path)
        #path = './Chi_xQSM_syn.mat' 
        #Save_mat(Recon_xQSM_syn, path)
        #path = './Chi_Unet_syn.mat' 
        #Save_mat(Recon_Unet_syn, path)
        ## or can be stored in .nii format; 
        path = 'Chi_xQSM_invivo.nii'
        Save_nii(Recon_xQSM_invivo, aff, path)
        path = 'Chi_Unet_invivo.nii'
        Save_nii(Recon_Unet_invivo, aff, path)
        #path = 'Chi_xQSM_syn.nii'
        #Save_nii(Recon_xQSM_syn, aff, path)
        #path = 'Chi_Unet_syn.nii'
        #Save_nii(Recon_xQSM_invivo, aff, path)
        

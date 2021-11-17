import torch 
import torch.nn as nn
import numpy as np
import scipy.io as scio
import time 

import sys

sys.path.append('./Model/')
from DCRNet import * 
 
if __name__ == '__main__':
    with torch.no_grad():     
        FileNo = 1 ## file identifier
        SourceDir = '../TestData/'
        No_subs = 1 ## total number of subjects to be reconstructed 
         
        print('Network Loading')
        ## load pre-trained network 
        dcrnet = DCRNet(5)
        dcrnet = nn.DataParallel(dcrnet)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dcrnet.load_state_dict(torch.load('./DCRNet_AF4.pth'))
        dcrnet.to(device)
        dcrnet.eval()
        
        ## load subsampling mask; 
        matImage = scio.loadmat('../TestData/Real_Mask_Acc4_256_by_128.mat')
        mask = matImage['mask']  
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        mask = mask.float()
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        mask = mask.to(device)

        for idx in range(FileNo, FileNo + No_subs):

            input_img_files = SourceDir + '/Input_' + str(FileNo) + '_img.mat'
            input_k_files = SourceDir + '/Input_' + str(FileNo) + '_k.mat'

            print('Loading Subsampled Data')   
            matTest = scio.loadmat(input_img_files, verify_compressed_data_integrity=False)
            image = matTest['inputs_img']
            image = np.array(image)

            image_r = np.real(image)
            image_i = np.imag(image)

            image_r = torch.from_numpy(image_r)
            image_i = torch.from_numpy(image_i)

            image_r = image_r.float()
            image_i = image_i.float()
  
            matTest = scio.loadmat(input_k_files, verify_compressed_data_integrity=False)
            k0 = matTest['inputs_k']
            k0= np.array(k0)

            k0_r = np.real(k0)
            k0_i = np.imag(k0)

            k0_r = torch.from_numpy(k0_r)
            k0_i = torch.from_numpy(k0_i)

            k0_r = k0_r.float()
            k0_i = k0_i.float()

            recons_r = torch.zeros(image_r.size())
            recons_i = torch.zeros(image_i.size())

            ini_recons_r = torch.zeros(image_r.size())
            ini_recons_i = torch.zeros(image_i.size())

            print('reconing ...')
            
            time_start=time.time()
            for i in range(0, image_r.size(3)):   ## 0 --- (no_echos - 1)
                for j in range(0, image_r.size(2)):
                    # conduct reconstruction in a slice-by-slice manner to save memory; 
                    # image-domain inputs
                    INPUT_r = image_r[:,:,j,i]  ##  256 * 128 
                    INPUT_i = image_i[:,:,j,i]

                    INPUT_r = torch.unsqueeze(INPUT_r, 0) ##  1 * 256 * 128
                    INPUT_r = torch.unsqueeze(INPUT_r, 0) ## 1 * 1 * 256 * 128

                    INPUT_i = torch.unsqueeze(INPUT_i, 0)  ## 1 * 256 * 128
                    INPUT_i = torch.unsqueeze(INPUT_i, 0)  ## 1 * 1 * 256 * 128

                    INPUT_r = INPUT_r.to(device)
                    INPUT_i = INPUT_i.to(device)

                    ## k0 for data consisteny
                    INPUT_k_r = k0_r[:,:,j,i]  ##  256 * 128
                    INPUT_k_i = k0_i[:,:,j,i]

                    INPUT_k_r = torch.unsqueeze(INPUT_k_r, 0) ##  1 * 256 * 256
                    INPUT_k_r = torch.unsqueeze(INPUT_k_r, 0) ## 1 * 1 * 256 * 256

                    INPUT_k_i = torch.unsqueeze(INPUT_k_i, 0)  ## 1 * 256 * 256
                    INPUT_k_i = torch.unsqueeze(INPUT_k_i, 0)  ## 1 * 1 * 256 * 256 

                    INPUT_k_r = INPUT_k_r.to(device)
                    INPUT_k_i = INPUT_k_i.to(device)

                    ################ Network Inference ##################

                    ini_r, ini_i, pred_r, pred_i = dcrnet(INPUT_r, INPUT_i, INPUT_k_r, INPUT_k_i, mask)

                    pred_r = torch.squeeze(pred_r, 0)  ## 1 * 256 * 256
                    pred_i = torch.squeeze(pred_i, 0)  ## 1 * 256 * 256

                    pred_r = torch.squeeze(pred_r, 0)  ##  256 * 256
                    pred_i = torch.squeeze(pred_i, 0)  ##  256 * 256
                    
                    ini_r = torch.squeeze(ini_r, 0)  ## 1 * 256 * 256
                    ini_i = torch.squeeze(ini_i, 0)  ## 1 * 256 * 256

                    ini_r = torch.squeeze(ini_r, 0)  ##  256 * 256
                    ini_i = torch.squeeze(ini_i, 0)  ##  256 * 256

                    ini_recons_r[:,:,j,i] = ini_r
                    ini_recons_i[:,:,j,i] = ini_i
                    
                    recons_r[:,:,j,i] = pred_r
                    recons_i[:,:,j,i] = pred_i

            time_end =time.time()
            print('Time Used:' + str(time_end - time_start))

            recons_r = recons_r.to('cpu')
            recons_r = recons_r.numpy()
            recons_i = recons_i.to('cpu')
            recons_i = recons_i.numpy()

            ini_recons_r = ini_recons_r.to('cpu')
            ini_recons_r = ini_recons_r.numpy()
            ini_recons_i = ini_recons_i.to('cpu')
            ini_recons_i = ini_recons_i.numpy()

            print('Saving results')
            path = SourceDir + 'rec_Input_' + str(FileNo) + '_real.mat'
            scio.savemat(path, {'recons_r':recons_r})

            path = SourceDir + 'rec_Input_' + str(FileNo) + '_imag.mat'
            scio.savemat(path, {'recons_i':recons_i})

            path = SourceDir + 'ini_rec_Input_' + str(FileNo) + '_real.mat'
            scio.savemat(path, {'ini_recons_r':ini_recons_r})

            path = SourceDir + 'ini_rec_Input_' + str(FileNo) + '_imag.mat'
            scio.savemat(path, {'ini_recons_i':ini_recons_i})
            print('Reconstruction Ends, Going to MatlabCode Folder for postprocessing')
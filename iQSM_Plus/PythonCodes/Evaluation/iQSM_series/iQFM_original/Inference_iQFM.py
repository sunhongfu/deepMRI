import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
from LoT_Unet_plus  import *
import time
from argparse import ArgumentParser
import os
from collections import OrderedDict

parser = ArgumentParser(description='iQSM')

parser.add_argument('-I', '--InputFile', type=str, default='./',
                    help='Input file saved using SaveInput.m')
parser.add_argument('-O', '--OutputDirectory', type=str, default='./',
                    help='output folder for iQSM and iQFM reconstruction')
parser.add_argument('-C', '--CheckpointsDirectory', type=str, default='./',
                    help='checkpoints folder for iQSM and iQFM pretrained networks')

args = parser.parse_args()

InputPath = args.InputFile
OutputPath = args.OutputDirectory
CheckpointsPath = args.CheckpointsDirectory

if __name__ == '__main__':
    print('iQFM-Net-original')
    with torch.no_grad():
        print('Network Loading')
        # load trained network
        #LGOP =  scio.loadmat("3D_LoTlacian_Operator.mat", verify_compressed_data_integrity=False)
        #conv_op = LGOP['LM']
        """
        conv_op_chi = [[[ 2.4266e-03,  6.4779e-03,  2.0754e-03],
                        [ 7.1839e-03,  2.9119e-02,  6.5230e-03],
                        [ 2.6548e-03,  7.1474e-03,  2.0783e-03]],

                        [[ 6.3654e-03,  3.0482e-02,  7.2806e-03],
                        [ 2.5580e-02, -2.9951e+00,  2.7243e-02],
                        [ 7.1869e-03,  2.9140e-02,  6.3353e-03]],

                        [[ 1.4938e-03,  6.7940e-03,  2.3584e-03],
                        [ 6.4713e-03,  3.0432e-02,  6.1307e-03],
                        [ 1.9087e-03,  7.5989e-03,  1.6020e-03]], ]
        """

        conv_op = [[[1/13,  3/26,  1/13],
                    [3/26,  3/13,  3/26],
                    [1/13,  3/26,  1/13]],

                   [[3/26,  3/13,  3/26],
                    [3/13, -44/13,  3/13],
                    [3/26,  3/13,  3/26]],

                   [[1/13,  3/26,  1/13],
                    [3/26,  3/13,  3/26],
                    [1/13,  3/26,  1/13]], ]
        conv_op = np.array(conv_op)
        conv_op = torch.from_numpy(conv_op)
        conv_op = conv_op.float()
        conv_op = torch.unsqueeze(conv_op, 0)
        conv_op = torch.unsqueeze(conv_op, 0)


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #conv_op = torch.rand(1, 1, 3,3,3)
        LoT_Layer_iQSM = LoTLayer(conv_op)

        LoT_Layer_iQSM = nn.DataParallel(LoT_Layer_iQSM)

        checkpoint_path = os.path.expanduser(
            CheckpointsPath) + '/LoTLayer_lfs_100.pth'
        LoT_Layer_iQSM.load_state_dict(torch.load(
            checkpoint_path, map_location=device))

        Unet_chi = Unet(4, 16, 1)
        Unet_chi = nn.DataParallel(Unet_chi)

        checkpoint_path = os.path.expanduser(
            CheckpointsPath) + '/iQFM_100.pth'
        Unet_chi.load_state_dict(torch.load(
            checkpoint_path, map_location=device))

            
        # state_dict_old = torch.load(
        #     checkpoint_path, map_location=device)

        # state_dict_new = OrderedDict()

        # for k, v in state_dict_old.items():
        #     if 'module.EncodeEmbedding' in k:
        #         k = k.replace('module.EncodeEmbedding', 'EncodeEmbedding') 
        #     if 'module.DecodeEmbedding' in k:
        #         k = k.replace('module.DecodeEmbedding', 'DecodeEmbedding') 
        #     state_dict_new[k] = v

        # Unet_chi.load_state_dict(state_dict_new)

        Unet_chi = Unet_chi.module

        LoT_Layer_iQSM = LoT_Layer_iQSM.module

        iQSM = LoT_Unet(LoT_Layer_iQSM, Unet_chi)
        iQSM = nn.DataParallel(iQSM)

        iQSM.to(device)

        iQSM.eval()

        matImage = scio.loadmat(os.path.abspath(os.path.expanduser(InputPath)))
        image = matImage['phase']
        image = np.array(image)

        image = torch.from_numpy(image)

        image = image.float()

        if len(image.size()) == 3:
            image = torch.unsqueeze(image, 0)
            image = torch.unsqueeze(image, 0)
        else:
            image = image.permute(3, 0, 1, 2)
            image = torch.unsqueeze(image, 1)

        mask = matImage['mask']
        mask = np.array(mask)

        mask = torch.from_numpy(mask)

        mask = mask .float()

        mask = torch.unsqueeze(mask, 0)

        mask = torch.unsqueeze(mask, 0)

        TE = matImage['TE']
        TE = np.array(TE)

        TE = torch.from_numpy(TE)

        TE = TE.float()

        # if len(image.size()) > 3:
        #     TE = torch.unsqueeze(TE, 1)
        #     TE = torch.unsqueeze(TE, 1)
        #     TE = torch.unsqueeze(TE, 1)

        B0 = matImage['B0']
        B0 = np.array(B0)

        B0 = torch.from_numpy(B0)

        B0 = B0.float()

        image = image.to(device)
        mask = mask.to(device)
        TE = TE.to(device)
        B0 = B0.to(device)

        print('reconing ...')

        time_start = time.time()
        
        pred_chi = torch.zeros(image.size()).to(device)

        for ii in range(0, image.size()[0]):
            tmp_img = image[ii, :, :, :].unsqueeze(0)
            tmp_TE = TE[ii]
            pred_chi[ii, :, :, :] = iQSM(tmp_img, mask, tmp_TE, B0)

        time_end = time.time()
        print(time_end - time_start)

        pred_chi = pred_chi * mask

        pred_chi = torch.squeeze(pred_chi, 0)
        pred_chi = torch.squeeze(pred_chi, 0)

        pred_chi = pred_chi.to('cpu')
        pred_chi = pred_chi.numpy()

        print('Saving results')

        path = os.path.expanduser(OutputPath) + '/iQSM.mat'
        scio.savemat(path, {'pred_chi': pred_chi})

        print('end')

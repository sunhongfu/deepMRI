import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
from Lap_Unet import *
import time
from argparse import ArgumentParser
import os

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

    with torch.no_grad():
        print('Network Loading')

        # load trained network
        #LGOP =  scio.loadmat("3D_Laplacian_Operator.mat", verify_compressed_data_integrity=False)
        #conv_op = LGOP['LM']
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

        Lap_Layer = LapLayer(conv_op)

        Unet_chi = Unet(4, 1, 1)
        Unet_chi = nn.DataParallel(Unet_chi)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.path.expanduser(
            CheckpointsPath) + '/iQSM_UnetPart.pth'
        Unet_chi.load_state_dict(torch.load(
            checkpoint_path, map_location=device))

        Unet_lfs = Unet(4, 1, 1)
        Unet_lfs = nn.DataParallel(Unet_lfs)
        checkpoint_path = os.path.expanduser(
            CheckpointsPath) + '/iQFM_UnetPart.pth'
        Unet_lfs.load_state_dict(torch.load(
            checkpoint_path, map_location=device))

        iQSM = Lap_Unet(Lap_Layer, Unet_chi)
        iQFM = Lap_Unet(Lap_Layer, Unet_lfs)

        iQSM.to(device)
        iQFM.to(device)

        iQSM.eval()
        iQFM.eval()

        matImage = scio.loadmat(os.path.expanduser(InputPath))
        image = matImage['phase']
        image = np.array(image)

        image = torch.from_numpy(image)

        image = image.float()

        image = torch.unsqueeze(image, 0)

        image = torch.unsqueeze(image, 0)

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

        pred_lfs = iQFM(image, mask, TE, B0)
        pred_chi = iQSM(image, mask, TE, B0)

        time_end = time.time()
        print(time_end - time_start)

        pred_lfs = pred_lfs * mask
        pred_chi = pred_chi * mask

        pred_lfs = torch.squeeze(pred_lfs, 0)
        pred_lfs = torch.squeeze(pred_lfs, 0)
        pred_chi = torch.squeeze(pred_chi, 0)
        pred_chi = torch.squeeze(pred_chi, 0)

        print(pred_lfs.size())

        pred_lfs = pred_lfs.to('cpu')
        pred_lfs = pred_lfs.numpy()
        pred_chi = pred_chi.to('cpu')
        pred_chi = pred_chi.numpy()

        print('Saving results')
        path = os.path.expanduser(OutputPath) + '/iQFM.mat'
        scio.savemat(path, {'pred_lfs': pred_lfs})

        path = os.path.expanduser(OutputPath) + '/iQSM.mat'
        scio.savemat(path, {'pred_chi': pred_chi})
        print('end')

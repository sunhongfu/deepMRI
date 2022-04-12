import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
from LoT_Unet import *
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
        print('Hemo_Cal_P2Q_16Learnabel_noise_DF')
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


        device = torch.device("cpu")

        #conv_op = torch.rand(1, 1, 3,3,3)
        LoT_Layer_iQSM = LoTLayer(conv_op)

        LoT_Layer_iQSM = nn.DataParallel(LoT_Layer_iQSM)

        checkpoint_path = os.path.expanduser(
            CheckpointsPath) + '/HemoCal_QL_16learnabel_DF_noiselayer/LPLayer_chi_100.pth'
        LoT_Layer_iQSM.load_state_dict(torch.load(
            checkpoint_path, map_location=device))

        print(LoT_Layer_iQSM.module.conv_x)
        kernel_x = LoT_Layer_iQSM.module.conv_x
        kernel_y = LoT_Layer_iQSM.module.conv_y
        kernels = torch.cat([kernel_x, kernel_y], dim = 0)
        kernels = kernels.to('cpu')
        kernels = kernels.numpy()
        path = os.path.expanduser(OutputPath) + '/iQSM_kernels.mat'
        scio.savemat(path, {'kernels': kernels})

        #conv_op2 = torch.rand(1, 1, 3,3,3)
        LoT_Layer_iQFM = LoTLayer(conv_op)

        LoT_Layer_iQFM = nn.DataParallel(LoT_Layer_iQFM)

        checkpoint_path = os.path.expanduser(
            CheckpointsPath) + '/HemoCal_QL_16learnabel_DF_noiselayer/LPLayer_lfs_100.pth'
        LoT_Layer_iQFM.load_state_dict(torch.load(
            checkpoint_path, map_location=device))

        print(LoT_Layer_iQFM.module.conv_x)
        kernel_x = LoT_Layer_iQFM.module.conv_x
        kernel_y = LoT_Layer_iQFM.module.conv_y
        kernels = torch.cat([kernel_x, kernel_y], dim = 0)
        kernels = kernels.to('cpu')
        kernels = kernels.numpy()
        path = os.path.expanduser(OutputPath) + '/iQFM_kernels.mat'
        scio.savemat(path, {'kernels': kernels})

        Unet_chi = Unet(4, 16, 1)
        Unet_chi = nn.DataParallel(Unet_chi)

        checkpoint_path = os.path.expanduser(
            CheckpointsPath) + '/HemoCal_QL_16learnabel_DF_noiselayer/iQSM_100.pth'
        Unet_chi.load_state_dict(torch.load(
            checkpoint_path, map_location=device))

        Unet_lfs = Unet(4, 16, 1)
        Unet_lfs = nn.DataParallel(Unet_lfs)
        checkpoint_path = os.path.expanduser(
            CheckpointsPath) + '/HemoCal_QL_16learnabel_DF_noiselayer/iTFE_100.pth'
        Unet_lfs.load_state_dict(torch.load(
            checkpoint_path, map_location=device))

        Unet_chi = Unet_chi.module
        Unet_lfs = Unet_lfs.module

        LoT_Layer_iQSM = LoT_Layer_iQSM.module
        LoT_Layer_iQFM = LoT_Layer_iQFM.module

        iQSM = LoT_Unet(LoT_Layer_iQSM, Unet_chi)
        iQFM = LoT_Unet(LoT_Layer_iQFM, Unet_lfs)

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

        pred_lfs, tmp_lfs = iQFM(image, mask, TE, B0)
        pred_chi, tmp_chi = iQSM(image, mask, TE, B0)

        time_end = time.time()
        print(time_end - time_start)

        pred_lfs = pred_lfs * mask
        pred_chi = pred_chi * mask

        pred_lfs = torch.squeeze(pred_lfs, 0)
        pred_lfs = torch.squeeze(pred_lfs, 0)
        pred_chi = torch.squeeze(pred_chi, 0)
        pred_chi = torch.squeeze(pred_chi, 0)


        tmp_lfs = tmp_lfs * mask
        tmp_chi = tmp_chi * mask

        tmp_lfs = torch.squeeze(tmp_lfs, 0)
        tmp_lfs = torch.squeeze(tmp_lfs, 0)
        tmp_chi = torch.squeeze(tmp_chi, 0)
        tmp_chi = torch.squeeze(tmp_chi, 0)

        print(pred_lfs.size())

        pred_lfs = pred_lfs.to('cpu')
        pred_lfs = pred_lfs.numpy()
        pred_chi = pred_chi.to('cpu')
        pred_chi = pred_chi.numpy()


        tmp_lfs = tmp_lfs.to('cpu')
        tmp_lfs = tmp_lfs.numpy()
        tmp_chi = tmp_chi.to('cpu')
        tmp_chi = tmp_chi.numpy()

        print('Saving results')
        path = os.path.expanduser(OutputPath) + '/iQFM.mat'
        scio.savemat(path, {'pred_lfs': pred_lfs})

        path = os.path.expanduser(OutputPath) + '/iQSM.mat'
        scio.savemat(path, {'pred_chi': pred_chi})
        """
        path = os.path.expanduser(OutputPath) + '/tmp_lfs.mat'
        scio.savemat(path, {'tmp_lfs': tmp_lfs})

        path = os.path.expanduser(OutputPath) + '/tmp_chi.mat'
        scio.savemat(path, {'tmp_chi': tmp_chi})
        """
        print('end')

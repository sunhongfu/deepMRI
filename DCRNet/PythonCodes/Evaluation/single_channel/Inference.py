import sys
sys.path.append('./Model/')
from DCRNet import *
import os
import time
import torch
from argparse import ArgumentParser
import scipy.io as scio
import numpy as np
import torch.nn as nn
import mat73


parser = ArgumentParser(description='DCRNet')

parser.add_argument('-I', '--InputFile', type=str, default='./',
                    help='Input file saved using Save_Input_Data_For_DCRNet.m')
parser.add_argument('-O', '--OutputDirectory', type=str, default='./',
                    help='output folder for DCRNet reconstruction')
parser.add_argument('-C', '--CheckpointsDirectory', type=str, default='./',
                    help='checkpoints folder for DCRNet pretrained networks')

args = parser.parse_args()

InputPath = args.InputFile
OutputPath = args.OutputDirectory
CheckpointsPath = args.CheckpointsDirectory

if __name__ == '__main__':
    with torch.no_grad():
        print('Network Loading')
        # load pre-trained network
        dcrnet = DCRNet(5)
        dcrnet = nn.DataParallel(dcrnet)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.path.expanduser(CheckpointsPath)
        dcrnet.load_state_dict(torch.load(
            checkpoint_path, map_location=device))
        dcrnet.to(device)
        dcrnet.eval()

        # load subsampling mask;
        matImage = mat73.loadmat(os.path.expanduser(InputPath))
        mask = matImage['mask']
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        mask = mask.float()
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        mask = mask.to(device)

        print('Loading Subsampled Data')
        image = matImage['inputs_img']
        image = np.array(image)

        image_r = np.real(image)
        image_i = np.imag(image)

        image_r = torch.from_numpy(image_r)
        image_i = torch.from_numpy(image_i)

        image_r = image_r.float()
        image_i = image_i.float()

        k0 = matImage['inputs_k']
        k0 = np.array(k0)

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

        time_start = time.time()
        for j in range(0, image_r.size(2)):
            # conduct reconstruction in a slice-by-slice manner to save memory;
            # image-domain inputs
            INPUT_r = image_r[:, :, j]  # 256 * 128
            INPUT_i = image_i[:, :, j]

            INPUT_r = torch.unsqueeze(INPUT_r, 0)  # 1 * 256 * 128
            INPUT_r = torch.unsqueeze(INPUT_r, 0)  # 1 * 1 * 256 * 128

            INPUT_i = torch.unsqueeze(INPUT_i, 0)  # 1 * 256 * 128
            INPUT_i = torch.unsqueeze(INPUT_i, 0)  # 1 * 1 * 256 * 128

            INPUT_r = INPUT_r.to(device)
            INPUT_i = INPUT_i.to(device)

            # k0 for data consisteny
            INPUT_k_r = k0_r[:, :, j]  # 256 * 128
            INPUT_k_i = k0_i[:, :, j]

            INPUT_k_r = torch.unsqueeze(INPUT_k_r, 0)  # 1 * 256 * 256
            INPUT_k_r = torch.unsqueeze(
                INPUT_k_r, 0)  # 1 * 1 * 256 * 256

            INPUT_k_i = torch.unsqueeze(INPUT_k_i, 0)  # 1 * 256 * 256
            INPUT_k_i = torch.unsqueeze(
                INPUT_k_i, 0)  # 1 * 1 * 256 * 256

            INPUT_k_r = INPUT_k_r.to(device)
            INPUT_k_i = INPUT_k_i.to(device)

            ################ Network Inference ##################

            ini_r, ini_i, pred_r, pred_i = dcrnet(
                INPUT_r, INPUT_i, INPUT_k_r, INPUT_k_i, mask)

            pred_r = torch.squeeze(pred_r, 0)  # 1 * 256 * 256
            pred_i = torch.squeeze(pred_i, 0)  # 1 * 256 * 256

            pred_r = torch.squeeze(pred_r, 0)  # 256 * 256
            pred_i = torch.squeeze(pred_i, 0)  # 256 * 256

            ini_r = torch.squeeze(ini_r, 0)  # 1 * 256 * 256
            ini_i = torch.squeeze(ini_i, 0)  # 1 * 256 * 256

            ini_r = torch.squeeze(ini_r, 0)  # 256 * 256
            ini_i = torch.squeeze(ini_i, 0)  # 256 * 256

            ini_recons_r[:, :, j] = ini_r
            ini_recons_i[:, :, j] = ini_i

            recons_r[:, :, j] = pred_r
            recons_i[:, :, j] = pred_i

        time_end = time.time()
        print('Time Used:' + str(time_end - time_start))

        recons_r = recons_r.to('cpu')
        recons_i = recons_i.to('cpu')
        ini_recons_r = ini_recons_r.to('cpu')
        ini_recons_i = ini_recons_i.to('cpu')

        recons_r = recons_r.numpy()
        recons_i = recons_i.numpy()

        ini_recons_r = ini_recons_r.numpy()
        ini_recons_i = ini_recons_i.numpy()

        print('Saving results')
        path = os.path.expanduser(OutputPath) + '/rec_real.mat'
        scio.savemat(path, {'recons_r': recons_r})

        path = os.path.expanduser(OutputPath) + '/rec_imag.mat'
        scio.savemat(path, {'recons_i': recons_i})

        path = os.path.expanduser(OutputPath) + '/rec_real_nodc.mat'
        scio.savemat(path, {'recons_r': ini_recons_r})

        path = os.path.expanduser(OutputPath) + '/rec_imag_nodc.mat'
        scio.savemat(path, {'recons_i': ini_recons_i})
        print('ends')

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import time
import torch.nn as nn
import torch
from utils.handy import (truncate_qsm, torch_from_nib_path, save_tensor_as_nii, generate_dipole,
                         forward_field_calc, calculate_3d_bounding_box, dipole_convolution_f, norm)
import numpy as np
import torch.nn.functional as F
from models.dip_nets import ModelBasedDIPNet
import random
import os


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):

    data_path = args.data_path

    vox = args.vox
    z_prjs = args.z_prjs

    padding_mode = args.padding_mode

    save_path = args.save_path

    is_field = args.is_field

    input_type = args.input_type

    use_GPU = args.use_GPU

    device = torch.device('cuda') if use_GPU else torch.device('cpu')

    lr = args.lr
    gamma = args.gamma
    step = args.step
    epoch_num = args.epoch_num

    grad_loss_order = args.grad_loss_order

    data = torch_from_nib_path(data_path).to(device).squeeze(-1)
    shape = torch.tensor(data.shape[2:]).to(device)

    if args.crop_background:

        # remove the unnessary background voxel to save memory
        bbox = calculate_3d_bounding_box(data)  # [70, 274, 60, 286, 2, 56]
        crop_size = [bbox[4], shape[2] - bbox[5], bbox[2], shape[1] - bbox[3], bbox[0], shape[0] - bbox[1]]
        data = data[:, :, bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]]

    mask = torch.zeros_like(data)
    mask[data != 0] = 1

    dipole = generate_dipole(data.shape, z_prjs, vox).to(device)

    if not is_field:

        data = forward_field_calc(data, z_prjs=z_prjs, vox=vox, need_padding=True, tpe='kspace') * mask

    tkd = truncate_qsm(data, dipole, ts=1/8)[0]

    if input_type == 'pure':
        input = forward_field_calc(tkd, z_prjs=[0, 0, 1], vox=vox, tpe='kspace', need_padding=True) * mask

    elif input_type == 'noise':
        input = torch.rand_like(data)

    elif input_type == 'tkd':
        input = tkd

    else:
        input = data

    vox = vox
    z_prjs = z_prjs

    seed_torch(args.seed)

    model = ModelBasedDIPNet(1, 32, 1, encoder_norm=nn.Identity, norm=nn.InstanceNorm3d, use_skip=False).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step, gamma=gamma)

    _, _, xx, yy, zz = data.shape

    if padding_mode == 'none':
        padding = 2, 2, 2

    elif padding_mode == 'half':
        padding = xx // 2, yy // 2, zz // 2

    elif padding_mode == 'full':
        padding = xx, yy, zz

    else:
        raise ValueError('padding mode not supported')

    device = device
    crit = nn.L1Loss(reduction='sum')

    px, py, pz = padding
    _, _, x, y, z = data.shape

    dipole_padded = generate_dipole(((x + 2 * px), (y + 2 * py), (z + 2 * pz)), z_prjs=z_prjs, vox=vox,
                             device=device).unsqueeze(0).unsqueeze(0)

    model.eval()

    start_time = time.time()
    for epoch in range(epoch_num + 1):

        optim.zero_grad()
        pred_chi = model(input, data, dipole)

        pred_chi = pred_chi * mask
        #
        dc = dipole_convolution_f(F.pad(pred_chi, [pz, pz, py, py, px, px], mode='circular'), dipole_padded)
        dc = dc[:, :, px: -px, py: -py, pz: -pz] * mask

        loss = crit(dc, data) + norm(data, pred_chi, dc, grad_loss_order)

        loss.backward()
        optim.step()
        scheduler.step()

        if epoch % 10 == 0:

            print({'epoch': epoch, 'lr_rate': optim.param_groups[0]['lr'], 'loss': loss.item(),
                    'time': int(time.time() - start_time)})

            with torch.no_grad():

                mean = pred_chi.sum() / (pred_chi != 0).sum()

                pred_chi = (pred_chi - mean) * mask

                if args.crop_background:
                    pred_chi = F.pad(pred_chi, crop_size)

                save_tensor_as_nii(pred_chi, save_path + str(epoch), vox)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="G:\projects\DIPQSM\evaluations\data\\test\hemorrage_anistropic.nii")
    parser.add_argument('--is_field', action='store_true', help='local field input if True, forward calculation needed for QSM input otherwise')
    parser.add_argument('--input_type', choices=['pure', 'phi', 'noise'], default='phi', help='pure: pure-axial local field input, phi: local field input, noise: random noise input')

    parser.add_argument('--vox', type=float, nargs=3, default=[0.6, 0.6, 2])
    parser.add_argument('--z_prjs', type=float, nargs=3, default=[0, 0, 1])

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--step', type=float, default=50)

    parser.add_argument('--padding_mode', type=str, default='half', choices=['none', 'half', 'full'])

    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--seed', type=int, default=3407)

    parser.add_argument('--grad_loss_order', type=int, default=2, choices=[1, 2])

    parser.add_argument('--crop_background', action='store_true')

    parser.add_argument('--use_GPU', type=bool, default=True)

    parser.add_argument('--save_path', type=str, default='')

    args = parser.parse_args()

    main(args)


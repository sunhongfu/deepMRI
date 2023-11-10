from pathlib import Path

import torch
import torch.nn as nn
import time
from utils.helper import affine_transformation, data_fidelity, get_scaling_mat
from networks import Unet, Refinement
from torch.utils.data import DataLoader
from utils.dataset import AbstractTrainingDataset
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-root', type=str, default='')
parser.add_argument('-file_path', type=str, default='file.txt')

# recon model init
parser.add_argument('-depth', type=int, default=4)
parser.add_argument('-recon_base', type=int, default=16)

# refine model init
parser.add_argument('-channel_num', type=int, default=128)
parser.add_argument('-refine_base', type=int, default=64)
parser.add_argument('-block_num', type=int, default=4)
parser.add_argument('-layer_num', type=int, default=3)
parser.add_argument('-use_dense', type=bool, default=True)

# training hyper-parameters
parser.add_argument('-batch_size', type=int, default=2)
parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-gpu_num', type=int, default=2)
parser.add_argument('-padding', type=int, default=16)

# optimizer hyper-parameters
parser.add_argument('-lr1', type=float, default=2e-5)
parser.add_argument('-lr2', type=float, default=2e-5)

parser.add_argument('-drop_step1', type=float, default=25)
parser.add_argument('-drop_step2', type=float, default=25)

parser.add_argument('-alpha1', type=float, default=0.5)
parser.add_argument('-alpha2', type=float, default=0.5)


class PleaseImplementYourDataset(AbstractTrainingDataset):

    @staticmethod
    def path_composition(path_fields: str) -> Path:

        """
        :param path_fields: the path fields corresponding to the index being called in __getitem__.
        :return: the path composited from given fields.
        """

        raise NotImplementedError('Please Implement Your Dataset.')


DEVICE = torch.device('cuda')

def train():

    args = parser.parse_args()

    model_name = 'AFTER-QSM'
    root = args.root
    file_path = args.file_path

    bs = args.batch_size
    epoch = args.epoch
    padding = args.padding

    depth = args.depth
    recon_base = args.recon_base

    channel_num = args.channel_num
    refine_base = args.refine_base
    block_num = args.block_num
    layer_num = args.layer_num
    use_dense = args.use_dense

    lr1 = args.lr1
    lr2 = args.lr2

    drop_step1 = args.drop_step1
    drop_step2 = args.drop_step2

    alpha1 = args.alpha1
    alpha2 = args.alpha2

    dataset = PleaseImplementYourDataset(root=root, file_path=file_path, device=DEVICE)

    model1 = nn.DataParallel(Unet(depth, recon_base)).to(DEVICE)

    model2 = nn.DataParallel(Refinement(num_channels=channel_num, base=refine_base, num_blocks=block_num,
                                        num_layers=layer_num, use_dense=use_dense)).to(DEVICE)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=bs, drop_last=True)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr1, betas=(0.5, 0.999), eps=1e-9, weight_decay=5e-4)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr2, betas=(0.5, 0.999), eps=1e-9, weight_decay=5e-4)

    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=drop_step1, gamma=alpha1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=drop_step2, gamma=alpha2)

    print(len(dataset))

    criterion = nn.MSELoss(reduction='sum')

    start_time = time.time()

    for epoch in range(1, epoch):

        for batch_no, values in enumerate(dataloader):

            chi, mask_hr, dipole, rot_mat, inv_mat, scale = values

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            label = affine_transformation(chi, get_scaling_mat(scale))
            mask_lr = torch.zeros_like(label)

            mask_lr[label != 0] = 1
            phi = data_fidelity(label, dipole) * mask_lr

            aff_phi = affine_transformation(phi, [get_scaling_mat(1/scale), rot_mat])

            aff_pred = model1(aff_phi)

            pred = affine_transformation(aff_pred, [inv_mat, get_scaling_mat(scale)]) * mask_lr

            pred = pred[:, :, padding: -padding, padding: -padding, padding: -padding]
            label = label[:, :, padding: -padding, padding: -padding, padding: -padding]

            pred = model2(pred.detach()) * mask_lr[:, :, padding: -padding, padding: -padding, padding: -padding]

            loss = criterion(aff_pred, affine_transformation(chi, rot_mat)) + criterion(pred, label)

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            if batch_no % 50 == 0:

                print({'epoch': epoch, 'batch_no': batch_no, 'lr_rate': optimizer1.param_groups[0]['lr'],
                       'loss': loss.item(), 'time': int(time.time() - start_time)})

        scheduler1.step()
        scheduler2.step()

        if epoch % 10 == 0:
            torch.save({
                    'model1_state': model1.state_dict(),
                    'model2_state': model2.state_dict()}, model_name + '_' + str(epoch) + '.pkl')
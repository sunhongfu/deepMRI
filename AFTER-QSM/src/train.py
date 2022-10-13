import torch
import torch.nn as nn
import time
from utils import affine_transformation, DataFidelity, get_scaling_mat, get_rotation_mat
from networks import Unet, Refinement
from torch.utils.data import DataLoader
from COSMOS7T.cosmos7T_dataset import COSMOS7TDataset


def get_ids(file_path):
    f = open(file_path, 'r')
    ids = []
    for line in f:
        ids.append(line.strip("\n"))
    return ids


BATCH_SIZE = 4
model_name = 'UnetY_episode13_b5'
DEVICE = torch.device('cuda')

dataset = COSMOS7TDataset(device=DEVICE)
model1 = nn.DataParallel(Unet(4, 16)).to(DEVICE)
model2 = nn.DataParallel(Refinement()).to(DEVICE)
dataloader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
optimizer1 = torch.optim.Adam(model1.parameters(), lr=2e-5, betas=(0.5, 0.999), eps=1e-9, weight_decay=5e-4)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=2e-5, betas=(0.5, 0.999), eps=1e-9, weight_decay=5e-4)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=25, gamma=0.5)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=25, gamma=0.5)
print(len(dataset))
criterion = nn.MSELoss(reduction='sum')
start_time = time.time()

for epoch in range(1, 201):
    for batch_no, values in enumerate(dataloader):

        chi, mask_hr, dipole, rot_mat, inv_mat, scale = values

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        label = affine_transformation(chi, get_scaling_mat(scale))
        mask_lr = torch.zeros_like(label)
        mask_lr[label != 0] = 1
        phi = DataFidelity(label, dipole) * mask_lr

        aff_phi = affine_transformation(phi, [get_scaling_mat(1/scale), rot_mat])
        aff_pred = model1(aff_phi)
        pred = affine_transformation(aff_pred, [inv_mat, get_scaling_mat(scale)]) * mask_lr

        pred = pred[:, :, 16: -16, 16: -16, 16: -16]
        label = label[:, :, 16: -16, 16: -16, 16: -16]

        pred = model2(pred) * mask_lr[:, :, 16: -16, 16: -16, 16: -16]

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
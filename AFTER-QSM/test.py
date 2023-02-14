import numpy as np
import torch
import nibabel as nib
import argparse
from networks import Refinement, Unet
import torch.nn as nn
from utils.helper import get_rotation_mat, get_scaling_mat, affine_transformation, forward_field_calc
import torch.nn.functional as F


parser = argparse.ArgumentParser()

parser.add_argument('-file_path', type=str, default='testing_data/ablation_study.nii.gz')
parser.add_argument('-vox', type=str, default='0.6, 0.6, 0.6')
parser.add_argument('-z_prjs', type=str, default='0, 0, 1')

parser.add_argument('-use_GPU', type=bool, default=True)
parser.add_argument('-GPU_NO', type=str, default='0')

parser.add_argument('-save_path', type=str, default='output')
parser.add_argument('-checkpoint_path', type=str, default='checkpoints/AFTER-QSM.pkl')

parser.add_argument('-segment_num', type=int, default=8, help='a compromise for memeory occpucation.'
                                                              'for more than 8GB but less than 12GB memory, 8 is preferable'
                                                              'for less than 24GB 4 is preferable depending on the matrix size'
                                                              'of the testing data'
                    )

parser.add_argument('-data_type', choices=['field', 'QSM'], help='direct recon if field, forward calculation otherwise.')


def main():

    args = parser.parse_args()

    file_path = args.file_path

    z_prjs = args.z_prjs
    vox = args.vox

    vox = [float(vox.split(', ')[0]), float(vox.split(', ')[1]),float(vox.split(', ')[2])]
    z_prjs = [float(z_prjs.split(',')[0]), float(z_prjs.split(',')[1]), float(z_prjs.split(',')[2])]

    segment_num = args.segment_num

    use_gpu = args.use_GPU
    gpu_no = args.GPU_NO
    device = torch.device('cuda:' + gpu_no) if use_gpu else torch.device('cpu')

    data = torch.from_numpy(nib.load(file_path).get_fdata()[np.newaxis, np.newaxis]).to(device, torch.float)[:, :, 1:-1, :, :]

    data_type = args.data_type

    mask = torch.zeros_like(data)
    mask[data !=0] = 1

    if data_type == 'QSM':

        data = forward_field_calc(data, vox=vox, z_prjs=z_prjs) * mask

    save_path = args.save_path

    state_dict = torch.load(args.checkpoint_path, map_location=device)

    qsm_recon_model = nn.DataParallel(Unet(4, 16)).to(device)
    qsm_refine_model = nn.DataParallel(Refinement(use_dense=True)).to(device)

    qsm_recon_model.load_state_dict(state_dict['recon_model_state'])
    qsm_refine_model.load_state_dict(state_dict['refine_model_state'])

    blur_pred, deblur_pred = run(qsm_recon_model, qsm_refine_model, data, mask, vox, z_prjs, segment_num)

    nib.save(nib.Nifti1Image(blur_pred.cpu().squeeze().numpy(), np.diag([*vox, 1])), str(save_path) + '/blur_pred')
    nib.save(nib.Nifti1Image(deblur_pred.cpu().squeeze().numpy(), np.diag([*vox, 1])), str(save_path) + '/deblur_pred')


def run(qsm_recon_model, qsm_refine_model, data, mask, vox, z_prjs, segment_num):

    zx, zy, zz = vox
    rotation = get_rotation_mat([0, 0, 1], z_prjs)
    scale = get_scaling_mat([0.6 / zz, 0.6 / zy, 0.6 / zx])

    inv_rotation = get_rotation_mat(z_prjs, [0, 0, 1])
    inv_scale = get_scaling_mat([zz / 0.6, zy / 0.6, zx / 0.6])

    step = data.shape[-1] // segment_num

    with torch.no_grad():

        aff_data = affine_transformation(data, [rotation, scale])

        aff_pred = qsm_recon_model(aff_data)

        blur_pred = affine_transformation(aff_pred, [inv_scale, inv_rotation]) * mask

        if segment_num == 1:

            deblur_pred = qsm_refine_model(blur_pred) * mask

        else:

            deblur_pred = torch.zeros_like(blur_pred)

            for i in range(segment_num):

                end = (i + 1) * step if i != segment_num - 1 else blur_pred.shape[-1]

                # Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now
                segment = F.pad(blur_pred[:, :, :, :, i * step: end].squeeze(), [8, 8], mode='reflect')
                segment = segment.unsqueeze(0).unsqueeze(0)

                deblur_pred[:, :, :, :, i * step: end] = qsm_refine_model(segment)[:, :, :, :, 8: -8] * mask[:, :, :, :, i * step: end]

    return blur_pred, deblur_pred


if __name__ == '__main__':
    main()

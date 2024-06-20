import numpy as np
import torch
import nibabel as nib
import argparse
from networks import Refinement, Unet
import torch.nn as nn
from utils.helper import get_rotation_mat, get_scaling_mat, affine_transformation, forward_field_calc, calculate_3d_bounding_box
import torch.nn.functional as F


def main():

    args = parser.parse_args()

    data_path = args.data_path

    z_prjs = args.z_prjs
    vox = args.vox

    segment_num = args.segment_num

    use_gpu = args.use_GPU
    gpu_no = args.GPU_NO
    device = torch.device('cuda:' + gpu_no) if use_gpu else torch.device('cpu')

    data = torch.from_numpy(nib.load(data_path).get_fdata()[np.newaxis, np.newaxis]).to(device, torch.float)
    # data = F.interpolate(data, scale_factor=[1, 1, 1/3])
    # data = data.flip(dims=[-2])
    mask = torch.zeros_like(data)
    mask[data != 0] = 1

    data_type = args.data_type

    if data_type == 'qsm':
        data = forward_field_calc(data, vox=vox, z_prjs=z_prjs) * mask

    bbox = calculate_3d_bounding_box(data)
    shape = torch.tensor(data.shape[2:]).to(device)
    data = data[:, :, bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]]

    # remove the unnessary background voxel to save memory
    crop_size = [bbox[4], shape[2] - bbox[5], bbox[2], shape[1] - bbox[3], bbox[0], shape[0] - bbox[1]]

    # padding1, rescale padding for resolution
    shape = torch.tensor(data.shape[2:]).to(device)
    new_shape = torch.round(shape * torch.tensor(vox).to(device) / 0.6).to(torch.int)
    padding1 = new_shape - shape + 8 * 2  # redundancy
    padding1 = (padding1[2] // 2, padding1[2] // 2, padding1[1] // 2, padding1[1] // 2, padding1[0] // 2, padding1[0] // 2)

    # padding2, cubic padding for head orientation
    padding2 = args.padding
    padding2 = [padding2[-1], padding2[-1], padding2[1], padding2[1],padding2[0],padding2[0]]

    padding = torch.tensor(padding1) + torch.tensor(padding2)

    data = F.pad(data, padding.tolist())

    save_path = args.save_path
    save_name = args.save_name

    state_dict = torch.load(args.checkpoint_path, map_location=device)

    qsm_recon_model = nn.DataParallel(Unet(4, 16)).to(device)
    qsm_refine_model = nn.DataParallel(Refinement(use_dense=True)).to(device)

    qsm_recon_model.load_state_dict(state_dict['recon_model_state'])
    qsm_refine_model.load_state_dict(state_dict['refine_model_state'])

    # consistent buffer device with the parameters
    if not use_gpu:
        qsm_recon_model = qsm_recon_model.module
        qsm_refine_model = qsm_refine_model.module

    with torch.no_grad():

        blur_pred = step1(qsm_recon_model, data, vox, z_prjs)

        # recover the original shape by padding the cropped background back
        blur_pred = F.pad(blur_pred[:, :, padding[4]: -padding[5], padding[2]: -padding[3], padding[0]: -padding[1]]
                          ,crop_size) * mask

        deblur_pred = step2(qsm_refine_model, blur_pred, mask, segment_num=segment_num)

    nib.save(nib.Nifti1Image(blur_pred.cpu().squeeze().numpy(), np.diag([*vox, 1])), str(save_path) + '/blur_' + save_name)
    nib.save(nib.Nifti1Image(deblur_pred.cpu().squeeze().numpy(), np.diag([*vox, 1])), str(save_path) + '/deblur_' + save_name)


def step1(qsm_recon_model, data, vox, z_prjs):

    zx, zy, zz = vox
    rotation = get_rotation_mat([0, 0, 1], z_prjs)
    scale = get_scaling_mat([0.6 / zz, 0.6 / zy, 0.6 / zx])

    inv_rotation = get_rotation_mat(z_prjs, [0, 0, 1])
    inv_scale = get_scaling_mat([zz / 0.6, zy / 0.6, zx / 0.6])

    shape = data.shape[2:]
    aff_data = affine_transformation(data, [rotation, scale])
    bbox = calculate_3d_bounding_box(aff_data)
    aff_pred = qsm_recon_model(aff_data[:, :, bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]])
    aff_pred = F.pad(aff_pred, [bbox[4], shape[2] - bbox[5], bbox[2], shape[1] - bbox[3], bbox[0], shape[0] - bbox[1]])
    return affine_transformation(aff_pred, [inv_scale, inv_rotation])


def step2(qsm_refine_model, blurry_qsm, mask, segment_num):

    step = round(blurry_qsm.shape[-1] / segment_num)

    if segment_num == 1:

        deblur_pred = qsm_refine_model(blurry_qsm) * mask

    else:

        deblur_pred = torch.zeros_like(blurry_qsm)

        for i in range(segment_num):

            end = min((i + 1) * step, blurry_qsm.shape[-1])

            # Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now
            segment = F.pad(blurry_qsm[:, :, :, :, i * step: end].squeeze(), [8, 8], mode='reflect')
            segment = segment.unsqueeze(0).unsqueeze(0)

            deblur_pred[:, :, :, :, i * step: end] = qsm_refine_model(segment)[:, :, :, :, 8: -8] * mask[:, :, :, :,
                                                                                                    i * step: end]

            if end == blurry_qsm.shape[-1]:
                break

    return deblur_pred


if __name__ == '__main__':

    phi = "C:\\Users\\trace\Documents\WeChat Files\s673242975\FileStorage\File\\2024-06\lfs_resharp_tik_0.0001_num_200.nii"
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_path', type=str, default=phi)
    parser.add_argument('-vox', type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument('-z_prjs', type=float, nargs=3, default=[0 ,0, 1])
    parser.add_argument('-padding', type=int, nargs=3, default=[32, 32, 32], help='padding for rotation.'
                                                                                  'This should be enough anisotropic resolution'
                                                                                  'up to [0.6, 0.6, 2], consider to increase it otherwise')

    parser.add_argument('-use_GPU', type=bool, default=True)
    parser.add_argument('-GPU_NO', type=str, default='0')

    parser.add_argument('-save_path', type=str, default='output')
    parser.add_argument('-save_name', type=str, default='AFTER-QSM.nii.gz')
    parser.add_argument('-checkpoint_path', type=str, default='checkpoints/AFTER-QSM.pkl')

    parser.add_argument('-segment_num', type=int, default=3, help='a compromise for memory occupation.'
                                                                  'Consider matrix size 256x256x128 in 1mm3 isotropic,'
                                                                  'for more than 8GB but less than 12GB, 8 is preferable'
                                                                  'for less than 24GB, 4 is preferable'
                        )

    parser.add_argument('-data_type', choices=['field', 'qsm'],
                        help='direct recon if field, forward calculation otherwise.')

    main()
import collections.abc
import math

import torch
import torch.fft as fft
import torch.nn.functional as F
import numpy as np
from math import floor
import torch.nn as nn
import nibabel as nib


def norm(phi, pred, dc, order=2):

    laplacian_order_2 = nn.Parameter(torch.tensor([[[2, 3, 2],
                                                    [3, 6, 3],
                                                    [2, 3, 2]],
                                                   [[3, 6, 3],
                                                    [6, -88, 6],
                                                    [3, 6, 3]],
                                                   [[2, 3, 2],
                                                    [3, 6, 3],
                                                    [2, 3, 2]]
                                                   ], dtype=torch.float).unsqueeze(0).unsqueeze(0),
                                     requires_grad=False) / 26

    sobel_order_1 = nn.Parameter(torch.tensor([[
        [1, 0, 1],
        [2, 0, -2],
        [1, 0, -1]],
        torch.zeros([3, 3]),
        [[-1, -2, -1],
         [-2, -4, -2],
         [-1, -2, -1]]]))

    kernel = sobel_order_1 if order == 1 else laplacian_order_2

    lap_dc = F.conv3d(dc, weight=kernel.to(pred.device), stride=1, padding=1)
    lap_phi = F.conv3d(phi, weight=kernel.to(pred.device), stride=1, padding=1)

    return F.l1_loss(lap_dc, lap_phi) * 0.5


def calculate_3d_bounding_box(tensor_3d):
    # Get the non-zero indices using torch.nonzero
    non_zero_indices = torch.nonzero(tensor_3d.squeeze(), as_tuple=False)

    if non_zero_indices.size(0) == 0:
        # Empty tensor, no bounding box to compute
        return None

    # Get the minimum and maximum coordinates (x, y, z) of the non-zero elements
    min_x, _ = torch.min(non_zero_indices[:, 0], dim=0)
    max_x, _ = torch.max(non_zero_indices[:, 0], dim=0)
    min_y, _ = torch.min(non_zero_indices[:, 1], dim=0)
    max_y, _ = torch.max(non_zero_indices[:, 1], dim=0)
    min_z, _ = torch.min(non_zero_indices[:, 2], dim=0)
    max_z, _ = torch.max(non_zero_indices[:, 2], dim=0)

    # The bounding box is defined by the minimum and maximum coordinates
    return [min_x.item() // 2 * 2, max_x.item() // 2 * 2, min_y.item() // 2 * 2, max_y.item() // 2 * 2, min_z.item() // 2 * 2, max_z.item() // 2 * 2]


def dipole_convolution_f(sus, dipole):
    b_sus, *_ = sus.size()
    b_dpl, *_ = dipole.size()
    if b_dpl == 1:
        dipole = dipole.repeat([b_sus, 1, 1, 1, 1])
    res = fft.ifftn(dipole * fft.fftn(sus))
    # res = dipole * fft.fftn(sus)
    return res.real


def truncate_qsm(phi, dipole, ts=0.2):
    sgn_dipole = torch.sgn(dipole)
    value_dipole = torch.abs(dipole)

    mask = value_dipole > ts
    value_dipole[~mask] = ts
    new_dipole = sgn_dipole * value_dipole
    new_dipole[new_dipole == 0] = ts
    phi_k = fft.fftn(phi)
    tkd_chi = phi_k / new_dipole
    tkd_chi = torch.real(fft.ifftn(tkd_chi))
    mask_phi_k = phi_k * mask
    return tkd_chi, fft.ifftn(mask_phi_k)


def generate_dipole(shape, z_prjs=(0, 0, 1), vox=(1, 1, 1), shift=True,
                    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    vox = np.array(vox) if isinstance(vox, collections.abc.Collection) else vox
    if len(shape) == 5:
        _, _, Nx, Ny, Nz = shape
        FOVx, FOVy, FOVz = vox * shape[2:]
    else:
        Nx, Ny, Nz = shape
        FOVx, FOVy, FOVz = vox * shape
    x = torch.linspace(-Nx / 2, Nx / 2 - 1, steps=Nx)
    y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
    z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)
    kx, ky, kz = torch.meshgrid(x / FOVx, y / FOVy, z / FOVz)
    D = 1 / 3 - (kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]) ** 2 / (kx ** 2 + ky ** 2 + kz ** 2)
    D[floor(Nx / 2), floor(Ny / 2), floor(Nz / 2)] = 0
    D = D if len(shape) == 3 else D.unsqueeze(0).unsqueeze(0)
    return torch.fft.fftshift(D).to(device) if shift else D.to(device)


def generate_dipole_img(shape, z_prjs=(0, 0, 1), vox=(1, 1, 1),
                        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    # all dimensions should be even

    vox = np.array(vox) if isinstance(vox, collections.abc.Collection) else vox
    if len(shape) == 5:
        _, _, Nx, Ny, Nz = shape
    else:
        Nx, Ny, Nz = shape

    x = torch.linspace(-Nx / 2, Nx / 2 - 1, steps=Nx)
    y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
    z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)

    x, y, z = torch.meshgrid(x, y, z)

    x = x * vox[0]
    y = y * vox[1]
    z = z * vox[2]

    d = np.prod(vox) * (3 * (x * z_prjs[0] + y * z_prjs[1] + z * z_prjs[2]) ** 2 - x ** 2 - y ** 2 - z ** 2) \
        / (4 * math.pi * (x ** 2 + y ** 2 + z ** 2) ** 2.5)

    d[torch.isnan(d)] = 0
    d = d if len(shape) == 3 else d.unsqueeze(0).unsqueeze(0)

    return torch.real(fft.fftn(fft.fftshift(d))).to(device)


def forward_field_calc(sus, z_prjs=(0, 0, 1), vox=(1, 1, 1), need_padding=False, tpe='img'):
    device = sus.device
    vox = torch.tensor(vox)
    _, _, Nx, Ny, Nz = sus.size()
    if need_padding:
        sus = F.pad(sus, [Nz // 2, Nz // 2, Ny // 2, Ny // 2, Nx // 2, Nx // 2])

    method = generate_dipole if tpe == 'kspace' else generate_dipole_img
    D = method(sus.shape, z_prjs, vox, device)
    ###
    field = torch.real(fft.ifftn(D * fft.fftn(sus)))
    return field[:, :, Nx // 2: - Nx // 2, Ny // 2: - Ny // 2, Nz // 2: - Nz // 2] if need_padding else field


def torch_from_nib_path(path, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    return torch.from_numpy(nib.load(path).get_fdata()[np.newaxis, np.newaxis]).float().to(device)


def save_tensor_as_nii(tensor, name, vox=(1, 1, 1)):
    return nib.save(nib.Nifti1Image(tensor.squeeze().detach().cpu().numpy(), np.diag((*vox, 1))), name + '.nii')


def save_array_as_nii(arr, name):
    return nib.save(nib.Nifti1Image(arr, np.eye(4)), name + '.nii')


def torch_from_numpy_path(path, device=torch.device('cuda')):
    return torch.from_numpy(np.load(path)[np.newaxis, np.newaxis]).float().to(device)

import collections.abc
import torch
import torch.nn.functional as F
import numpy as np
from math import floor, ceil
import nibabel as nib
from typing import List, Union
import torch.fft as FFT


def reshape_for_Unet_compatibility(layer=4):
    def outer(eval_func):

        def wrapper(*args, **kwargs):

            phi = args[1]
            b, _, w, h, d = phi.shape

            padding = [floor((ceil(d / 2 ** layer) * 2 ** layer - d) / 2),
                       ceil((ceil(d / 2 ** layer) * 2 ** layer - d) / 2),
                       floor((ceil(h / 2 ** layer) * 2 ** layer - h) / 2),
                       ceil((ceil(h / 2 ** layer) * 2 ** layer - h) / 2),
                       floor((ceil(w / 2 ** layer) * 2 ** layer - w) / 2),
                       ceil((ceil(w / 2 ** layer) * 2 ** layer - w) / 2)]

            pred = eval_func(*(args[0], F.pad(phi, padding)), **kwargs)

            if len(pred) == 1:
                b, _, w, h, d = pred.shape
                return pred[:, :, padding[-2]: w - padding[-1], padding[-4]: h - padding[-3],
                       padding[-6]: d - padding[-5]]

            else:
                res = []
                for ele in pred:
                    b, _, w, h, d = ele.shape
                    res.append(ele[:, :, padding[-2]: w - padding[-1], padding[-4]: h - padding[-3],
                               padding[-6]: d - padding[-5]])

                return res

        return wrapper

    return outer


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
    return [min_x.item(), max_x.item(), min_y.item(), max_y.item(), min_z.item(), max_z.item()]


def cubic_padding(func):
    """
    cubic padding for affine transformation
    :param func: The affine transformation function to be wrapped.
    :return: post-processed affine transformation function
    """
    def wrapper(*args, **kwargs):
        img = args[0]
        b, _, w, h, d = img.shape
        max_dim = max(w, h, d)

        # cubic padding to avoid unreasonably stretching
        padding = [floor((max_dim - d) / 2), ceil((max_dim - d) / 2), floor((max_dim - h) / 2),
                   ceil((max_dim - h) / 2), floor((max_dim - w) / 2), ceil((max_dim - w) / 2)]
        img = F.pad(img, padding)

        return func(*(img, *args[1:]), **kwargs)[:, :, padding[-2]: w + padding[-2], padding[-4]: h + padding[-4],
               padding[-6]: d + padding[-6]]

    return wrapper


def skew(vector):
    """
    skew-symmetric operator for rotation matrix generation
    """

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def get_rotation_mat(ori1, ori2):
    """
    generating pythonic style rotation matrix
    :param ori1: your current orientation
    :param ori2: orientation to be rotated
    :return: pythonic rotation matrix.
    """
    ori1 = np.array(ori1) if isinstance(ori1, collections.abc.Collection) else ori1
    ori2 = np.array(ori2) if isinstance(ori2, collections.abc.Collection) else ori2
    v = np.cross(ori1, ori2)
    c = np.dot(ori1, ori2)
    mat = np.identity(3) + skew(v) + np.matmul(skew(v), skew(v)) / (1 + c)
    # return torch.from_numpy(mat).float()
    return torch.from_numpy(np.flip(mat).copy()).float().unsqueeze(0).unsqueeze(0)


def get_scaling_mat(scale):
    return torch.diag_embed(torch.tensor(scale)).float()


@cubic_padding
def affine_transformation(img, affine: Union[List[torch.Tensor], torch.Tensor], pure_translation=None, mode='bilinear'):
    """
    :param img: img to be transformed
    :param affine: affine matrices
    :param pure_translation: pure shift
    :return: affine transformed img
    """
    device = img.device
    b = img.shape[0]

    # add no pure translation
    if pure_translation is None:
        pure_translation = torch.zeros(b, 3, 1).to(device)

    # calculate affine matrices
    affine_mat = torch.eye(3, 3)
    if not isinstance(affine, list):

        affine_mat = affine.squeeze()

    elif len(affine) == 1:

        affine_mat = affine[0]

    else:
        for index in range(len(affine) - 1):
            affine_mat = torch.matmul(affine[index].squeeze().to(device), affine[index + 1].squeeze().to(device))
            affine[index + 1] = affine_mat

    # apply one-step affine transform
    affine_mat = affine_mat.repeat([b, 1, 1]) if len(affine_mat.shape) == 2 else affine_mat
    affine_matrix = torch.cat([affine_mat.to(device), pure_translation], dim=2)
    grid = F.affine_grid(affine_matrix, img.shape, align_corners=False)

    rot_img = F.grid_sample(input=img, grid=grid, mode=mode)
    _, _, w, h, d = rot_img.shape
    return rot_img


def generate_dipole(shape, z_prj=(0, 0, 1), vox=(1, 1, 1), shift=True,
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
    D = 1 / 3 - (kx * z_prj[0] + ky * z_prj[1] + kz * z_prj[2]) ** 2 / (kx ** 2 + ky ** 2 + kz ** 2)
    D[floor(Nx / 2), floor(Ny / 2), floor(Nz / 2)] = 0
    D = D if len(shape) == 3 else D.unsqueeze(0).unsqueeze(0)
    return torch.fft.fftshift(D).to(device) if shift else D.to(device)


def data_fidelity(chi, dipole):

    x_k = FFT.fftn(chi, dim=(-3, -2, -1))

    x_k = x_k * dipole

    x_img = FFT.ifftn(x_k, dim=(-3, -2, -1))

    x_img = torch.real(x_img)

    return x_img


def forward_field_calc(sus, z_prjs=(0, 0, 1), vox=(1, 1, 1), need_padding=False):

    device = sus.device
    vox = torch.tensor(vox)
    _, _, Nx, Ny, Nz = sus.size()
    if need_padding:
        sus = F.pad(sus, [Nz//2, Nz//2, Ny//2, Ny//2, Nx//2, Nx//2])
    sz = torch.tensor(sus.size()).to(torch.long)[2:]
    Nx = sz[0].item()
    Ny = sz[1].item()
    Nz = sz[2].item()
    FOVx, FOVy, FOVz = vox * sz
    x = torch.linspace(-Nx / 2, Nx / 2 - 1, Nx)
    y = torch.linspace(-Ny / 2, Ny / 2 - 1, Ny)
    z = torch.linspace(-Nz / 2, Nz / 2 - 1, Nz)
    [kx, ky, kz] = torch.meshgrid(x / FOVx, y / FOVy, z / FOVz)
    D = 1 / 3 - torch.pow((kx * z_prjs[0] + ky * z_prjs[1] + kz * z_prjs[2]), 2) / (kx ** 2 + ky ** 2 + kz ** 2)
    D[floor(Nx / 2), floor(Ny / 2), floor(Nz / 2)] = 0
    D = FFT.fftshift(D).to(device)
    ###
    D = D.unsqueeze(0).unsqueeze(0)
    ###
    field = torch.real(FFT.ifftn(D * FFT.fftn(sus)))

    return field[:, :, Nx // 4: - Nx // 4, Ny // 4: - Ny // 4, Nz // 4: - Nz // 4] if need_padding else field


def torch_from_nib_path(path, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    return torch.from_numpy(nib.load(path).get_fdata()[np.newaxis, np.newaxis]).float().to(device)


def save_tensor_as_nii(tensor, name, vox=(1, 1, 1)):
    return nib.save(nib.Nifti1Image(tensor.squeeze().detach().cpu().numpy(), np.diag((*vox, 1))), name + '.nii')


def save_array_as_nii(arr, name):
    return nib.save(nib.Nifti1Image(arr, np.eye(4)), name + '.nii')


def torch_from_numpy_path(path, device=torch.device('cuda')):
    return torch.from_numpy(np.load(path)[np.newaxis, np.newaxis]).float().to(device)


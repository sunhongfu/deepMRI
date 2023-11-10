import torch
from torch.utils.data import dataset
from abc import abstractmethod
import nibabel as nib
import torch.nn.functional as F
import numpy as np
import random
from helper import generate_dipole, get_rotation_mat
from pathlib import Path


class AbstractTrainingDataset(dataset):

    def __init__(self, root, file_path, device, z_prjs_path='zprjs.txt', vox_path='vox.txt', need_padding=True):

        self.root = root
        self.device = device
        self.entries = []
        self.need_padding = need_padding

        with open(file_path, 'r') as f:
            self.fields = []
            for field in f:
                self.fields.append(field.strip('\n'))

        with open(z_prjs_path, 'r') as f:
            self.z_prjs = []
            for line in f:
                self.z_prjs.append(line.strip('\n'))

        with open(vox_path, 'r') as f:
            self.vox = []
            for line in f:
                self.voxes.append(line.strip('\n'))

        for field in self.fields:
            self.entries.append(self.root / AbstractTrainingDataset.path_composition(field))

    @staticmethod
    @abstractmethod
    def path_composition(path_fields: str) -> Path:

        """
        :param path_fields: the path fields corresponding to the index being called in __getitem__.
        :return: the path composited from given fields.
        """

        raise NotImplementedError

    def __getitem__(self, index):

        chi_path = self.entries[index]
        chi = torch.from_numpy(nib.load(str(chi_path)).get_fdata()[np.newaxis]).to(self.device, torch.float)
        chi = F.pad(chi, [16, 16, 16, 16, 16, 16]) if self.need_padding else chi

        vox = self.voxes[index]
        z_prj = self.z_prjs[index]

        vox = [float(vox.strip('[').strip(']').split(', ')[0]), float(vox.strip('[').strip(']').split(', ')[1]),
               float(vox.strip('[').strip(']').split(', ')[2])]
        random.shuffle(vox)

        z_prj = [float(z_prj.split(',')[0]), float(z_prj.split(',')[1]), float(z_prj.split(',')[2])]

        dipole = generate_dipole(chi.shape[1:], vox=vox, z_prj=z_prj).unsqueeze(0)

        mask = torch.zeros_like(chi)
        mask[chi != 0] = 1

        scale = torch.tensor([vox[2]/0.6, vox[1]/0.6, vox[0]/0.6])

        rot_mat = get_rotation_mat([0, 0, 1], z_prj)
        inv_mat = rot_mat.transpose(-1, -2)

        return chi, mask, dipole, rot_mat, inv_mat, scale


class TestDataset(dataset):

    def __init__(self, data_path, vox, z_prjs):

        nib_file = nib.load(str(data_path)).get_fdata()
        vox

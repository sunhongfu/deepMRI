import numpy as np
from handy import get_rotation_mat, forward_field_calc, generate_dipole
from utils.eval import AbstractConfig
import torch
import nibabel as nib
from abc import abstractmethod
from pathlib import Path
from handy import *


class SimpleTest(AbstractConfig):

    def data_pre_processing(self, entry):
        pass

    @staticmethod
    def set_entry_path(entries, file_names_to_save):
        pass

    @abstractmethod
    def data_post_processing(self, entry):
        pass

    @staticmethod
    def get_file_name(path):
        return Path(path).parts[-1].split('.')[0]


class SimpleInvivoTest(SimpleTest):

    def __init__(self, path, z_prj, name, path_to_save='', vox=(1, 1, 1)):
        super(SimpleTest, self).__init__('a simple test', file_save_path=path_to_save)
        self.resolution = vox
        self.data_fields = ['phi', 'z_prj']
        if isinstance(path, list):
            for index, (p, z) in enumerate(path, z_prj):
                self.entries.append({
                    'phi': p,
                    'z_prj': z})
                self.file_names_to_save.append(name + '_' + self.get_file_name(path) + str(index))
        else:
            self.entries.append({
                'phi': path,
                'z_prj': z_prj})
            self.file_names_to_save.append(name + '_' + self.get_file_name(path))

    def data_post_processing(self, entry):
        z_prj = torch.cat(entry.pop('z_prj'), dim=0)
        mat = get_rotation_mat(np.array([0, 0, 1]), z_prj).to(self.device)
        inv_mat = mat.transpose(-1, -2)
        mask = torch.zeros_like(entry['phi'])
        mask[entry['phi'] != 0] = 1
        entry['mat'] = mat
        entry['inv_mat'] = inv_mat
        entry['mask'] = mask
        entry['z_prj'] = z_prj


class SimpleSimulateTest(SimpleTest):

    def __init__(self, path, z_prj, name, path_to_save='', vox=(1, 1, 1)):
        super(SimpleTest, self).__init__(name, file_save_path=path_to_save)
        self.data_fields = ['chi', 'z_prj']
        if isinstance(path, list):
            for index, (p, z) in enumerate(path, z_prj):
                self.entries.append({
                    'chi': p,
                    'z_prj': np.array(z)})
                self.file_names_to_save.append(name + '_' + self.get_file_name(path) + str(index))
        else:
            self.entries.append({
                'chi': path,
                'z_prj': np.array(z_prj)})
            self.file_names_to_save.append(name + '_' + self.get_file_name(path))
        self.resolution = torch.tensor(vox)

    def data_post_processing(self, entry):
        z_prj = entry.pop('z_prj').squeeze()
        chi = entry.pop('chi')
        mat = get_rotation_mat(np.array([0, 0, 1]), z_prj).to(self.device)
        inv_mat = mat.transpose(-1, -2)
        field = forward_field_calc(chi, z_prj, self.resolution, True)
        mask = torch.zeros_like(chi)
        mask[chi != 0] = 1
        entry['phi'] = field
        entry['mat'] = mat
        entry['inv_mat'] = inv_mat
        entry['mask'] = mask
        entry['z_prj'] = z_prj

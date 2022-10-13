import numpy as np
import nibabel as nib
import scipy.io as scio
import torch.nn as nn
from abc import abstractmethod, ABC
import os
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calc_ssim(label, pred):
    pred = pred.squeeze().cpu().numpy()
    return ssim(label, pred)


def calc_psnr(label, pred):
    pred = pred.squeeze().cpu().numpy()
    return psnr(label, pred)


class AbstractRunner:

    def __init__(self, *config, state_dict_path, root_path, requires_stat=False, label_name=None, device=torch.device('cuda')):
        self.root_path = Path(root_path)
        self.configs = config
        self.require_stats = requires_stat
        self.label_name = label_name
        self.model = []
        self.device = device
        self.state_dict_path = state_dict_path
        models = self.set_model()
        if type(models[0]) is not tuple:
            self.model.append((models[0], models[1]))
        else:
            for m in models:
                self.model.append(m)
        self.model_prepare()

    def start(self):
        for config in self.configs:
            config.device = self.device
            self.prepare_entry_and_run(config)

    def prepare_entry_and_run(self, config):
        dataloader = DataLoader(config, shuffle=False, batch_size=1)
        print('-----------------dataset ' + config.name + ' starts----------------------------')
        for index, entry in enumerate(dataloader):
            data = config.data_processing(entry)
            config.data_post_processing(data)
            with torch.no_grad():
                pred = self.run(data, self.model, index, config.name)
                if not os.path.exists(self.root_path / config.save_path):
                    path = self.root_path / config.save_path
                    path.mkdir(parents=True)
                if len(pred) == 1:
                    config.save(pred, self.root_path / config.save_path / config.file_names_to_save[index], config.resolution)
                else:
                    for sub_index, tensor in enumerate(pred):
                        config.save(tensor, self.root_path / config.save_path / (config.file_names_to_save[index]
                                    + '#' + str(sub_index) + '.nii'), config.resolution)
            if self.require_stats:
                label = entry[-1]
                ssim_value = calc_ssim(label, pred)
                psnr_value = calc_psnr(label, pred)
                print(ssim_value)
                print(psnr_value)

    def model_prepare(self):
        models = []
        for model_tuple in self.model:
            model, state = model_tuple
            try:
                model.load_state_dict(state, True)
                model = model.cpu() if str(self.device) == 'cpu' else model.cuda()

            except Exception:
                model = nn.DataParallel(model)
                model.load_state_dict(state, True)
                model = model.module.cpu() if str(self.device) == 'cpu' else model.module.cuda()
            model.eval()
            models.append(model)
        self.model = models

    @staticmethod
    def run(entries, model, index, name):
        raise NotImplementedError

    def set_model(self):
        raise NotImplementedError


class AbstractConfig(ABC, TensorDataset):

    def __init__(self, name, file_save_path, dtype=torch.float, fields=None):
        super(AbstractConfig, self).__init__()
        self.entries = []
        self.dtype = dtype
        self.file_names_to_save = []
        self.save_path = file_save_path
        self.name = name
        self.device = torch.device('cuda')
        self.set_entry_path(self.entries, self.file_names_to_save)
        self.data_fields = ['phi', 'mask', 'mat', 'inv_mat']
        self.resolution = [1, 1, 1]
        if fields is not None:
            self.data_fields.extend(fields)

    @staticmethod
    @abstractmethod
    def set_entry_path(entries, file_names_to_save):
        raise NotImplementedError

    @abstractmethod
    def data_pre_processing(self, entry):
        """
        manipulate data at numpy level.
        :param entry: dictionary where data store for one specific round.
        :return:
        """
        raise NotImplementedError

    def data_processing(self, entry):
        for key, value in entry.items():
                if key == 'z_prj':
                    continue
                if type(value) == torch.Tensor:
                    entry[key] = value.unsqueeze(1).to(self.device, self.dtype)

        return entry

    def data_post_processing(self, entry):
        pass

    def save(self, tensor, file_name, resolution=(1, 1, 1)):
        nifti = nib.Nifti1Image(tensor.squeeze().cpu().numpy(), np.diag((*resolution, 1)))
        current_path = os.getcwd()
        save_path = file_name.parent
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        nib.save(nifti, save_path / file_name.parts[-1])
        print(file_name.parts[-1] + ' saved')
        os.chdir(current_path)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        data = {}
        entry = self.entries[index]
        keys_to_remove = []
        for k, v in entry.items():
            short_v = str(v).strip('.gz')
            try:
                if short_v.endswith('nii'):
                    try:
                        data[k] = nib.load(v).get_fdata()
                    except FileNotFoundError:
                        data[k] = nib.load(short_v).get_fdata()
                elif short_v.endswith('mat'):
                    data[k] = scio.loadmat(v)
                elif short_v.endswith('npy'):
                    data[k] = np.load(v)
                else:
                    data[k] = v.strip('_X')
            except AttributeError:
                data[k] = v
        self.data_pre_processing(data)
        for k, v in data.items():
            if k not in self.data_fields:
                keys_to_remove.append(k)
        for key in keys_to_remove:
            data.pop(key)
        return data

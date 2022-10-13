from eval import AbstractRunner
from networks import Unet, Refinement
import torch
from simple_tests import SimpleSimulateTest, SimpleInvivoTest
import argparse
from utils import get_scaling_mat, affine_transformation, reshape_for_Unet_compatibility
import torch.nn.functional as F


class Runner(AbstractRunner):

    def __init__(self, *config, state_dict_path, root_path='', device=torch.device('cuda')):
        super(Runner, self).__init__(*config, state_dict_path=state_dict_path, root_path=root_path, requires_stat=False,
                                     label_name='label', device=device)

    @staticmethod
    @reshape_for_Unet_compatibility(layer=6)
    def test(entries, model, name):
        net1, net2 = model
        phi, mat, inv_mat, mask, z_prj = entries.values()

        new_mask = affine_transformation(mask, [get_scaling_mat([0.6 / 0.7, 0.6 / 0.64, 0.6 / 0.64]), mat])
        phi = affine_transformation(phi * mask, [get_scaling_mat([0.6 / 0.7, 0.6 / 0.64, 0.6 / 0.64]), mat])
        pred = net1(phi) * new_mask
        pred1 = affine_transformation(pred, [inv_mat, get_scaling_mat([0.7 / 0.6, 0.64 / 0.6, 0.64 / 0.6])]) * mask
        pred2 = net2(pred1[:, :, 32: -32, 32: -32, 32: -32]) * mask[:, :, 32: -32, 32: -32, 32: -32]
        return pred1[:, :, 48: -48, 48: -48, 48: -48], pred2

    @staticmethod
    def data_preprocessing(entries):
        phi = entries['phi']
        mask = entries['mask']

        phi = F.pad(phi, [48, 48, 48, 48, 48, 48])
        mask = F.pad(mask, [48, 48, 48, 48, 48, 48])

        entries['phi'] = phi
        entries['mask'] = mask

    def set_model(self):
        net1 = Unet(4, 16)
        net2 = Refinement()
        state_dict1 = torch.load(self.state_dict_path, map_location=self.device)['model1_state']
        state_dict2 = torch.load(self.state_dict_path, map_location=self.device)['model2_state']

        return (net1, state_dict1), (net2, state_dict2)

    @staticmethod
    def run(entries, model, index, name):
        Runner.data_preprocessing(entries)
        return Runner.test(entries, model, name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    state_dict_path = ''
    test_data_path = ''
    save_path = ''
    device = torch.device('cuda')

    simple_test = SimpleSimulateTest if args.type == 'simulation' else SimpleInvivoTest

    runner = Runner(
        SimpleSimulateTest(test_data_path,
                           name='test', path_to_save='test',
                           vox=[0.6, 0.6, 0.6], z_prj=[0, 0, 1]),
        state_dict_path=state_dict_path, root_path='evaluation/results/', device=device
    )



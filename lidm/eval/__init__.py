"""
@Author: Haoxi Ran
@Date: 01/03/2024
@Citation: Towards Realistic Scene Generation with LiDAR Diffusion Models

"""

import os

import torch
import yaml

from lidm.utils.misc_utils import dict2namespace
from ..modules.rangenet.model import Model as rangenet

try:
    from ..modules.spvcnn.model import Model as spvcnn
    from ..modules.minkowskinet.model import Model as minkowskinet
except:
    print('To install torchsparse 1.4.0, please refer to https://github.com/mit-han-lab/torchsparse/tree/74099d10a51c71c14318bce63d6421f698b24f24')

# user settings
DEFAULT_ROOT = './pretrained_weights'
MODAL2BATCHSIZE = {'range': 100, 'voxel': 50, 'point_voxel': 25}
OUTPUT_TEMPLATE = 50 * '-' + '\n|' + 16 * ' ' + '{}:{:.4E}' + 17 * ' ' + '|\n' + 50 * '-'

# eval settings (do not modify)
VOXEL_SIZE = 0.05
NUM_SECTORS = 16
AGG_TYPE = 'depth'
TYPE2DATASET = {'32': 'nuscenes', '64': 'kitti'}
DATA_CONFIG = {'64': {'x': [-50, 50], 'y': [-50, 50], 'z': [-3, 1]},
               '32': {'x': [-30, 30], 'y': [-30, 30], 'z': [-3, 6]}}
MODALITY2MODEL = {'range': 'rangenet', 'voxel': 'minkowskinet', 'point_voxel': 'spvcnn'}
DATASET_CONFIG = {'kitti': {'size': [64, 1024], 'fov': [3, -25], 'depth_range': [1.0, 56.0], 'depth_scale': 6},
                  'nuscenes': {'size': [32, 1024], 'fov': [10, -30], 'depth_range': [1.0, 45.0]}}


def build_model(dataset_name, model_name, device='cpu'):
    # config
    model_folder = os.path.join(DEFAULT_ROOT, dataset_name, model_name)

    if not os.path.isdir(model_folder):
        raise Exception('Not Available Pretrained Weights!')

    config = yaml.safe_load(open(os.path.join(model_folder, 'config.yaml'), 'r'))
    if model_name != 'rangenet':
        config = dict2namespace(config)

    # build model
    model = eval(model_name)(config)

    # load checkpoint
    if model_name == 'rangenet':
        model.load_pretrained_weights(model_folder)
    else:
        ckpt = torch.load(os.path.join(model_folder, 'model.ckpt'), map_location="cpu")
        model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()

    return model

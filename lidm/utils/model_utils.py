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

DEFAULT_ROOT = './pretrained_weights'


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

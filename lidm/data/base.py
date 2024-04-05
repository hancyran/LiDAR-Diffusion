import pdb
from abc import abstractmethod
from functools import partial

import PIL
import numpy as np
from PIL import Image

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, IterableDataset

from ..utils.aug_utils import get_lidar_transform, get_camera_transform, get_anno_transform


class DatasetBase(Dataset):
    def __init__(self, data_root, split, dataset_config, aug_config, return_pcd=False, condition_key=None,
                 scale_factors=None, degradation=None, **kwargs):
        self.data_root = data_root
        self.split = split
        self.data = []
        self.aug_config = aug_config

        self.img_size = dataset_config.size
        self.fov = dataset_config.fov
        self.depth_range = dataset_config.depth_range
        self.filtered_map_cats = dataset_config.filtered_map_cats
        self.depth_scale = dataset_config.depth_scale
        self.log_scale = dataset_config.log_scale

        if self.log_scale:
            self.depth_thresh = (np.log2(1./255. + 1) / self.depth_scale) * 2. - 1 + 1e-6
        else:
            self.depth_thresh = (1./255. / self.depth_scale) * 2. - 1 + 1e-6
        self.return_pcd = return_pcd

        if degradation is not None and scale_factors is not None:
            scaled_img_size = (int(self.img_size[0] / scale_factors[0]), int(self.img_size[1] / scale_factors[1]))
            degradation_fn = {
                "pil_nearest": PIL.Image.NEAREST,
                "pil_bilinear": PIL.Image.BILINEAR,
                "pil_bicubic": PIL.Image.BICUBIC,
                "pil_box": PIL.Image.BOX,
                "pil_hamming": PIL.Image.HAMMING,
                "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]
            self.degradation_transform = partial(TF.resize, size=scaled_img_size, interpolation=degradation_fn)
        else:
            self.degradation_transform = None
        self.condition_key = condition_key

        self.lidar_transform = get_lidar_transform(aug_config, split)
        self.anno_transform = get_anno_transform(aug_config, split) if condition_key in ['bbox', 'center'] else None
        self.view_transform = get_camera_transform(aug_config, split) if condition_key in ['camera'] else None

        self.prepare_data()

    def prepare_data(self):
        raise NotImplementedError

    def process_scan(self, range_img):
        range_img = np.where(range_img < 0, 0, range_img)

        if self.log_scale:
            # log scale
            range_img = np.log2(range_img + 0.0001 + 1)

        range_img = range_img / self.depth_scale
        range_img = range_img * 2. - 1.

        range_img = np.clip(range_img, -1, 1)
        range_img = np.expand_dims(range_img, axis=0)

        # mask
        range_mask = np.ones_like(range_img)
        range_mask[range_img < self.depth_thresh] = -1

        return range_img, range_mask

    @staticmethod
    def load_lidar_sweep(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_semantic_map(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_camera(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_annotation(*args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = dict()
        return example


class Txt2ImgIterableBaseDataset(IterableDataset):
    """
    Define an interface to make the IterableDatasets for text2img data chainable
    """
    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass
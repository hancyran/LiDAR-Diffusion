from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GeoConverter(nn.Module):
    def __init__(self, curve_length=4, bev_only=False, dataset_config=dict()):
        super().__init__()
        self.curve_length = curve_length
        self.coord_dim = 3 if not bev_only else 2
        self.convert_fn = self.batch_range2bev if bev_only else self.batch_range2xyz

        fov = dataset_config.fov
        self.fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
        self.fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
        self.fov_range = abs(self.fov_down) + abs(self.fov_up)  # get field of view total in rad
        self.depth_scale = dataset_config.depth_scale
        self.depth_min, self.depth_max = dataset_config.depth_range
        self.log_scale = dataset_config.log_scale
        self.size = dataset_config['size']
        self.register_conversion()

    def register_conversion(self):
        scan_x, scan_y = np.meshgrid(np.arange(self.size[1]), np.arange(self.size[0]))
        scan_x = scan_x.astype(np.float64) / self.size[1]
        scan_y = scan_y.astype(np.float64) / self.size[0]

        yaw = (np.pi * (scan_x * 2 - 1))
        pitch = ((1.0 - scan_y) * self.fov_range - abs(self.fov_down))

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('cos_yaw', torch.cos(to_torch(yaw)))
        self.register_buffer('sin_yaw', torch.sin(to_torch(yaw)))
        self.register_buffer('cos_pitch', torch.cos(to_torch(pitch)))
        self.register_buffer('sin_pitch', torch.sin(to_torch(pitch)))

    def batch_range2xyz(self, imgs):
        batch_depth = (imgs * 0.5 + 0.5) * self.depth_scale
        if self.log_scale:
            batch_depth = torch.exp2(batch_depth) - 1
        batch_depth = batch_depth.clamp(self.depth_min, self.depth_max)

        batch_x = self.cos_yaw * self.cos_pitch * batch_depth
        batch_y = -self.sin_yaw * self.cos_pitch * batch_depth
        batch_z = self.sin_pitch * batch_depth
        batch_xyz = torch.cat([batch_x, batch_y, batch_z], dim=1)

        return batch_xyz

    def batch_range2bev(self, imgs):
        batch_depth = (imgs * 0.5 + 0.5) * self.depth_scale
        if self.log_scale:
            batch_depth = torch.exp2(batch_depth) - 1
        batch_depth = batch_depth.clamp(self.depth_min, self.depth_max)

        batch_x = self.cos_yaw * self.cos_pitch * batch_depth
        batch_y = -self.sin_yaw * self.cos_pitch * batch_depth
        batch_bev = torch.cat([batch_x, batch_y], dim=1)

        return batch_bev

    def curve_compress(self, batch_coord):
        compressed_batch_coord = F.avg_pool2d(batch_coord, (1, self.curve_length))

        return compressed_batch_coord

    def forward(self, input):
        input = input / 2. + .5  # [-1, 1] -> [0, 1]

        input_coord = self.convert_fn(input)
        if self.curve_length > 1:
            input_coord = self.curve_compress(input_coord)

        return input_coord
